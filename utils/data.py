import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union
from uuid import uuid4
from xml.etree import ElementTree as ET

import numpy as np
import ollama
import requests
import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.data.data_reader_writer import (FileBasedDataReader,
                                               FileBasedDataWriter)
from magic_pdf.pipe.OCRPipe import OCRPipe
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from utils.retrieval import BM25, Reranker, VectorStore
from utils.utils import flatten_list
from utils.query import translation_eng2chn, is_relevant_check
from utils.instructions import conversation_summarization_instruction

# 加载用于计算句子相似度的模型
model_name = 'BAAI/bge-small-en-v1.5' # 也可以替换为其他模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 计算两个句子之间的余弦相似度
def calculate_similarity(query: str, abstract: str) -> float:
    inputs = tokenizer([query, abstract], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model(**inputs)[0][0, 0]  # 第一个句子的嵌入表示
    similarity = torch.cosine_similarity(query_embedding.unsqueeze(0), query_embedding.unsqueeze(0))
    return similarity.item()

def query_arxiv(search_term: str) -> List[Union[str, dict]]:
    """
    查询 arXiv 并根据摘要与查询词的相似度对结果进行排序。
    
    功能描述：
    1. 通过 arXiv API 查询与搜索词相关的文献。
    2. 对每篇文献的摘要计算与搜索词的相似度。
    3. 根据相似度对文献结果进行排序，并返回前 100 条（如有）。
    
    参数：
        search_term (str): 用户输入的搜索词或查询短语，用于在 arXiv 中检索相关文献。
    
    返回值：
        List[Union[str, dict]]: 按相似度降序排列的文献列表。每个文献为字典形式，包含以下字段：
            - 'arxiv_id' (str): 文献在 arXiv 上的唯一标识符。
            - 'title' (str): 文献标题。
            - 'abstract' (str): 文献摘要，格式化为无换行符的连续文本。
        如果请求失败，返回一个空列表。
    
    主要步骤：
    1. 构造 arXiv API 的查询 URL。
    2. 使用 HTTP GET 请求获取 API 的 XML 响应。
    3. 解析 XML 数据，提取文献的标题、摘要和 arXiv ID。
    4. 计算每篇文献摘要与查询词的相似度。
    5. 按照相似度降序排列文献，并格式化摘要内容。
    6. 返回包含相似文献信息的字典列表。
    
    注意事项：
    - API 请求的最大返回结果数为 100。
    - 如果请求失败（HTTP 状态码非 200），函数将返回一个空列表。
    - 需要预定义 `calculate_similarity` 函数，该函数接收查询词和摘要文本，返回相似度分数。

    异常处理：
    - 如果 API 请求失败或响应格式错误，将返回空列表，避免程序崩溃。
    """
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=all:{search_term}&start=0&max_results=100"
    response = requests.get(base_url + query)
    
    if response.status_code == 200:
        # 解析 XML 响应
        root = ET.fromstring(response.content)
        results = []
        similarities = []
        
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
            arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]  # 提取 arXiv ID
            
            # 计算摘要与查询的相似度
            similarity_score = calculate_similarity(search_term, abstract)
            similarities.append((arxiv_id, title, abstract, similarity_score))
        
        # 按照相似度排序
        similarities = sorted(similarities, key=lambda x: x[3], reverse=True)
        
        # 返回与查询相似度高的文献
        for arxiv_id, title, abstract, _ in similarities:
            abstract = [s.strip().replace('\n', ' ') for s in re.split('\n\s+?', abstract)]
            results.append({'arxiv_id': arxiv_id, 'title': title, 'abstract': '\n'.join(abstract)})
        
        return results
    else:
        return []  # 请求失败时返回空列表

def query_arxiv_papers(keywords: List[str], n: int) -> List[Dict[str, str]]:
    """
    使用关键字查询 arXiv API 并提取论文元数据（不依赖第三方库）。

    功能描述：
    - 根据多个关键字构建查询字符串，通过 arXiv API 搜索相关论文。
    - 若返回的结果数量不足，逐步减少关键字进行查询，直到满足最低结果数量要求或耗尽关键字。
    - 返回每篇论文的标题、摘要和 arXiv ID。

    参数：
        keywords (List[str]): 查询的关键字列表。
        n (int): 所需的最小查询结果数量。

    返回：
        List[Dict[str, str]]: 包含论文元数据的字典列表，每个字典包含以下字段：
            - "title": 论文标题。
            - "abstract": 论文摘要。
            - "arxiv_id": 论文的 arXiv ID。

    示例：
    ```
    papers = query_arxiv_papers(["deep learning", "transformer"], 10)
    for paper in papers:
        print(f"Title: {paper['title']}")
        print(f"Abstract: {paper['abstract']}")
        print(f"arXiv ID: {paper['arxiv_id']}")
    ```

    """

    # 定义 API 基础 URL
    base_url = "http://export.arxiv.org/api/query"

    def fetch_results(query: str) -> List[Dict[str, str]]:
        """
        通过指定的查询字符串调用 arXiv API，并提取结果。

        参数：
            query (str): 查询字符串。

        返回：
            List[Dict[str, str]]: 提取的论文元数据列表。
        """
        params = {
            "search_query": query,
            "start": 0,
            "max_results": n,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        # 发起请求
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"请求失败，状态码：{response.status_code}")
            return []

        # 解析 XML 响应
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(response.content)
        entries = root.findall("atom:entry", ns)
        results = []

        # 提取论文元数据
        for entry in entries:
            title = entry.find("atom:title", ns).text.strip()
            abstract = entry.find("atom:summary", ns).text.strip()
            arxiv_id = entry.find("atom:id", ns).text.strip().split("/")[-1]
            results.append({"title": title, "abstract": abstract, "arxiv_id": arxiv_id})

        return results

    results = []

    # 使用所有关键字构建查询
    query = " AND ".join(keywords)
    results.extend(fetch_results(query))

    # 若结果不足，逐步减少关键字查询
    while len(results) < n and len(keywords) > 1:
        keywords.pop()  # 移除最后一个关键字
        query = " AND ".join(keywords)
        results.extend(fetch_results(query))

    # 若结果仍然不足，使用单个关键字进行查询
    if len(results) < n and keywords:
        query = keywords[0]
        results.extend(fetch_results(query))

    print("相关论文信息提取完成")
    return results


def rank_by_aggregated_reverse_value(search_term: str, subsets: List[List[Dict[str, str]]], exclude: List = []) -> List[Dict[str, str]]:
    """
    根据文献在多个子集中的排名倒数值总和对文献进行排序。

    功能描述：
    - 对多个子集中包含的文献计算与查询词的相似度。
    - 使用排名倒数值（1 / rank）作为每篇文献的评分，并累加文献在不同子集中的评分。
    - 返回按评分从高到低排序的文献列表。

    参数：
        search_term (str): 查询词，用于计算与文献摘要的相似度。
        subsets (List[List[Dict[str, str]]]): 文献子集列表，每个子集是一个包含文献的列表，每篇文献包含以下字段：
            - "arxiv_id": 文献的唯一标识符。
            - "title": 文献标题。
            - "abstract": 文献摘要。
        exclude (List): 要排除的文献 ID 列表（默认值为空）。

    返回：
        List[Dict[str, str]]: 按评分从高到低排序的文献列表，每篇文献包含以下字段：
            - "arxiv_id": 文献的唯一标识符。
            - "title": 文献标题。
            - "abstract": 文献摘要。

    示例用法：
    ```
    ranked_papers = rank_by_aggregated_reverse_value("deep learning", subsets)
    ```

    主要流程：
    1. **初始化数据结构：**
       - `scores`：用于存储每篇文献的总评分。
       - `papers_metadata`：存储文献的元数据（标题和摘要），避免重复存储。
    2. **计算相似度：**
       - 遍历每个子集中的文献，计算其与查询词的相似度分数。
       - 按相似度对子集排序后，为每篇文献分配排名倒数值（1 / rank），并累加到总分。
    3. **排序与输出：**
       - 根据总评分对所有文献排序。
       - 构建输出文献列表，包括文献的 ID、标题和摘要。

    异常处理：
        - 假设输入数据符合规范，不包含特殊异常处理逻辑。

    注意：
        - 该方法依赖预加载的模型 `model` 和分词器 `tokenizer`，需要确保在函数调用前正确初始化。

    """
    # 初始化分数和元数据存储
    scores = defaultdict(float)  # 每篇文献的总评分
    papers_metadata = {}         # 文献的元数据

    # 遍历所有子集
    for subset in subsets:
        similarities = []  # 存储当前子集的文献相似度信息
        unique_ids = set()  # 当前子集中已经处理的文献 ID

        for paper in subset:
            # 跳过已处理或需要排除的文献
            if paper["arxiv_id"] in unique_ids or paper["arxiv_id"] in exclude:
                continue
            unique_ids.add(paper["arxiv_id"])  # 添加到已处理集合

            # 计算查询词与文献摘要的相似度
            inputs = tokenizer.encode_plus(
                search_term, paper["abstract"], return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                similarity_score = model(**inputs).logits.squeeze().item()  # 获取相似度分数

            similarities.append((paper["arxiv_id"], paper["title"], paper["abstract"], similarity_score))

        # 对当前子集的文献按相似度降序排序
        similarities = sorted(similarities, key=lambda x: x[3], reverse=True)

        # 计算排名倒数值并累加到总分
        for rank, (arxiv_id, title, abstract, _) in enumerate(similarities):
            reverse_rank = 1 / (rank + 1)  # 计算倒数值
            scores[arxiv_id] += reverse_rank  # 累加到总评分
            if arxiv_id not in papers_metadata:
                papers_metadata[arxiv_id] = {"title": title, "abstract": abstract}

    # 根据总评分降序排序
    ranked_papers = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 构建结果列表
    result = []
    for arxiv_id, _ in ranked_papers:
        abstract = papers_metadata[arxiv_id]["abstract"]
        title = papers_metadata[arxiv_id]["title"]
        abstract = [s.strip().replace('\n', ' ') for s in re.split('\n\s+?', abstract)]
        result.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": '\n'.join(abstract),
        })

    return result

def download_arxiv_pdf(arxiv_id: str):
    """
    下载指定 arXiv ID 的 PDF 文件到缓存目录。

    功能描述：
    - 根据给定的 arXiv ID 构造 PDF 下载链接。
    - 检查本地缓存目录中是否已存在该文件，避免重复下载。
    - 如果文件不存在，下载 PDF 文件并保存到本地。

    参数：
        arxiv_id (str): arXiv 论文的唯一标识符（例如 '2301.12345'）。

    返回值：
        None

    异常：
        - 如果下载失败或网络请求出错，将抛出 `requests.exceptions.RequestException`。

    文件保存位置：
    - 缓存目录 `caches/paper/{arxiv_id}.pdf`

    示例用法：
    ```
    download_arxiv_pdf("2301.12345")
    ```

    主要流程：
    1. **设置缓存目录：**
       - 检查保存目录 `caches/paper` 是否存在，如果不存在则创建。
    2. **检查文件是否已存在：**
       - 如果 PDF 文件已存在，则跳过下载步骤。
    3. **下载 PDF 文件：**
       - 根据构造的 URL 下载 PDF 文件。
       - 使用流式写入将 PDF 内容保存到本地。
    """
    # 设置保存目录路径
    paper_dir = os.path.abspath('caches/paper')
    os.makedirs(paper_dir, exist_ok=True)  # 如果目录不存在则创建

    # 构造 PDF 文件的下载 URL 和保存路径
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"  # 构造下载链接
    file_path = os.path.join(paper_dir, f"{arxiv_id}.pdf")  # 构造保存路径

    # 检查 PDF 文件是否已存在
    if os.path.exists(file_path):
        print(f"PDF 文件已存在: {file_path}")
        return

    try:
        # 发起 HTTP 请求下载 PDF 文件
        response = requests.get(pdf_url, stream=True)  # 使用流式下载以节省内存
        response.raise_for_status()  # 检查 HTTP 响应状态码是否成功

        # 将 PDF 文件写入本地
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):  # 每次写入 8KB 数据
                file.write(chunk)

        # 下载成功的提示信息
        print(f"PDF 下载成功: caches/paper/{arxiv_id}.pdf")
    except requests.exceptions.RequestException as e:
        # 抛出请求异常
        raise requests.exceptions.RequestException(f"PDF 下载失败: {pdf_url}") from e
        
def pdf_parser(pdf_file_name):
    """
    解析 PDF 文件并生成 Markdown 文件，同时提取图片资源。

    功能描述：
    - 解析指定的 PDF 文件，提取文本内容和图片资源，并生成 Markdown 文件。
    - 如果 PDF 文件已被解析过，则直接读取并返回缓存的 Markdown 文件内容。

    参数：
        pdf_file_name (str): 要解析的 PDF 文件名（包含扩展名）。

    返回值：
        str: 解析后的 Markdown 内容字符串。如果文件已存在，则直接返回缓存内容。

    主要流程：
    1. **准备环境和目录：**
       - 创建用于存储图片资源的目录（`caches/images`）。
       - 指定 PDF 文件的存储目录（`caches/paper`）和 Markdown 文件的存储目录（`caches/md`）。
    2. **检查 Markdown 缓存：**
       - 如果指定的 PDF 文件已被解析过，直接返回对应的 Markdown 文件内容。
    3. **初始化写入器和读取器：**
       - 使用 `FileBasedDataWriter` 初始化图片和 Markdown 文件写入器。
       - 使用 `FileBasedDataReader` 读取 PDF 文件内容。
    4. **解析 PDF 文件：**
       - 使用 `OCRPipe` 工具分析 PDF 文件并提取内容。
       - 调用 `pipe_analyze` 方法分析 PDF 文件结构。
       - 调用 `pipe_parse` 方法提取文本和图片。
    5. **生成 Markdown 文件：**
       - 使用 `pipe_mk_markdown` 方法生成 Markdown 内容。
       - 将 Markdown 文件写入缓存目录。
    6. **返回解析结果：**
       - 返回解析后的 Markdown 内容字符串。

    异常处理：
    - 代码假设所有外部依赖（如 `FileBasedDataWriter`、`FileBasedDataReader`、`OCRPipe` 等）已正确实现。
    - 如果文件读取或写入失败，程序可能会抛出异常，需在调用方处理。

    示例用法：
    ```
    content = pdf_parser("example.pdf")
    print(content)
    ```
    """
    print('当前正在解析的文件:', pdf_file_name)

    ## 第一步：准备环境和目录
    # 设置图片缓存目录、PDF 文件目录和 Markdown 文件目录
    local_image_dir = str(Path(__file__).parent.parent.joinpath(f'caches/images'))
    local_pdf_dir = str(Path(__file__).parent.parent.joinpath(f'caches/paper'))
    local_md_dir = str(Path(__file__).parent.parent.joinpath(f'caches/md'))
    os.makedirs(local_image_dir, exist_ok=True)  # 确保图片目录存在

    # 检查 Markdown 文件是否已存在
    md_file_path = os.path.join(local_md_dir, f'{pdf_file_name}.md')
    if os.path.exists(md_file_path):
        print(f"PDF 文件已存在: {pdf_file_name}.md")
        return open(md_file_path, 'r').read()  # 直接读取 Markdown 文件并返回

    ## 第二步：初始化写入器和读取器
    # 创建图片和 Markdown 的写入器
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    image_dir = str(os.path.basename(local_image_dir))  # 获取图片目录的相对路径

    # 读取 PDF 文件内容
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(os.path.join(local_pdf_dir, pdf_file_name))  # 读取 PDF 文件为字节流

    ## 第三步：解析 PDF 文件
    # 初始化 OCR 管道并传入 PDF 内容
    pipe = OCRPipe(pdf_bytes, [], image_writer)

    # 分析 PDF 文件结构（注释掉了分类步骤，可根据需要启用）
    # pipe.pipe_classify()
    pipe.pipe_analyze()  # 分析 PDF 文件
    pipe.pipe_parse()    # 提取 PDF 内容

    # 生成 Markdown 内容
    md_content = pipe.pipe_mk_markdown(
        image_dir, drop_mode=DropMode.NONE, md_make_mode=MakeMode.MM_MD
    )

    ## 第四步：写入 Markdown 文件
    # 将生成的 Markdown 内容写入缓存目录
    if isinstance(md_content, list):  # 如果内容是列表，拼接为字符串
        md_writer.write_string(f"{pdf_file_name}.md", "\n".join(md_content))
    else:  # 如果内容是字符串，直接写入
        md_writer.write_string(f"{pdf_file_name}.md", md_content)

    ## 第五步：返回 Markdown 内容
    return md_content

class DocumentParser:
    def __init__(self, vllm='llama3.2-vision'):
        """
        初始化 DocumentParser 类。

        :param vllm: 用于解析图像或表格的语言模型名称，默认为 'llama3.2-vision'。
        """
        self.vllm = vllm  # 模型名称，用于生成摘要或解析内容
        
    def parse_document(self, title, abstract, arxiv_id, content):
        """
        按段落解析 PDF 文档内容。
        功能：
        1. 保留正文部分（从 "Introduction" 开始，不包括 "Reference" 和 "Appendix" 部分）。
        2. 按段处理内容，基于长度和语义信息进行分块。
        3. 对于图像和表格，仅保留章节信息；对于正文段落，保留上下文信息。
        4. TODO: 添加拓扑结构的解析。

        :param title: 文档标题。
        :param abstract: 文档摘要。
        :param arxiv_id: 文档的 arXiv ID。
        :param content: 文档全文内容。
        :return: 解析后的文档对象列表，或 None（如果解析失败）。
        """
        print(f'{arxiv_id}\t{title} 开始解析...')

        # 清理标题和摘要，去掉多余的换行符和空格
        title = ' '.join(t.strip() for t in title.split('\n'))
        abstract = ' '.join(p.strip() for p in abstract.split('\n'))

        try:
            # 正则表达式提取正文内容起止位置
            start_pattern = r'#?\s*(1|I)?\.?\s*(Introduction|INTRODUCTION)'  # 匹配正文开始位置
            start_idx = re.search(start_pattern, content).start()
            end_pattern = r'#?\s*([A-Z]|\d+(\.\d+)*|[IVXLCDM]+)?\.?\s*(Reference|REFERENCE|References|REFERENCES)'  # 匹配正文结束位置
            end_idx = re.search(end_pattern, content).start()
            # 按换行符分割正文内容为段落，并去掉多余空白
            paras = [p.strip() for p in content[start_idx-1:end_idx].split('\n') if p]
        except AttributeError:
            # 如果未找到开始或结束位置，返回 None
            return None

        # 初始化解析结果列表
        doc = []
        sec_pattern = r'^#?\s*([A-Z]|\d+(\.\d+)*|[IVXLCDM]+)\.\s*(.+)'  # 正则表达式匹配章节标题

        # 添加摘要部分到文档对象
        doc.append(Document(
            id=str(uuid4()),  # 生成唯一 ID
            page_content=abstract,  # 内容为摘要
            metadata={
                'arxiv_id': arxiv_id,  # 文档 arXiv ID
                'title': title,  # 文档标题
                'type': 'text',  # 类型为文本
                'section': 'abstract',  # 章节为摘要
                'previous': None,  # 无前一段落
                'next': None  # 暂时无下一段落
            }
        ))

        cur_section = ""  # 当前章节标题
        i = 0  # 当前段落索引
        while i < len(paras):
            # 如果文档对象列表非空，记录上一段落 ID
            if len(doc):
                previous_id = doc[-1].id

            # 检测章节标题
            if re.match(sec_pattern, paras[i]):
                print('正在解析章节标题：', paras[i])
                cur_section = ' '.join([e for e in re.match(sec_pattern, paras[i]).groups() if e])
                i += 1

            # 检测图片或表格
            elif i < len(paras) - 2 and paras[i].startswith('![]') and (paras[i+1].startswith('Figure') or paras[i+1].startswith('Table')):
                file_name = paras[i].split('/')[-1].strip()[:-1]  # 提取文件名
                caption = paras[i+1] if (i + 1 < len(paras) and ('Fig' in paras[i+1] or 'Table' in paras[i+1])) else ""
                summary = self.summarize_table_image(file_name, caption)  # 调用方法生成图片或表格摘要
                doc.append(Document(
                    id=uuid4(),
                    page_content=summary,
                    metadata={
                        'id_': str(uuid4()),
                        'arxiv_id': arxiv_id,
                        'title': title,
                        'type': 'image/table',
                        'section': cur_section,
                        'previous': previous_id,
                        'next': None
                    }
                ))
                i += 2  # 跳过图片和表格的两段内容

            # 检测公式
            elif i < len(paras) - 2 and paras[i] == '$$':
                cur_content = '$' + paras[i+1] + '$'  # 组合公式
                doc.append(Document(
                    id=uuid4(),
                    page_content=self.summarize_equation(cur_content),  # 调用方法生成公式摘要
                    metadata={
                        'id_': str(uuid4()),
                        'arxiv_id': arxiv_id,
                        'title': title,
                        'type': 'equation',
                        'section': cur_section,
                        'previous': previous_id,
                        'next': None
                    }
                ))
                i += 2  # 跳过公式内容

            # 检测参考文献部分，停止解析
            elif 'References' in paras[i]:
                break

            # 普通文本段落
            else:
                doc.append(Document(
                    id=uuid4(),
                    page_content=paras[i].strip(),
                    metadata={
                        'id_': str(uuid4()),
                        'arxiv_id': arxiv_id,
                        'title': title,
                        'type': 'text',
                        'section': cur_section,
                        'previous': previous_id,
                        'next': None
                    }
                ))
                i += 1

        # 更新每个文档对象的 next 指针
        for i in range(len(doc) - 1):
            doc[i].metadata['next'] = doc[i+1].id

        return doc

    def summarize_equation(self, equation):
        """
        根据公式内容生成摘要。

        :param equation: 待摘要的公式内容。
        :return: 公式摘要字符串。
        """
        response = ollama.chat(
            model=self.vllm,
            messages=[
                {
                    'role': 'user',
                    'content': f'Please summarize the following equation based on the given caption. It has to be descriptive and include the main points while avoiding any irrelevant details.\nEquation: {equation}',
                }
            ]
        )
        return response['message']['content']

    def summarize_table_image(self, table_image, caption):
        """
        根据图片或表格内容生成摘要。

        :param table_image: 图片或表格文件名。
        :param caption: 图片或表格说明。
        :return: 图片或表格的摘要字符串。
        """
        print('正在解析图片或表格：', table_image)
        response = ollama.chat(
            model=self.vllm,
            messages=[
                {
                    'role': 'user',
                    'content': f'Please summarize the following image or table based on the given caption. It has to be descriptive and include the main points while avoiding any irrelevant details.\nCaption: {caption}',
                    'images': [Path(__file__).parent.parent.absolute().joinpath(f'caches/images/{table_image}')]
                }
            ]
        )
        return response['message']['content']
    
def query_single_question_in_stores(multiple_queries: List[str], 
                                    vector_store: VectorStore, 
                                    sparse_store: BM25, 
                                    reranker: Reranker,
                                    llm: ChatOllama):
    top_k = 3
        
    vector_store_retrieval, bm25_retrieval, retrieval_result = [], [], []
    for query in multiple_queries:
        vector_store_retrieval.extend(vector_store.vector_store_query(query, top_k))
        bm25_retrieval.extend([sparse_store.bm25_rank_documents(query, top_k)])
    retrieval_result.extend([p.page_content for p in vector_store_retrieval])
    retrieval_result.extend(bm25_retrieval)
    
    # 展平嵌套列表
    retrieval_result = flatten_list(retrieval_result)

    # print(retrieval_result)  # 调试打印

    retrieval_result = reranker.rerank(multiple_queries[0], retrieval_result)
    rerank_result = []
    for i in range(len(retrieval_result)):
        if is_relevant_check(multiple_queries[0], retrieval_result[i], llm):
            rerank_result.append(retrieval_result[i])
        if len(rerank_result) >= top_k:
            break
            
    if rerank_result:
        print('\n检索到相关文档，根据检索到的文档生成回答：')
        response = llm.invoke(
        f"Please answer the following question based on the following context by user:\nQuestion: {multiple_queries[0]}\nContext:\n" + \
        "\n".join(rerank_result)
    ).content
        response = translation_eng2chn(response, llm)
        qa_summary = llm.invoke(conversation_summarization_instruction.format(multiple_queries[0], response)).content
        stores_updating_summary(qa_summary, sparse_store, vector_store)
        return response
    else:
        return None
    
def stores_updating_summary(summary: str, sparse_store: BM25, vector_store: VectorStore):
    """
    更新稀疏存储（BM25）和向量存储中的文档信息。

    该函数根据提供的摘要创建一个新文档，分配相应的元数据，并将该文档添加到稀疏存储（如 BM25）和向量存储中。

    :param summary: str
        要添加的新文档的文本摘要。
    :param sparse_store: BM25
        用于存储文档的稀疏存储实例（如 BM25）。
    :param vector_store: VectorStore
        用于存储文档向量的向量存储实例，以便进行相似性搜索。
    """
    # 创建一个新的文档
    new_document = Document(page_content=summary, 
                            metadata={'arxiv_id': None,
                            'title': None,
                            'type': 'text',
                            'section': 'qa_pair_summary',
                            'previous': None,
                            'next': None})

    # 添加到向量存储
    vector_store.vector_store_add_documents([[new_document]])
    vector_store.vector_store_save()
    # 添加到 BM25 存储
    sparse_store.bm25_add_documents([{'text': new_document.page_content,
                                'metadata': new_document.metadata}])
    sparse_store.bm25_save()

class StructuralSemanticChunker:
    def __init__(self, model='BAAI/bge-small-en-v1.5') -> None:
        """
        初始化 StructuralSemanticChunker。

        :param model: 用于嵌入的模型名称，默认为 'BAAI/bge-small-en-v1.5'
        """
        self.embed = HuggingFaceEmbeddings(model_name=model)

    def calculate_cosine_distances(self, sentences):
        """
        计算句子之间的余弦距离。

        :param sentences: 包含句子及其嵌入的列表
        :return: 余弦距离列表和更新后的句子列表
        """
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]['combined_sentence_embedding']
            embedding_next = sentences[i + 1]['combined_sentence_embedding']

            # 计算余弦相似度
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

            # 转换为余弦距离
            distance = 1 - similarity

            # 将余弦距离添加到列表中
            distances.append(distance)

            # 将距离存储在字典中
            sentences[i]['distance_to_next'] = distance

        # 可选：处理最后一个句子
        # sentences[-1]['distance_to_next'] = None  # 或一个默认值

        return distances, sentences

    def chunking(self, docs):
        """
        将文档分割成语义块。

        :param docs: 包含句子的文档
        :return: 语义块列表
        """
        # 初始化起始索引
        start_index = 0

        # 创建一个列表来保存分组的句子
        chunks = []

        # 使用正则表达式分割文档
        docs = re.split(r'(?<=[.?!])\s+', docs)

        # 获取句子的嵌入
        embeddings = self.embed.embed_documents([x['combined_sentence'] for x in docs])
        for i, sentence in enumerate(docs):
            sentence['combined_sentence_embedding'] = embeddings[i]

        # 计算句子之间的余弦距离
        distances, sentences = self.calculate_cosine_distances(sentences)

        # 获取断点距离阈值
        breakpoint_percentile_threshold = 95
        breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)  # 如果需要更多块，降低百分位数截止

        # 获取超过阈值的距离索引
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]  # 超过阈值的断点索引

        # 遍历断点以切割句子
        for index in indices_above_thresh:
            # 结束索引是当前断点
            end_index = index

            # 从当前起始索引到结束索引切割句子
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunks.append(combined_text)

            # 更新下一组的起始索引
            start_index = index + 1

        # 如果还有剩余的句子，处理最后一组
        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunks.append(combined_text)

        return chunks

class PerplexityFilter:
    def __init__(self, model_name='Qwen/Qwen2.5-0.5B-Instruct', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化 PerplexityFilter。

        :param model_name: 用于计算困惑度的模型名称，默认为 'Qwen/Qwen2.5-0.5B-Instruct'
        :param device: 计算设备，默认为 'cuda' 如果可用，否则为 'cpu'
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device

    def calculate_perplexity(self, sentence):
        """
        计算句子的困惑度。

        :param sentence: 句子
        :return: 句子的困惑度
        """
        # Tokenize 输入句子
        encodings = self.tokenizer(sentence, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)

        # 初始化变量
        nlls = []
        seq_len = input_ids.size(1)

        # 遍历每个时间步，逐字计算困惑度
        for i in range(1, seq_len):
            # 当前时间步的输入序列
            input_ids_step = input_ids[:, :i]
            target_id = input_ids[:, i]  # 目标 token 是下一个字

            with torch.no_grad():
                # 获取模型输出 logits
                outputs = self.model(input_ids_step)
                logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
                next_token_logits = logits[:, -1, :]  # 只取最后一个 token 的预测分布

                # 计算目标 token 的概率
                probs = torch.softmax(next_token_logits, dim=-1)
                target_prob = probs[:, target_id].squeeze()  # 取出目标 token 的概率

                # 计算负对数似然 (NLL)
                nll = -torch.log(target_prob)
                nlls.append(nll)

        # 计算整句困惑度
        avg_nll = torch.stack(nlls).mean()
        perplexity = torch.exp(avg_nll)

        return perplexity.item()

    def filter_sentences_by_perplexity(self, document, threshold_percentage=20):
        """
        过滤掉文档中困惑度最大的 20% 的句子，并保留原始顺序。

        :param document: 文档内容
        :param threshold_percentage: 过滤掉的句子比例，默认为 20%
        :return: 过滤后的句子列表
        """
        # 将文档分割成句子
        sentences = re.split(r'(?<=[.?!])\s+', document)

        # 计算每个句子的困惑度
        perplexities = [self.calculate_perplexity(sentence) for sentence in sentences]

        # 将句子和困惑度打包成元组列表
        sentence_perplexity_pairs = list(zip(sentences, perplexities))

        # 按困惑度排序
        sentence_perplexity_pairs.sort(key=lambda x: x[1])

        # 计算需要过滤掉的句子数量
        num_sentences_to_filter = max(1, int(len(sentences) * threshold_percentage / 100))

        # 获取困惑度最小的句子
        filtered_sentence_perplexity_pairs = sentence_perplexity_pairs[:-num_sentences_to_filter]

        # 提取句子并保留原始顺序
        filtered_sentences = [pair[0] for pair in filtered_sentence_perplexity_pairs]

        # 重新组合句子以保留原始顺序
        original_order_sentences = []
        for sentence in sentences:
            if sentence in filtered_sentences:
                original_order_sentences.append(sentence)

        return original_order_sentences
    
if __name__ == '__main__':
    import json
    from pathlib import Path
    from pprint import pprint

    # download_arxiv_pdf('2305.06705')
    result = query_arxiv('The recent development of large language models', 100)
    # for res in result:
    #     # pprint(res, width=96)
    #     print(re.split('\n\s*', res['abstract']))
    #     break
    with open(Path(__file__).parent.parent.joinpath('data/arxiv_search_result.json'), 'w') as f:
        json.dump(result, f, indent=4)