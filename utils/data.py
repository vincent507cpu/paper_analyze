import re
import math
import os
import requests
import json
from collections import defaultdict
from typing import List, Union, Dict
from uuid import uuid4
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import jieba
import ollama
from tqdm import tqdm
from langchain_core.documents import Document
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.pipe.OCRPipe import OCRPipe
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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

# 查询 arXiv，按照摘要与查询的相似度进行排序
def query_arxiv(search_term: str) -> List[Union[str, dict]]:
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
    使用关键字查询arXiv，不使用第三方库。

    参数:
        keywords (list): 关键字字符串列表。
        n (int): 最小所需查询结果数量。

    返回:
        list: 包含论文元数据的字典列表，包含'title'、'abstract'和'arXiv ID'。
    """
    base_url = "http://export.arxiv.org/api/query"

    def fetch_results(query: str) -> List[Dict[str, str]]:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": n,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            return []

        # 解析XML响应
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(response.content)
        entries = root.findall("atom:entry", ns)
        results = []

        for entry in tqdm(entries, desc="Fetching results"):
            title = entry.find("atom:title", ns).text.strip()
            abstract = entry.find("atom:summary", ns).text.strip()
            arxiv_id = entry.find("atom:id", ns).text.strip().split("/")[-1]
            results.append({"title": title, "abstract": abstract, "arxiv_id": arxiv_id})

        return results

    results = []

    # 使用所有关键字查询
    query = " AND ".join(keywords)
    results = fetch_results(query)

    
    keywords.pop()  # 移除最后一个关键字
    query = " AND ".join(keywords)
    results = fetch_results(query)
    
    # 如果结果仍然不足，使用第一个关键字查询
    if len(results) < n and keywords:
        query = keywords[0]
        results = fetch_results(query)
    print('相关论文信息提取完成')
    return results


def rank_by_aggregated_reverse_value(search_term: str, subsets: List[List[Dict[str, str]]], exclude: List = []) -> List[Dict[str, str]]:
    """
    根据文献在多个子集中的排名倒数值总和进行排序。

    参数:
    - search_term: 查询字符串
    - subsets: 子集列表，每个子集是一个包含文献的列表，每篇文献包含 "arxiv_id"、"title" 和 "abstract"

    返回:
    - List[Dict[str, str]]: 按总分从高到低排序的文献列表
    """
    scores = defaultdict(float)  # 存储每篇文献的总分
    papers_metadata = {}         # 存储文献的元数据，避免重复

    # 遍历每个子集，计算倒数值并累加
    for subset in subsets:
        similarities = []
        unique_ids = set()
        
        for paper in subset:
            if paper["arxiv_id"] in unique_ids or paper['arxiv_id'] in exclude:
                continue
            unique_ids.add(paper["arxiv_id"])
            
            # 使用 Reranker 模型计算相似度
            inputs = tokenizer.encode_plus(
                search_term, paper["abstract"], return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                similarity_score = model(**inputs).logits.squeeze().item()

            similarities.append((paper["arxiv_id"], paper["title"], paper["abstract"], similarity_score))

        # 按相似度排序，并为每篇文献分配排名倒数值
        similarities = sorted(similarities, key=lambda x: x[3], reverse=True)
        for rank, (arxiv_id, title, abstract, _) in enumerate(similarities):
            reverse_rank = 1 / (rank + 1)  # 倒数值
            scores[arxiv_id] += reverse_rank  # 累加到总分
            if arxiv_id not in papers_metadata:
                papers_metadata[arxiv_id] = {"title": title, "abstract": abstract}

    # 按总分排序
    ranked_papers = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 构建输出数据
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
    下载指定 arXiv ID 的 PDF 文件到指定目录。

    :param arxiv_id: arXiv ID (例如 '2301.12345')
    :param save_dir: 保存目录路径
    """
    # 检查保存目录是否存在，如果不存在则创建
    paper_dir = os.path.abspath('caches/paper')
    os.makedirs(paper_dir, exist_ok=True)
    
    # 构造 PDF 文件的下载 URL
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    file_path = os.path.join(paper_dir, f"{arxiv_id}.pdf")
    
    if os.path.exists(file_path):
        print(f"PDF 文件已存在: {file_path}")
        return
    
    try:
        # 下载 PDF 文件
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 将 PDF 保存到指定路径
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"PDF 下载成功: caches/paper/{arxiv_id}.pdf")
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException()
        
def pdf_parser(pdf_file_name):
    print('当前正在解析的文件:', pdf_file_name)
    ## prepare env
    local_image_dir = str(Path(__file__).parent.parent.joinpath(f'caches/images'))
    local_pdf_dir = str(Path(__file__).parent.parent.joinpath(f'caches/paper'))
    local_md_dir = str(Path(__file__).parent.parent.joinpath(f'caches/md'))
    os.makedirs(local_image_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(local_md_dir, f'{pdf_file_name}.md')):
        print(f"PDF 文件已存在: {pdf_file_name}.md")
        return open(os.path.join(local_md_dir, f'{pdf_file_name}.md'), 'r').read()
    
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    image_dir = str(os.path.basename(local_image_dir))

    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(os.path.join(local_pdf_dir, pdf_file_name))   # read the pdf content


    pipe = OCRPipe(pdf_bytes, [], image_writer)

    # pipe.pipe_classify()
    pipe.pipe_analyze()
    pipe.pipe_parse()

    md_content = pipe.pipe_mk_markdown(
        image_dir, drop_mode=DropMode.NONE, md_make_mode=MakeMode.MM_MD
    )
    
    if isinstance(md_content, list):
        md_writer.write_string(f"{pdf_file_name}.md", "\n".join(md_content))
    else:
        md_writer.write_string(f"{pdf_file_name}.md", md_content)
    
    # print(f'提取 {pdf_file_name} 内容成功\n')
    return md_content

class DocumentParser:
    def __init__(self, vllm='llama3.2-vision'):
        """
        初始化 DocumentParser。
        """
        self.vllm = vllm  # 可选：用于处理图像或表格的模型名称
        
    def parse_document(self, title, abstract, arxiv_id, content):
        """
        按段落解析 PDF 内容。
        - 保留正文部分（Introduction 以后，不包含 Reference 和 Appendix）。
        - 按段抓取，先按照长度进行语义分块，保留 Section、上下段信息。图片、表格仅保留 Section 信息。
        # TODO: 拓扑结构信息

        :param title: 文档标题
        :param abstract: 文档摘要
        :param arxiv_id: 文档的 arXiv ID
        :param content: 文档的全文内容
        :return: 文档对象列表
        """
        print(f'{arxiv_id}\t{title} 开始解析...')

        # 清理标题和摘要
        title = ' '.join(t.strip() for t in title.split('\n'))
        abstract = ' '.join(p.strip() for p in abstract.split('\n'))

        try:
            # 提取正文内容
            start_pattern = r'#?\s*(1|I)?\.?\s*(Introduction|INTRODUCTION)'
            start_idx = re.search(start_pattern, content).start()
            end_pattern = r'#?\s*([A-Z]|\d+(\.\d+)*|[IVXLCDM]+)?\.?\s*(Reference|REFERENCE|References|REFERENCES)'
            end_idx = re.search(end_pattern, content).start()
            paras = [p.strip() for p in content[start_idx-1:end_idx].split('\n') if p]
        except AttributeError as e:
            return None

        # 初始化解析过程
        doc = []
        sec_pattern = r'^#?\s*([A-Z]|\d+(\.\d+)*|[IVXLCDM]+)\.\s*(.+)'  # 匹配章节标题模式

        # 添加摘要部分
        doc.append(Document(
            id=str(uuid4()),
            page_content=abstract,
            metadata={
                'arxiv_id': arxiv_id,
                'title': title,
                'type': 'text',
                'section': 'abstract',
                'previous': None,
                'next': None
            }
        ))

        cur_section = ""
        i = 0
        while i < len(paras):
            if len(doc):
                previous_id = doc[-1].id

            if re.match(sec_pattern, paras[i]):  # 检测章节标题
                print('正在解析：', paras[i])
                cur_section = ' '.join([e for e in re.match(sec_pattern, paras[i]).groups() if e])
                i += 1

            elif i < len(paras) - 2 and paras[i].startswith('![]') and (paras[i+1].startswith('Figure') or paras[i+1].startswith('Table')):  # 检测图片或表格
                file_name = paras[i].split('/')[-1].strip()[:-1]
                caption = paras[i+1] if (i + 1 < len(paras) and ('Fig' in paras[i+1] or 'Table' in paras[i+1])) else ""
                
                summary = self.summarize_table_image(file_name, caption)
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
                i += 2
                
            elif i < len(paras) - 2 and paras[i+1].startswith('![]') and (paras[i].startswith('Figure') or paras[i].startswith('Table')):
                caption = paras[i]
                file_name = paras[i+1].split('/')[-1].strip()[:-1] if (i + 1 < len(paras) and paras[i+1].startswith('![]')) else ""
                print(file_name)

                summary = self.summarize_table_image(file_name, caption)
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
                i += 2

            elif i < len(paras) - 2 and paras[i] == '$$':  # 检测公式
                cur_content = '$' + paras[i+1] + '$'
                doc.append(Document(
                    id=uuid4(),
                    page_content=self.summarize_equation(cur_content),
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
                i += 2

            elif 'References' in paras[i]:  # 检测参考文献部分，停止抓取
                break

            else:  # 处理普通文本段落
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

        # 更新每个文档的 next 指针
        for i in range(len(doc) - 1):
            doc[i].metadata['next'] = doc[i+1].id

        return doc
    
    def summarize_equation(self, equation):
        response = ollama.chat(
            model=self.vllm,
            messages=[
                {
                    'role': 'user',
                    'content': f'Please summarize a following equation based the given caption. It has to be descriptive and include the main points while avoid_ing any irrelevant details.\nequation: {equation}',
                }
            ]
        )
        return response['message']['content']
    
    def summarize_table_image(self, table_image, caption):
        
        print('正在解析：', table_image)
        response = ollama.chat(
            model=self.vllm,
            messages=[
                {
                    'role': 'user',
                    'content': f'Please summarize a following image or table based the given caption. It has to be descriptive and include the main points while avoid_ing any irrelevant details.\ncaption: {caption}',
                    'images': [Path(__file__).parent.parent.absolute().joinpath(f'caches/images/{table_image}')]
                }
            ]
        )
        return response['message']['content']
    
def structural_semantic_chunking():
    pass

class StructuralSemanticChunker:
    def __init__(self, model='BAAI/bge-small-en-v1.5') -> None:
        pass
    
    def construct_chunks(self, docs):
        pass
    
    def calculate_cosine_distances(self, sentences):
        pass
    
    def aggregate_chunks(self, chunks):
        pass

def filter_by_ppl():
    pass

if __name__ == '__main__':
    from pprint import pprint
    import json
    from pathlib import Path
    # download_arxiv_pdf('2305.06705')
    result = query_arxiv('The recent development of large language models', 100)
    # for res in result:
    #     # pprint(res, width=96)
    #     print(re.split('\n\s*', res['abstract']))
    #     break
    with open(Path(__file__).parent.parent.joinpath('data/arxiv_search_result.json'), 'w') as f:
        json.dump(result, f, indent=4)