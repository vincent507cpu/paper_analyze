import re
import math
import os
import requests
from typing import List, Union
from uuid import uuid4
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import jieba
import ollama
from langchain_core.documents import Document
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.pipe.OCRPipe import OCRPipe

def query_arxiv(search_term: str, max_results: int = 5) -> List[Union[str, dict]]:
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=all:{search_term}&start=0&max_results={max_results}"
    response = requests.get(base_url + query)
    
    if response.status_code == 200:
        # Parse the XML response
        root = ET.fromstring(response.content)
        results = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
            arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]  # Extract the arXiv ID
            
            try:
                download_arxiv_pdf(arxiv_id)
            except requests.exceptions.RequestException as e:
                print(f"下载失败: {e}")
                continue
            
            results.append({'arxiv_id': arxiv_id, 'title': title, 'abstract': abstract})
        return results
    else:
        return []  # Return an empty list on failure
    
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
    ################
    path = os.path.join(local_md_dir, f'{pdf_file_name}.pdf.md')
    ################
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
    
    print(f'提取 {pdf_file_name} 内容成功\n')
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

        # 提取正文内容
        start_pattern = r'#?\s*(1|I)?\.?\s*(Introduction|INTRODUCTION)'
        start_idx = re.search(start_pattern, content).start()
        end_pattern = r'#?\s*([A-Z]|\d+(\.\d+)*|[IVXLCDM]+)?\.?\s*(Reference|REFERENCE|References|REFERENCES)'
        end_idx = re.search(end_pattern, content).start()
        paras = [p.strip() for p in content[start_idx-1:end_idx].split('\n') if p]

        # 初始化解析过程
        # doc_id = str(uuid4())
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
    '''相关一般，没有实装'''
    pass

def filter_by_ppl():
    '''相关一般，没有实装'''
    pass

if __name__ == '__main__':
    download_arxiv_pdf('2305.06705')