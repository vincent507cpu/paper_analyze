import re
import json
import os
import requests
from xml.etree import ElementTree as ET
from typing import List, Dict, Union

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field

from langchain_ollama.llms import OllamaLLM
llm = OllamaLLM(model="qwen2.5:1.5b")

keyword_extract_prompt = """你是一名资深编辑，需要对给定的查询语句进行以下处理：

1. 规范化查询语句：使其表达简洁、清晰。
2. 提取查询主体：确定查询的主要对象（可以是名词或名词性短语）。

请输出：规范化后的查询主体。

示例：
INPUT: Transformers 的编码器是如何设计的？
"OUTPUT": Transformers 的编码器

INPUT: LLaVa 的线性层起到了什么作用？
"OUTPUT": LLaVa 的线性层

INPUT: Attention 好像很厉害，请问它是干嘛的？
"OUTPUT": Attention

INPUT: {}
OUTPUT: 
"""

def translation_chn2eng(keywords: Union[List[str], str]) -> Dict:
    re_zh = re.compile('([\u4E00-\u9FA5]+)')
    eng_keywords = []
    prompt_chn_eng = """这是一个中文的查询，请将其尽可能准确地翻译成英文，不要解释、添加与翻译的原文无关的内容： {}
    
    翻译："""
    
    if isinstance(keywords, str):
        keywords = [keywords]
    for keyword in keywords:
        if re_zh.search(keyword):
            eng_keywords.append(llm.invoke(prompt_chn_eng.format(keyword)))
        else:
            eng_keywords.append(keyword)
            
    return eng_keywords

def translation_eng2chn(string: str) -> str:
    prompt_chn_eng = """这是一个中文的文本，请将其尽可能准确地翻译成英文，不要解释、添加与翻译的原文无关的内容： {}
    
    翻译："""
    return llm.invoke(prompt_chn_eng.format(' '.join(string)))

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
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
            arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]  # Extract the arXiv ID
            results.append({'arxiv_id': arxiv_id, 'title': title, 'summary': summary})
        return results
    else:
        return []  # Return an empty list on failure
    
def download_ingest_arxiv_paper(arxiv_id: str) -> None:
    print(arxiv_id, 'ingested, pdf file saved to', os.path.abspath(f"../../data/{arxiv_id}.pdf"))

def retrieve_query(query: str, k: int, vector_store: str = None):
    print(f'retrieved top {k} passages for query: {query}\ngenerating answer...')
    
def genrate_answer(query: str, retrieval: List[str]) -> str:
    prompt = f"""
    You are a professional AI assistant.
    You are given a query and a list of passages retrieved from the internet.
    Please generate an answer to the query based on the retrieved passages.
    
    Query: {query}
    Retrieved Passages:
    {retrieval}
    
    Answer:
    """
    # answer = llm.invoke(prompt)
    # return translation_eng2chn(answer)
    return "编码部分（encoders）由多层编码器(Encoder)组成（Transformer论文中使用的是6层编码器，这里的层数6并不是固定的，你也可以根据实验效果来修改层数）。同理，解码部分（decoders）也是由多层的解码器(Decoder)组成（论文里也使用了6层解码器）。每层编码器网络结构是一样的，每层解码器网络结构也是一样的。不同层编码器和解码器网络结构不共享参数。"

if __name__ == "__main__":
    original_query = input('请输入你的问题：')
    
    print('正在处理问题...')
    query_keywords = llm.invoke(keyword_extract_prompt.format(original_query))
    
    print('正在翻译问题...')
    keywords_eng = translation_chn2eng(query_keywords)
    
    print('正在搜索相关论文...')
    papers = query_arxiv(keywords_eng)
    
    print('正在下载论文并将论文载入数据库中...')
    for paper in papers:
        download_ingest_arxiv_paper(paper['arxiv_id'])
    
    print('正在检索数据库...')
    retrieve_query(original_query, 5)
    
    print('正在生成答案...')
    print(genrate_answer(original_query, None))