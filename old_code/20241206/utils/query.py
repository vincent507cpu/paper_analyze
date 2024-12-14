

import re
from typing import Dict, List, Union

from langchain_ollama.chat_models import ChatOllama
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)

def query_rewritten(query):
    return query

def translation_chn2eng(query: str, llm=llm) -> str:
    eng_query = '\n\n'
    
    prompt_chn_eng = """这是一个中文的查询，请将其尽可能准确地翻译成英文，不要解释、添加与翻译的原文无关的内容： {}
    
    翻译："""
    
    while '\n' in  eng_query:
        eng_query = llm.invoke(prompt_chn_eng.format(query)).content
                
    return eng_query

def translation_eng2chn(string: str, llm=llm) -> str:
    prompt_chn_eng = """这是一个中文的文本，请将其尽可能准确地翻译成英文，不要解释、添加与翻译的原文无关的内容： {}
    
    翻译："""
    return llm.invoke(prompt_chn_eng.format(string))