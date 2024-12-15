import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Union

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel, Field

from utils.data import query_arxiv_papers, rank_by_aggregated_reverse_value
from utils.prompts import (keyword_extraction_instruction,
                           query_clearification_instruction,
                           query_generation_instruction,
                           text_isRel_instruction,
                           get_contextualized_question_instruction)

# class IsRelOutput(BaseModel):
#     """Output schema for the relavency of the query and the content."""

#     Response: bool = Field(
#         description="Whether the query is related to the paper or not."
#     )
#     Reasoning: str = Field(
#         description="Reasoning behind the output."
#     )
# llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
# original_parser = PydanticOutputParser(pydantic_object=IsRelOutput)
# parser = OutputFixingParser.from_llm(parser=original_parser, llm=llm)

def translation_chn2eng(query: str, llm) -> str:
    eng_query = '\n\n'
    
    prompt_chn_eng = """这是一个中文的查询，请将其尽可能准确地翻译成英文，不要解释、添加与翻译的原文无关的内容： {}
    
    翻译："""
    
    while '\n' in  eng_query:
        eng_query = llm.invoke(prompt_chn_eng.format(query)).content
                
    return eng_query

def translation_eng2chn(string: str, llm) -> str:
    prompt_eng_chn = """There is an English text, please translate it as accurately as possible into Chinese, do not add irrelevant content: {}
    
    Translate: """
    return llm.invoke(prompt_eng_chn.format(string)).content

def query_rewritten(eng_query: str, llm) -> str:
    prompt = query_clearification_instruction + f"Please rewrite the query: ```{eng_query}```\n\nOUTPUT:\n"
    
    return llm.invoke(prompt).content

def multiple_query_generation(eng_query: str, llm):
    
    prompt = query_generation_instruction + f"Please generate five diverse questions based on the text: ```{eng_query}```\n\nOUTPUT:\n"
    
    return llm.invoke(prompt).content.split('\n')

def keywords_extraction(query_eng: str, llm) -> List[str]:
    prompt = keyword_extraction_instruction + f"Please extract 3 most important keywords based on the text: ```{query_eng}```\n\nOUTPUT:\n"
    print('关键词提取完成')
    return llm.invoke(prompt).content.split(';')

def is_relevant_check(query_eng: str, context: str, llm):
    template = f'\n1. content: {context}\n2. question: {query_eng}\n\nOUTPUT:\n'
    full_prompt = text_isRel_instruction + template
    response = llm.invoke(full_prompt).content
    # return parser.parse(llm.invoke(full_prompt).content).Response
    
    if 'True' in response:
        return True
    else:
        return False
    
def get_contextualized_question_prompt(instruction=get_contextualized_question_instruction):
    """
    定义一个用于改写用户问题的提示模板，确保问题在上下文中具有明确性。
    - 如果没有聊天历史，问题保持不变。
    - 如果有历史，尝试基于上下文进行指代消解。
    - 如果不够完整，补充缺乏的内容。
    """
    contextualize_question_prompt = ChatPromptTemplate([
        ("system", instruction),
        MessagesPlaceholder("chat_history"),
        ("human", "History: {chat_history}\nCurrent Question: {input}\nDoes it need anaphora resolution:")
    ])

    return contextualize_question_prompt

def intention_detection(query: str, llm) -> bool:
    prompt = (
        "Determine if the following text is a causual chat or an academic inquiry. Only response 'True' or 'False'.\n"
        "EXAMPLE:\n-----"
        "INPUT: Hello!\n"
        "OUTPUT: False\n"
        "INPUT: Shall we go for lunch?\n"
        "OUTPUT: False\n"
        "-----\n"
        "INPUT: Introduce the movie Inception\n"
        "OUTPUT: False\n"
        "-----\n"
        "INPUT: What is perceptron?\n"
        "OUTPUT: True\n"
        "INPUT: Introduce self-attention machansim\n"
        "OUTPUT: True\n"
        "INPUT: Why Llava performed so well?\n"
        "OUTPUT: True\n"
        "INPUT: How do multimodal language model work?\n"
        "OUTPUT: True\n"
        "INPUT: {}\n"
        "OUTPUT: "
    )
    response = llm.invoke(prompt.format(query))
    return eval(response.content)

def chn_chat(query: str, llm):
    response = llm.invoke(f'请回答如下问题：{query}\n回答：')
    return response.content

if __name__ == "__main__":
    # chn_query = "自注意力机制"
    # eng_query = translation_chn2eng(chn_query)
    # print(eng_query)
    # rewritten_query = query_rewritten(eng_query)
    # print(rewritten_query)
    
    eng_query = 'Self-attention mechanism'
    rewritten_query = query_rewritten(eng_query)
    # print(rewritten_query)
    # print()
    multiple_queries = multiple_query_generation(rewritten_query)
    # print(multiple_queries)
    
    queried_papers = []
    keywords = []
    for query in multiple_queries:
        # print(query)
        keywords = keywords_extraction(query)
        # print(keywords)
        papers = query_arxiv_papers(keywords, 5)
        # print(papers)
        print()
        queried_papers.append(papers)
        
    related_papers = rank_by_aggregated_reverse_value(rewritten_query, queried_papers)
    print(related_papers[:3])