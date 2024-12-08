# import os
# import re
# from pathlib import Path

# # local_md_dir = Path(__file__).parent.parent.joinpath(f'caches/images')
# # # print(os.path.join(local_md_dir, f'{1234}.pdf.md'))
# # # print(type(local_md_dir))
# # print(local_md_dir)

# # # print(r'#?\s*\d?\s*Introduction'.upper())
# # print(os.path.abspath('../store'))
# a = Path(__file__)
# print(a)
# print(a.parent.parent.joinpath('store'))

# # content = '# 1. INTRODUCTION  '
# # content = '# II.BACKGROUND  '
# # content = '# A. Research Hypotheses  '
# content = '# REFERENCES  '
# # start_pattern = r'^#?\s*(1|I)?\.?\s*(Introduction|INTRODUCTION)'

# # start_idx = re.search(start_pattern, content).start()
# # start_pattern = r'^#?\s*([A-Z]|\d+(\.\d+)*|[IVXLCDM]+)\.\s*(.+)'
# start_pattern = r'#?\s*([A-Z]|\d+(\.\d+)*|[IVXLCDM]+)?\.?\s*Reference(s)?'
# start_idx = re.search(start_pattern, content, re.IGNORECASE).start()
# print(start_idx)
# print(re.search(start_pattern, content, re.IGNORECASE).groups())

# print(Path(__file__).parent.parent.joinpath('store'))

# import logging

# # 配置日志等级
# logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s - %(funcName)s - line %(lineno)d - %(levelname)s - %(message)s')

# # 测试不同等级日志
# logging.debug("This is a debug message")
# logging.info("This is an info message")
# logging.warning("This is a warning message")
# logging.error("This is an error message")
# logging.critical("This is a critical message")

from langchain_ollama.chat_models import ChatOllama
llm = ChatOllama(model="qwen2.5:1.5b")

def translation_chn2eng(query: str, llm=llm) -> str:
    eng_query = '\n\n'
    
    prompt_chn_eng = """这是一个中文的查询，请将其尽可能准确地翻译成英文，不要解释、添加与翻译的原文无关的内容： {}
    
    翻译："""
    
    while '\n' in  eng_query:
        eng_query = llm.invoke(prompt_chn_eng.format(query)).content
                
    return eng_query

print(translation_chn2eng('如何用python实现一个简单的文本分类模型？'))