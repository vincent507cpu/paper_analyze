import json
import os
import time
import warnings

import pandas as pd
import requests
from tqdm import tqdm
from pathlib import Path
from urllib3.exceptions import ProtocolError
from http.client import RemoteDisconnected

from config import Config

query_clearification_instruction = (
    "Rewrite the given query to make it more concise, specific and logical while correct grammatical mistakes or typos in style of question: ```{{query}}```\n\n"
    "EXAMPLE:\n------------------------\nINPUT:\n"
    "Please rewrite the query: ```gradient descent optimize```\n"
    "\nOUTPUT:\n"
    "How does gradient descent perform optimization?\n\n"
    
    "Please rewrite the query: ```'Stereo Anywhere' aplication```\n"
    "\nOUTPUT:\n"
    "What is the application of 'Stereo Anywhere'?"
)

query_generation_instruction = (
    f"Please generate five diverse questions based on the given text: ```{{text}}```. "
    '"Diverse" means the questions should be different from each other in terms of wording, level of description clarity, completeness, and phrasing. For example, some questions can be very detailed and specific, while others can be more general and open-ended. '
    "The questions may even include intentional misspellings, grammatical errors, or misleading words to add to the diversity. "
    "Ensure that the questions are distinct and cover a range of aspects related to the text.\n\n"
    "EXAMPLE:\n------------------------\nINPUT:\n"
    "Please generate five diverse questions based on the text: ```Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail```\n"
    
    "\nOUTPUT:\n"
    "Can you explain what is meant by 'zero-shot' in the context of deep stereo matching and how it contributes to the robustness of the 'Stereo Anywhere' method?\n"
    "How does the 'Stereo Anywhere' technique ensure robust performance even in scenarios where traditional stereo or monocular methods fail?\n"
    "In what ways does 'Stereo Anywhere' differ from other deep stereo matching techniques, particularly in terms of robustness and zero-shot learning capabilities?\n"
    "Can u giv som examples of real-world aplications where 'Stereo Anywhere' would be really usefull, especialy in situtations were other methods might fail?\n"
    "What are the potential drawbacks or obstacles of the 'Stereo Anywhere' approach, and how might these be overcome in future research to ensure even greater reliability and accuracy in stereo matching?\n\n"
)

keyword_extraction_instruction = (
    "Extract the 3 most important keywords from a given query, place them in descending order of their importance, and separate them by ';': ```text```\n\n"
    "EXAMPLE:\n------------------------\nINPUT:\n"
    "Please extract 3 most important keywords based on the text: ```Can you explain the concept of 'Perturb-and-Revise' in the context of 3D editing and how it addresses the limitations of existing methods in modifying geometry and appearance?```\n"
    
    "\nOUTPUT:\n"
    "Perturb-and-Revise;3D editing;geometry\n\n"
)

query_rewritten_instruction = (
    "Rewrite the given query and split the query into subqueries with complete context, separated by ';' if necessary: ```text```\n\n"
    "EXAMPLE:\n------------------------\nINPUT:\n"
    "Please rewrite the query: ```Can you explain the concept of 'Perturb-and-Revise' in the context of 3D editing and how it addresses the limitations of existing methods in modifyinggeometry and appearance?```\n"
    
    "\nOUTPUT:\n"
    "Can you explain the concept of 'Perturb-and-Revise' in the context of 3D editing?;How does 'Perturb-and-Revise' address the limitations of existing methods in modifying geometry?;How does 'Perturb-and-Revise' address the limitations of existing methods in modifying appearance?\n\n"
)

def query_generation(api_key, post_url, model, instruction, prompt, json_format=False):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    if not json_format:
        payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 1.2,
            }
    else:
        payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
                "temperature": 1.2,
                'response_format': {"type": "json_object"},
            }
    
    try:
        response = requests.post(post_url, headers=headers, json=payload)
        response = response.json()['choices'][0]['message']['content']
        
    except requests.exceptions.ConnectionError as e:
        print("ConnectionError: 网络连接问题", e)
        
    except RemoteDisconnected as e:
        print("RemoteDisconnected: 远端关闭了连接", e)
        
    except ProtocolError as e:
        print("ProtocolError: 协议错误", e)

    except requests.exceptions.RequestException as e:
        print("其他请求相关异常", e)
    except KeyError:
        print("KeyError: 缺少关键信息")
    
    try:
        return response
    except Exception:
        return ''

if __name__=='__main__':
    data_root = Path(__file__).parent.parent.absolute().joinpath('data')
    metadata = pd.read_csv(data_root.joinpath('latest_cs_papers_metadata.csv'))
    
    all_queries = {}
    
    for _, row in tqdm(metadata.iterrows()):
        query_prompt = (f"\nPlease generate five diverse questions based on the text: ```{row['abstract']}```. "
                        "\nOUTPUT:\n")
        
        queries = query_generation(Config.gpt_key, Config.gpt_url, Config.gpt_model, query_generation_instruction, query_prompt)
        queries = [q[1:].strip() for q in queries.split('\n') if q]

        for q in queries:
            rewritten_prompt = (f"Please rewrite the query: {q}"
                                "\nOUTPUT:\n")
            rewritten_queries_res = query_generation(Config.gpt_key, Config.gpt_url, Config.gpt_model, query_rewritten_instruction, rewritten_prompt)
            for rewritten_query in rewritten_queries_res.split(';'):
                keyword_prompt = (f'Please extract 3 most important keywords based on the text: ```{q}```'
                                "\nOUTPUT:\n")
                keywords = query_generation(Config.gpt_key, Config.gpt_url, Config.gpt_model, keyword_extraction_instruction, q)
                all_queries[queries] = {q: keywords}

    with open(data_root.joinpath('query_generation_result.json'), 'w') as f:
        json.dump(all_queries, f, indent=4, ensure_ascii=False)