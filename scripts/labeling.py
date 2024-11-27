# /*
#  * @Author: Zhai Wenjia 
#  * @Date: 2024-11-01 10:57:54 
#  * @Last Modified by:   Zhai Wenjia 
#  * @Last Modified time: 2024-11-01 10:57:54 
#  */

# 示例：python labeling.py -i MMQA -o MMQA -v
# 将 config1.py 改为 config.py，填入 GPT 信息

import json
import os
import requests
import argparse
import logging
import time
import random
from pathlib import Path
from string import punctuation
from urllib3.exceptions import ProtocolError
from http.client import RemoteDisconnected

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from config import Config

random.seed(42)
remove_list = stopwords.words('english') + list(punctuation)

def load_logger(script_path):
    root_dir = Path(script_path).parent.parent.absolute()
    os.makedirs(os.path.join(root_dir, 'logs'), exist_ok=True)
    log_file_name = f'{Path(script_path).stem}-{time.strftime("%Y_%m_%d", time.localtime())}.log'
    
    logging.basicConfig(
        level=logging.DEBUG, 
        format='%(asctime)s - %(lineno)d - %(message)s',  # Added correct placeholders
        datefmt='%Y-%m-%d %H:%M:%S', 
        filename=os.path.join(root_dir, 'logs', log_file_name)
    )
    
    return logging

def normalize(text: str):
    tokens = word_tokenize(str(text).lower())
    return set([t for t in tokens if t not in remove_list])

def calc_em(pred, reference):
    return int(normalize(pred) == normalize(reference))

def gpt_text_inference(instruction, prompt, verbose=False):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Config.gpt_key}"
    }
    
    payload = {
        "model": Config.gpt_model,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.8,
        # 'response_format': {"type": "json_object"},
    }
    
    try:
        response = requests.post(Config.gpt_url, headers=headers, json=payload)
        response = response.json()['choices'][0]['message']['content']
        response = '{' + ''.join(response.split('{')[1:])
        response = ''.join(response.rsplit('}')[:-1]) + '}'
        if verbose:
            logger.info(response)
        return eval(response)
    except requests.exceptions.ConnectionError as e:
        return {'Reasoning': e, 'Response': 'error'}
    except RemoteDisconnected as e:
        return {'Reasoning': e, 'Response': 'error'}
    except ProtocolError as e:
        return {'Reasoning': e, 'Response': 'error'}
    except requests.exceptions.RequestException as e:
        return {'Reasoning': e, 'Response': 'error'}
    except KeyError as e:
        return {'Reasoning': e, 'Response': 'error'}
    except SyntaxError as e:
        return {'Reasoning': e, 'Response': 'error'}
    except TypeError as e:
        return {'Reasoning': e, 'Response': 'error'}
    except AttributeError as e:
        return {'Reasoning': e, 'Response': 'error'}

instruction = '''
You are an expert in reading comprehension and logical reasoning, you are given a task: for a given query, two literally different answers are provided, your goal is to determine if these two answers are semantically equivalent for the query. Please think carefully and provide a concise but logical explanation. 

Objective: Given a question and two possible answers, determine if they are semantically equivalent.

You are provided with the following inputs:
1. Query: {The question}
2. Answer A: {The first answer}
3. Answer B: {The second answer}

Based on this, provide a clear explanation with logical steps. Your response should include:
1. Reasoning: A step-by-step explanation of how you analyzed the text to determine if the two answers are semantically equivalent.
2. Response: A binary value (`1` or `0`)indicating whether the two answers are semantically equivalent.

-----
SCHEMA
-----

{
    "Reasoning": "Step-by-step reasoning explaining why the two answers are semantically equivalent for the given question.",
    "Response": "1"/"0"
}

-----
EXAMPLE
-----

1. Query: "What is the shape of the pendant on Mary Tyler Moore's necklace?"
2. Answer A: "circle"
3. Answer B: "round"

Output:
{
    "Reasoning": "Answer A describes the pendant as 'circle,' which is a specific shape. Answer B describes it as 'round,' a term that also refers to circular shapes. Both terms indicate a shape with no angles and a consistent curvature, making them semantically equivalent in this context.", "Response": "1"
}

-----

1. Query: "What hand does Hamlet have on his head?", 
2. Answer A: "right", 
3. Answer B: "left"

Output:
{
    "Reasoning": "Answer A states 'right' as the hand on Hamlet's head, while Answer B states 'left.' Both answers specify a different hand, which is significant as the question explicitly refers to the position of one hand over the other. Since right and left are distinct in meaning and cannot be considered equivalent for this query, the answers are not semantically equivalent.", "Response": "0"
}

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-i', '--input_dir', dest='input_dir', required=True, help='Directory of the input files')
    parser.add_argument('-o', '--output_dir', dest='output_dir', help='Directory of the output files', default='hybrid')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False, help='Whether to show details of inference process')
    args = parser.parse_args()
    
    logger = load_logger(__file__)
    
    data_path = Path(__file__).parent.parent.absolute()
    os.makedirs(data_path.joinpath('output').joinpath(args.output_dir), exist_ok=True)
    input_files = data_path.joinpath('output').joinpath('hybrid.json')
    with open(input_files, 'r') as f:
        data = json.load(f)
    
    os.makedirs(data_path.joinpath('output'), exist_ok=True)
    output_file_paths = data_path.joinpath('output').joinpath(args.output_dir).rglob('*.json')
    output_filenames = set([f.stem for f in output_file_paths])
    
    for i, (q, a, p) in tqdm(enumerate(zip(data['query'], data['answer'], data['predict']))):
        if os.path.exists(data_path.joinpath('output').joinpath(args.output_dir).joinpath(f'{i}.json')):
            continue
        output = {'query':q, 'gold_answer': a, 'predict': p}
    
        if calc_em(a, p):
            output['consistency'] = "1"
            output['reasoning'] = "Literally the same answer"
        else:
            # gold_answer = ', '.join(gold_answer) if len(data['gold_answers']) > 1 else data['gold_answers'][0]
            prompt = f"1. Query: {output['query']}\n2. Answer A: {a}\n3. Answer B: {p}"
            inference = gpt_text_inference(instruction, prompt, args.verbose)
            output['consistency'] = inference['Response']
            output['reasoning'] = inference['Reasoning']

        try:
            with open(data_path.joinpath('output').joinpath(args.output_dir).joinpath(f'{i}.json'), 'w') as f:
                json.dump(output, f, ensure_ascii=False)
                logger.info(f'Writing {i}.json with consistency inference')
        except:
            if args.verbose:
                logger.info(f'Error in writing {i}.json')
        