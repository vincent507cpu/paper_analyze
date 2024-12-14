import os
import re
import codecs

stopwords = set()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stopwords.txt'),
          encoding='utf-8') as f:
    for line in f.readlines():
        stopwords.add(line.strip())
        
def has_chn(string):
    re_zh = re.compile('([\u4E00-\u9FA5]+)')
    return re_zh.search(string)

def filter_stop(words):
    return [w for w in words if w not in stopwords]

def get_sentences(doc):
    line_break = re.compile('[\r\n]')
    delimiter = re.compile('[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences

def flatten_list(nested_list):
    """
    递归展平嵌套列表。
    
    :param nested_list: 嵌套列表
    :return: 展平后的列表
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            # 如果元素是列表，则递归展平
            flat_list.extend(flatten_list(item))
        else:
            # 如果元素不是列表，直接添加到结果中
            flat_list.append(item)
    return flat_list