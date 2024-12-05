import re
import math
from typing import List
from uuid import uuid4
from collections import Counter

import jieba
import ollama
from langchain_core import Document

from utils import filter_stop

class DocumentParser:
    def __init__(self, title, abstract, arxiv_id, content, vllm='llava-phi3'):
        self.title = ' '.join(t.strip() for t in title.split('\n'))
        self.abstract = ' '.join(p.strip() for p in abstract.split('\n'))
        self.arxiv_id = arxiv_id
        self.vllm = vllm
        self.doc_id = str(uuid4())
        
        start_pattern = r'#?\s*1?\s*Introduction'
        start_idx = re.search(start_pattern, content).start()
        end_pattern = r'#?\s*1?\s*Ref'
        end_idx = re.search(end_pattern, content).start()
        self.paras = [p.strip() for p in content[start_idx-1:end_idx].split('\n') if p]
        
    def parse_document(self):
        '''
        按段落提取 PDF 内容：
        - 保留正文部分（Introduction 以后，不包含 Reference 和 Appendix）。
        - 按段抓取，先按照长度进行语义分块，保留 Section、上下段信息。图片、表格仅保留 Section 信息。
        TODO: 拓扑结构信息
        '''
        print(f'{self.arxiv_id}\t{self.title} 开始解析...')
        doc = []
        cur_content = []
        # cur_section, next_section = '', ''
        skip = False
        pattern = r'^#?\s*(\d+(?:\.\d+)*)\s+(.+)'
        
        doc.append(Document(id=self.doc_id, page_content=self.abstract, metadata={'id_': str(uuid4()), 'arxiv_id': self.arxiv_id, 'title': self.title, 'type': 'text', 'section': 'abstract', 'previous': None, 'next': None}))
        
        i = 0
        while i < len(self.paras):
            # print(self.paras[i])
            # print(f'current: {i}')
            if len(doc):
                previous_id = doc[-1].metadata['id_']
                
            if re.match(pattern, self.paras[i]): # 检测 section 标题
                print('sec:', self.paras[i])
                cur_section =  ' '.join(re.match(pattern, self.paras[i]).groups())
                
                # tmp = Document(id=self.doc_id, page_content='\n'.join(cur_content), metadata={'id_': str(uuid4()), 'arxiv_id': self.arxiv_id, 'title': self.title, 'type': 'text', 'section': cur_section, 'previous': None, 'next': None})
                # doc.append(tmp)
                # # print(tmp, '\n')
                # cur_content = []
                # cur_section = next_section
                i += 1
                
            elif self.paras[i].startswith('![]') or 'Fig' in self.paras[i] or 'Table' in self.paras[i]: # 检测图片、表格
                if self.paras[i].startswith('![]'):
                    file_name = self.paras[i].split('/')[1].strip()[:-1]
                    if 'Fig' in self.paras[i+1] or 'Table' in self.paras[i+1]:
                        caption = self.paras[i+1]
                elif 'Fig' in self.paras[i] or 'Table' in self.paras[i]:
                    caption = self.paras[i]
                    if self.paras[i+1].startswith('![]'):
                        file_name = self.paras[i+1].split('/')[1].strip()[:-1]
                else:
                    doc.append(Document(id=self.doc_id, page_content=self.paras[i].strip(), metadata={'id_': str(uuid4()), 'arxiv_id': self.arxiv_id, 'title': self.title, 'type': 'text', 'section': cur_section, 'previous': previous_id, 'next': None}))
                    i += 1
                    continue
                
                summary = self.summarize_table_image(file_name, caption)
                # doc.append(Document(id=self.doc_id, page_content='\n'.join(cur_content), metadata={'id_': str(uuid4()), 'arxiv_id': self.arxiv_id, 'title': self.title, 'type': 'image/table', 'section': cur_section, 'previous': None, 'next': None}))
                doc.append(Document(id=self.doc_id, page_content=summary, metadata={'id_': str(uuid4()), 'arxiv_id': self.arxiv_id, 'title': self.title, 'type': 'image/table', 'section': cur_section, 'previous': previous_id, 'next': None}))
                cur_content = []
                # print('image / table:', self.paras[i], summary)
                i += 2
                
            elif self.paras[i] == '$$': # 检测公式
                cur_content = '$' + self.paras[i+1] + '$'
                doc.append(Document(id=self.doc_id, page_content=cur_content, metadata={'id_': str(uuid4()), 'arxiv_id': self.arxiv_id, 'title': self.title, 'type': 'equation', 'section': cur_section, 'previous': previous_id, 'next': None}))
                i += 2
            elif 'References' in self.paras[i]: # reference 开始，结束抓取
                break
            else:
                doc.append(Document(id=self.doc_id, page_content=self.paras[i].strip(), metadata={'id_': str(uuid4()), 'arxiv_id': self.arxiv_id, 'title': self.title, 'type': 'text', 'section': cur_section, 'previous': previous_id, 'next': None}))
                i += 1
        
        # doc.append(Document(id=self.doc_id, page_content='\n'.join(cur_content), metadata={'id_': str(uuid4()), 'arxiv_id': self.arxiv_id, 'title': self.title, 'type': 'text', 'section': cur_section, 'previous': None, 'next': None}))
        # print(doc[-1])
        
        for i in range(len(doc) - 1):
            # if not i:
            #     doc[0].metadata['next'] = doc[1].metadata['id_']
            # elif i == len(cur_content) - 1:
            #     doc[i].metadata['previous'] = doc[i-1].metadata['id_']
            # else:
            #     doc[i].metadata['previous'] = doc[i-1].metadata['id_']
                doc[i].metadata['next'] = doc[i+1].metadata['id_']
            
        # print(doc)
        # print(len(doc))
        return doc
                    
    def summarize_table_image(self, table_image, caption):
        response = ollama.chat(
            model=self.vllm,
            messages=[
                {
                    'role': 'user',
                    'content': f'Please summarize a following image or table based the given caption. It has to be descriptive and include the main points while avoid_ing any irrelevant details.\ncaption: {caption}',
                    'image': f'../../data/images/{table_image}'
                }
            ]
        )
        return response['message']['content']

class BM25:
    def __init__(self, corpus: List[List[str]], k1=1.5, b=0.75):
        assert isinstance(corpus, list), "Corpus must be a list of documents"
        assert all([isinstance(c, str) for c in corpus]), "Corpus must be a list of strings"
        
        tmp = []
        for para in corpus:
            filtered = filter_stop(jieba.lcut(para))
            tmp.append([s for s in filtered if s != ' '])
        
        self.k1 = k1
        self.b = b
        self.corpus = dict((i, doc) for i, doc in enumerate(corpus))
        self.doc_lengths = [len(doc) for doc in tmp]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_count = len(corpus)
        self.doc_term_freqs = [Counter(doc) for doc in tmp]
        self.build_inverted_index()

    def build_inverted_index(self):
        self.inverted_index = {}
        for doc_id, doc_term_freq in enumerate(self.doc_term_freqs):
            for term, freq in doc_term_freq.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_id, freq))

    def idf(self, term):
        doc_freq = len(self.inverted_index.get(term, []))
        if doc_freq == 0:
            return 0
        return math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def bm25_score(self, query_terms, doc_id):
        score = 0
        doc_length = self.doc_lengths[doc_id]
        for term in query_terms:
            tf = self.doc_term_freqs[doc_id].get(term, 0)
            idf = self.idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)
        return score

    def query(self, query):
        query_terms = [w.lower() for w in jieba.cut(query) if w.lower() != ' ']
        docs_w_scores = [(self.corpus[doc_id], self.bm25_score(query_terms, doc_id)) for doc_id in self.corpus.keys()]
        sorted_docs_by_scores = sorted(docs_w_scores, key=lambda x: x[1], reverse=True)
        return sorted_docs_by_scores

def text_loader():
    pass
    
def table_loader():
    pass

def image_loader():
    pass

def structural_semantic_chunking():
    pass

def filter_by_ppl():
    pass