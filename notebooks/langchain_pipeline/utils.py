import os
import re
import codecs
import math
from collections import Counter
from typing import List
import jieba

stop_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'stopwords.txt')

stop = set()
fr = codecs.open(stop_path, 'r', 'utf-8')
for word in fr:
    stop.add(word.strip())
fr.close()
re_zh = re.compile('([\u4E00-\u9FA5]+)')

def filter_stop(words):
    return [w for w in words if w not in stop]

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

class BM25:
    def __init__(self, corpus: List[List[str]], k1=1.5, b=0.75):
        assert isinstance(corpus, list), "Corpus must be a list of documents"
        assert all([isinstance(c, str) for c in corpus]), "Corpus must be a list of strings"
        
        tmp = []
        for para in corpus:
            filtered = filter_stop(jieba.lcut(para))
            if len(filtered) > 1:
                tmp.append([s for s in filtered if s != ' '])
            elif filtered != [' ']:
                tmp.append(filtered)
        
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
    
if __name__ == '__main__':
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog outpaces a swift fox",
        "The dog is lazy but the fox is swift",
        "Lazy dogs and swift foxes"
    ]
    doc = []
    for sent in corpus:
        words = [w.lower() for w in jieba.cut(sent) if w != ' ']
        words = filter_stop(words)
        print(words)
        doc.append(words)
    
    bm25 = BM25(doc)
    query = "quick brown dog"
    result = bm25.query(query)

    print("BM25 Scores for the query '{}':".format(query))
    for doc_id, score in result:
        print("Document {}: {}".format(doc_id, score))