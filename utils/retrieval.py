import math
import json
import os
from pathlib import Path
from collections import Counter
from typing import List

import jieba

class BM25:
    def __init__(self, corpus: List[List[str]], k1=1.5, b=0.75):
        assert isinstance(corpus, list), "Corpus must be a list of documents"
        assert all([isinstance(c, list) for c in corpus]), "Documents must be lists of words"
        
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_count = len(corpus)
        self.doc_term_freqs = [Counter(doc) for doc in corpus]
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

    def rank_documents(self, query):
        query_terms = [w.lower() for w in jieba.cut(query) if w.lower() != ' ']
        scores = [(doc_id, self.bm25_score(query_terms, doc_id)) for doc_id in range(self.doc_count)]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return sorted_scores
    
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
    result = bm25.rank_documents(query)

    print("BM25 Scores for the query '{}':".format(query))
    for doc_id, score in result:
        print("Document {}: {}".format(doc_id, score))