import math
import os
import pickle
from collections import Counter
from typing import List, Tuple
from pathlib import Path

import jieba
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .utils import filter_stop

class BM25:
    def __init__(self, corpus: List[str], k1=1.5, b=0.75):
        """
        初始化 BM25 模型。

        :param corpus: 文档列表，每个文档为字符串。
        :param k1: BM25 算法中的调节参数，默认值为 1.5。
        :param b: BM25 算法中的调节参数，默认值为 0.75。
        """
        assert isinstance(corpus, list), "文档必须是字符串列表"
        self._original_corpus = corpus  # 保存原始文本
        docs = []
        for sent in corpus:
            words = [w.lower() for w in jieba.cut(sent) if w != ' ']  # 分词并转为小写
            words = filter_stop(words)  # 去除停用词
            docs.append(words)

        self._k1 = k1
        self._b = b
        self._corpus = docs  # 保存处理后的文档
        self._doc_lengths = [len(doc) for doc in docs]  # 计算每篇文档的长度
        self._avg_doc_length = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0  # 平均文档长度
        self._doc_count = len(docs)  # 文档总数
        self._doc_term_freqs = [Counter(doc) for doc in docs]  # 每篇文档的词频统计
        self._build_inverted_index()

    def _build_inverted_index(self):
        """构建倒排索引。"""
        self._inverted_index = {}
        for doc_id, doc_term_freq in enumerate(self._doc_term_freqs):
            for term, freq in doc_term_freq.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = []
                self._inverted_index[term].append((doc_id, freq))

    def _idf(self, term):
        """计算某个词的逆文档频率（IDF）。"""
        doc_freq = len(self._inverted_index.get(term, []))  # 包含该词的文档数量
        if doc_freq == 0:
            return 0
        return math.log((self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def _bm25_score(self, query_terms, doc_id):
        """计算某篇文档对给定查询的 BM25 得分。"""
        score = 0
        doc_length = self._doc_lengths[doc_id]
        for term in query_terms:
            tf = self._doc_term_freqs[doc_id].get(term, 0)  # 词频
            idf = self._idf(term)  # IDF 值
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (1 - self._b + self._b * (doc_length / self._avg_doc_length))
            score += idf * (numerator / denominator)
        return score

    def rank_documents(self, query: str, k: int = 5):
        """
        根据查询对文档进行 BM25 排序。

        :param query: 查询字符串
        :param k: 返回的文档数量，默认值为 5
        :return: (得分, 原始文档内容) 的排序列表
        """
        query_terms = [w.lower() for w in jieba.cut(query) if w.lower() != ' ']  # 对查询分词
        scores = [
            (self._bm25_score(query_terms, doc_id), self._original_corpus[doc_id])
            for doc_id in range(self._doc_count)
        ]
        return sorted(scores, key=lambda x: x[0], reverse=True)[:k]

    def save(self, file_name: str = 'bm25_store.pkl'):
        """
        将 BM25 模型保存到文件。

        :param file_name: 保存文件名
        """
        os.makedirs(Path(__file__).parent.parent.joinpath('store'), exist_ok=True)
        file_path = Path(__file__).parent.parent.joinpath(f'store/{file_name}')
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_name: str = 'bm25_store.pkl'):
        """
        从文件加载 BM25 模型。

        :param file_name: 文件名
        :return: 加载的 BM25 模型
        """
        file_path = Path(__file__).parent.parent.joinpath(f'store/{file_name}')
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def add_document(self, documents: List[str]):
        """
        向 BM25 模型中添加新文档，并更新模型。

        :param documents: 新文档列表
        """
        assert isinstance(documents, list), "新文档必须是字符串列表"
        assert isinstance(documents[0], str), "新文档必须是字符串"

        new_docs = []
        new_doc_term_freqs = []

        for sent in documents:
            words = [w.lower() for w in jieba.cut(sent) if w != ' ']  # 分词并转为小写
            words = filter_stop(words)  # 去除停用词
            new_docs.append(words)
            new_doc_term_freqs.append(Counter(words))  # 统计词频

        # 更新文档相关属性
        self._original_corpus.extend(documents)  # 添加原始文档
        self._corpus.extend(new_docs)
        self._doc_lengths.extend(len(doc) for doc in new_docs)
        self._doc_count += len(new_docs)
        self._avg_doc_length = sum(self._doc_lengths) / self._doc_count
        self._doc_term_freqs.extend(new_doc_term_freqs)

        # 更新倒排索引
        for doc_id, doc_term_freq in enumerate(new_doc_term_freqs, start=self._doc_count - len(new_docs)):
            for term, freq in doc_term_freq.items():
                if term not in self._inverted_index:
                    self._inverted_index[term] = []
                self._inverted_index[term].append((doc_id, freq))

class VectorStore:
    def __init__(self, vector_store_name='paper_vector_store', embed_model='BAAI/bge-small-en-v1.5') -> None:
        """
        初始化向量存储类。如果存储文件存在，则自动载入；否则生成一个空的向量存储。
        
        :param vector_store_name: 向量存储文件名
        :param embed_model: 用于生成向量的嵌入模型
        """
        self.vector_store_name = vector_store_name
        self.vector_store_path = Path(__file__).parent.parent.joinpath(f'store/{vector_store_name}')
        self.embed = HuggingFaceEmbeddings(model_name=embed_model)

        if os.path.exists(self.vector_store_path):
            self.vector_store = self._load()
        else:
            # 如果存储文件不存在，初始化为空
            print(f"在 {self.vector_store_name} 处未发现向量数据库，将初始化为空")
            self.vector_store = None

    def _load(self):
        """
        加载已有的向量存储。
        """
        print(f"从 {self.vector_store_name} 加载向量数据库")
        return FAISS.load_local(self.vector_store_path, self.embed, allow_dangerous_deserialization=True)

    def save(self):
        """
        保存向量存储到本地文件。
        """
        self.vector_store.save_local(self.vector_store_path)
        print(f"向量数据库保存至 {self.vector_store_name}")

    def query(self, query, k):
        """
        根据查询语句检索最相关的文档。
        
        :param query: 查询语句
        :param k: 返回的文档数量
        :return: 检索结果
        """
        if self.vector_store is None:
            print("向量存储为空，无法执行查询")
            return []
        
        retriever = self.vector_store.as_retriever(search_kwargs={'k': k})
        print(f"为 {query} 检索 {k} 个最相关的文档")
        return retriever.invoke(query)

    def add_documents(self, documents):
        """
        向向量存储中添加新的文档。
        在加入新文档之前，仅检查是否已有相同的 page_content，若存在则跳过。
        
        :param documents: 文档列表
        """
        print(f"准备向向量数据库中添加 {len(documents)} 条信息")

        # 获取已有文档的内容，用于检查重复数据
        existing_contents = set()
        if hasattr(self, 'vector_store') and self.vector_store is not None:
            # 检查向量存储是否为空
            if self.vector_store.docstore._dict:
                for doc in self.vector_store.docstore._dict.values():
                    existing_contents.add(doc.page_content)
            else:
                print("向量存储为空，尚无已有文档。")

        # 筛选出尚未存在的文档
        unique_documents = []
        for doc in documents:
            if doc.page_content not in existing_contents:
                unique_documents.append(doc)

        # 如果有新文档，更新向量存储
        if unique_documents:
            print(f"\n向向量数据库中添加 {len(unique_documents)} 条新信息")
            if self.vector_store:
                # 将新文档添加到现有存储中
                self.vector_store.add_documents(unique_documents)
            else:
                # 如果向量存储为空，则初始化存储
                self.vector_store = FAISS.from_documents(unique_documents, self.embed)
            self.save()
        else:
            print("没有新文档需要添加")
            
class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        """
        初始化 BGE Reranker 模型。

        :param model_name: 模型名称，默认为 "BAAI/bge-reranker-base"。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank(self, query: str, texts: list, k: int = 5):
        """
        使用 BGE 模型对文本列表进行重排序。

        :param query: 查询字符串
        :param texts: 待排序的文本列表
        :param k: 返回的前 k 个结果
        :return: 排序后的 (相关性得分, 文本) 列表
        """
        scores = []
        for text in texts:
            # 将 query 和每个 text 编码为模型输入
            inputs = self.tokenizer(query, text, return_tensors="pt", truncation=True)
            with torch.no_grad():  # 不需要梯度
                logits = self.model(**inputs).logits
                score = logits.item()  # 提取相关性得分
                scores.append((score, text))

        # 根据分数进行降序排序并返回前 k 个文本
        sorted_texts = [text for _, text in sorted(scores, key=lambda x: x[0], reverse=True)]
        return sorted_texts[:k]
        
if __name__ == '__main__':
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog outpaces a swift fox",
        "The dog is lazy but the fox is swift",
        "Lazy dogs and swift foxes"
    ]
    
    bm25 = BM25(corpus)
    query = "quick brown dog"
    result = bm25.rank_documents(query, 2)

    print("BM25 Scores for the query '{}':".format(query))
    for score, doc in result:
        print("{}: {}".format(score, doc))
    print()
    
    bm25.add_document(["The dog is lazy and the fox is swift", "Lazy dogs and swift foxes"])
    query = "quick brown dog"
    result = bm25.rank_documents(query, k=2)

    print("BM25 Scores for the query '{}':".format(query))
    for score, doc in result:
        print("{}: {}".format(score, doc))