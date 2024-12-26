import math
import os
import pickle
from collections import Counter
from typing import List, Dict
from pathlib import Path

import jieba
import torch
import faiss
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from utils import filter_stop

class BM25:
    def __init__(self, documents: List[str]=None, 
                 bm25_store_name: str = 'bm25_store.pkl', 
                 k1=1.5, b=0.75):
        """
        初始化 BM25 模型。如果存档文件存在，加载数据并添加传入的文档；否则以传入的文档初始化模型。

        :param documents: 初始文档数据列表，每个文档为 {'text': 文本, 'metadata': 元数据} 格式。
        :param bm25_store_name: 存档文件名。
        :param k1: BM25 参数，默认值 1.5。
        :param b: BM25 参数，默认值 0.75。
        """
        self._bm25_store_name = bm25_store_name
        # 获取存档文件路径
        file_path = self._get_store_path()
        # 如果存档文件存在，加载数据并添加文档；否则初始化新模型
        
        self._k1 = k1
        self._b = b
        self._avg_doc_length = 0
        self._doc_count = 0
        self._doc_lengths = []
        self._doc_term_freqs = []
        self._inverted_index = {}
        self._original_corpus = []
        self._metadata = []

        if documents is not None:
            # self._initialize_new_model()
            self.bm25_add_documents(documents)

    def _get_store_path(self):
        """获取存档文件路径。"""
        return Path(__file__).parent.parent.joinpath(f'store/{self._bm25_store_name}')

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
        doc_freq = len(self._inverted_index.get(term, []))
        if doc_freq == 0:
            return 0
        return math.log((self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def _bm25_score(self, query_terms, doc_id):
        """计算某篇文档对给定查询的 BM25 得分。"""
        score = 0
        doc_length = self._doc_lengths[doc_id]
        for term in query_terms:
            tf = self._doc_term_freqs[doc_id].get(term, 0)
            idf = self._idf(term)
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (1 - self._b + self._b * (doc_length / self._avg_doc_length))
            score += idf * (numerator / denominator)
        return score

    def bm25_add_documents(self, documents: List):
        """
        向 BM25 模型中添加新文档，并更新模型。

        :param documents: 文档列表，每个文档为 {'text': 文本, 'metadata': 元数据} 格式。
        """
        if not documents:
            return
        assert isinstance(documents, list), "文档必须是字典列表"

        new_docs = []
        new_doc_term_freqs = []
        new_metadata = []

        for doc in documents:
            text = doc.page_content if hasattr(doc, 'page_content') else doc
            if text in self._original_corpus:  # 去重
                continue

            words = [w.lower() for w in jieba.cut(text) if w.strip()]
            words = self._filter_stop(words)  # 去除停用词
            if not words:  # 如果文档在过滤后为空，跳过
                continue

            new_docs.append(text)
            new_doc_term_freqs.append(Counter(words))
            # new_metadata.append(doc.metadata)

        # 更新模型
        self._original_corpus.extend(new_docs)
        self._doc_lengths.extend(len(freq) for freq in new_doc_term_freqs)
        self._metadata.extend(new_metadata)
        self._doc_count += len(new_docs)
        self._avg_doc_length = sum(self._doc_lengths) / self._doc_count if self._doc_count else 0
        self._doc_term_freqs.extend(new_doc_term_freqs)

        self._build_inverted_index()

    def bm25_rank_documents(self, query: str, k: int = 5) -> List[Dict[str, str]]:
        """
        根据查询对文档进行 BM25 排序。

        :param query: 查询字符串。
        :param k: 返回的文档数量，默认值为 5。
        :return: 格式为 [{'text': 文本, 'metadata': 元数据}, ...] 的列表。
        """
        query_terms = [w.lower() for w in jieba.cut(query) if w.strip()]
        query_terms = self._filter_stop(query_terms)

        scores = [
            (self._bm25_score(query_terms, doc_id), doc_id)
            for doc_id in range(self._doc_count)
        ]
        ranked = sorted(scores, key=lambda x: x[0], reverse=True)[:k]
        # print(len(self._metadata))
        # for _, idx in ranked:
        #     print(self._original_corpus[idx])
        # return [{'text': self._original_corpus[doc_id], 'metadata': self._metadata[doc_id]} for _, doc_id in ranked]
        return [self._original_corpus[doc_id] for _, doc_id in ranked]

    def bm25_save(self):
        """
        将 BM25 模型保存到文件。
        """
        file_path = self._get_store_path()
        os.makedirs(file_path.parent, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def bm25_load(cls, file_path: Path):
        """
        从文件加载 BM25 模型。
        """
        with open(file_path, 'rb') as f:
            loaded_model = pickle.load(f)
        # 创建一个未初始化的实例
        instance = cls(documents=None)
        # 将加载的属性赋值给实例
        instance.__dict__.update(loaded_model.__dict__)
        return instance

    def _filter_stop(self, words):
        """去除停用词。"""
        return filter_stop(words)

class VectorStore:
    def __init__(self, 
                 vector_store_name='paper_vector_store', 
                 embed_model='BAAI/bge-small-en-v1.5',
                 documents: List=None) -> None:
        """
        初始化向量存储类。如果存储文件存在，则自动载入；否则生成一个空的向量存储。
        
        :param vector_store_name: 向量存储文件名
        :param embed_model: 用于生成向量的嵌入模型
        """
        self.vector_store_name = vector_store_name
        self.vector_store_path = Path(__file__).parent.parent.joinpath(f'store/{vector_store_name}')
        self.embed = HuggingFaceEmbeddings(model_name=embed_model)

        if os.path.exists(self.vector_store_path):
            self.vector_store = self._vector_store_load()
        else:
            # 如果存储文件不存在，初始化为空
            print(f"\n在 {self.vector_store_name} 处未发现向量数据库，将初始化一个为空向量数据库")
            index = faiss.IndexFlatL2(len(self.embed.embed_query('hello')))
            self.vector_store = FAISS(self.embed, index, 
                                      InMemoryDocstore(),
                                      index_to_docstore_id={})
            
        if documents:
            self.vector_store_add_documents(documents)

    def _vector_store_load(self):
        """
        加载已有的向量存储。
        """
        vector_store = FAISS.load_local(self.vector_store_path, self.embed, allow_dangerous_deserialization=True)
        print(f"\n从 {self.vector_store_name} 加载向量数据库")
        return vector_store

    def vector_store_save(self):
        """
        保存向量存储到本地文件。
        """
        self.vector_store.save_local(self.vector_store_path)
        print(f"向量数据库保存至 {self.vector_store_name}")

    def vector_store_query(self, query, k):
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
        # print(f"为 {query} 检索 {k} 个最相关的文档")
        return retriever.invoke(query)

    def vector_store_add_documents(self, documents):
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
            self.vector_store_save()
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

    def rerank(self, query: str, texts: List[str]) -> List[str]:
        """
        使用 BGE 模型对文本列表进行重排序。

        :param query: 查询字符串
        :param texts: 待排序的文本列表
        :return: 排序后的文本列表
        """
        # 为 query 构建与 texts 等长的列表
        query_list = [query] * len(texts)
        
        # 批量对 query 和 texts 进行编码
        inputs = self.tokenizer(query_list, texts, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():  # 不需要梯度
            logits = self.model(**inputs).logits  # 获取相关性得分

        # 提取分数
        scores = logits.squeeze().tolist()  # 转为 Python 列表
        if isinstance(scores, float):  # 处理单条文本情况
            scores = [scores]

        # 根据分数排序
        sorted_texts = [text for _, text in sorted(zip(scores, texts), key=lambda x: x[0], reverse=True)]

        return sorted_texts
        
if __name__ == '__main__':
    from langchain_core.documents import Document
    # 初始化文档
    corpus = [
        {"text": "The quick brown fox jumps over the lazy dog", "metadata": {"id": 1, "source": "example1"}},
        {"text": "A quick brown dog outpaces a swift fox", "metadata": {"id": 2, "source": "example2"}},
        {"text": "The dog is lazy but the fox is swift", "metadata": {"id": 3, "source": "example3"}},
        {"text": "Lazy dogs and swift foxes", "metadata": {"id": 4, "source": "example4"}}
    ]
    
    # 将字典转换为 Document 对象
    document_corpus = [Document(id=doc['metadata']['id'], page_content=doc['text'], metadata=doc['metadata']) for doc in corpus]
    
    # 初始化 BM25 模型
    bm25 = BM25(documents=document_corpus)
    
    # 查询
    query = "quick brown dog"
    result = bm25.bm25_rank_documents(query, k=2)

    # 输出查询结果
    print("\nBM25 Scores for the query '{}':".format(query))
    for doc in result:
        print("Text: {}, Metadata: {}".format(doc['text'], doc['metadata']))

    # 添加新文档
    new_documents = [
        {"text": "The dog is lazy and the fox is swift", "metadata": {"id": 5, "source": "example5"}},
        {"text": "Lazy dogs and swift foxes", "metadata": {"id": 6, "source": "example6"}}
    ]
    # 将字典转换为 Document 对象
    new_corpus = [Document(id=doc['metadata']['id'], page_content=doc['text'], metadata=doc['metadata']) for doc in corpus]
    bm25.bm25_add_documents(new_corpus)

    # 再次查询
    query = "quick brown dog"
    result = bm25.bm25_rank_documents(query, k=2)

    # 输出更新后的查询结果
    print("\nBM25 Scores for the query '{}':".format(query))
    for doc in result:
        print("Text: {}, Metadata: {}".format(doc['text'], doc['metadata']))

    print('\nVector Store')
    vector_store = VectorStore(documents=document_corpus)
    print(vector_store.vector_store_query(query, k=2))