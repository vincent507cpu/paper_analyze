import os
from pathlib import Path

import ollama

from utils.data import query_arxiv, DocumentParser, download_arxiv_pdf, pdf_parser
from utils.retrieval import BM25, VectorStore, Reranker
from utils.query import translation_chn2eng, translation_eng2chn, query_rewritten
from utils.utils import has_chn

def main():
    if os.path.exists('store/bm25_store.pkl'):
        bm25_retriever = BM25.load()
    else:
        bm25_retriever = None
    docs = []
    bm25_corpus = []
    
    # master_mode = input('欢迎使用文献助手！是否需要高级模式（y / n）：') or 'n'
    master_mode = 'n'
    if master_mode == 'y':
        print('进入高级模式，将使用自定义参数')
        while True:
            query = input("输入你的问题：")
            rewritten_query = query_rewritten(query)
            eng_query = translation_chn2eng(rewritten_query)
            
            max_result = input('问题理解完毕。你需要查找文献的数量（默认为 5 篇，指定不超过 10 篇）：') or 5
            assert 0 <= max_result <= 10, '输入的数字不在范围内'
            
            query_paper_result = query_arxiv(eng_query, max_results=max_result)
            if not len(query_paper_result):
                print('未找到相关文章，请重新输入...')
                continue
            
            print(f'查询到 {len(query_paper_result)} 篇文章，载入中...')
            vllm = input('请选择要使用的本地多模态模型（默认为 llama3.2-vision）：') or 'llama3.2-vision'
            parser = DocumentParser(vllm)
            for queried_paper in query_paper_result:
                download_arxiv_pdf(queried_paper['arxiv_id'])
                parsed_pdf = pdf_parser(queried_paper['arxiv_id'])
                
                doc = parser.parse_document(queried_paper['title'], queried_paper['abstract'], queried_paper['arxiv_id'], parsed_pdf)
                docs.append(doc)
                doc_bm25 = [d.page_content for d in doc]
                bm25_corpus.extend(doc_bm25)
                print(f"文章 {queried_paper['arxiv_id']} 处理完毕，待入库")

            # 载入（初始化）数据库，准备查询
            vector_store_name = input('请输入向量库名称（默认为 paper_vector_store）：') or 'paper_vector_store'
            embed_model = input('请输入嵌入模型名称（默认为 BAAI/bge-small-en-v1.5）：') or 'BAAI/bge-base-en-v1.5'
            vector_store = VectorStore(vector_store_name, embed_model)
            
            if not bm25_retriever:
                bm25_retriever = BM25(bm25_corpus)
            else:
                bm25_retriever.add_document(bm25_corpus)
            bm25_retriever.save()
            
            vector_store.add_documents(docs)
            vector_store.save()
            
            top_k = input('数据库已更新完毕，请输入搜索数量（默认为 5 篇，指定不超过 10 条）：') or 5
            assert 0 <= top_k <= 10, '输入的数字不在范围内'
            bm25_retrieval = [text for _, text in bm25_retriever.rank_documents(eng_query, top_k)]
            vector_store_retrieval = vector_store.query(eng_query, top_k)
            
            print('检索完成，进入重排序阶段')
            reranker = Reranker()
            result = reranker.rerank(eng_query, bm25_retrieval + vector_store_retrieval, top_k)
            print('重排序完成，生成结果...')
            
            response = ollama.chat(
                model="llama3.2-vision",
                messages=[
                    {"role": "system", "content": "Please answer the following question based on the following context by user:\nQuestion: {eng_query}"},
                    {"role": "user", "content": f"\n".join(result)}
                ]
            )
            final_response = translation_eng2chn(response['message']['content'])
            print(f'最终回复：\n{final_response}')
            
    else:
        print('进入普通模式，将使用默认参数')
        while True:
            parser = DocumentParser('llama3.2-vision')
            query = input("输入你的问题：")
            
            if has_chn(query):
                print('这条语句需要翻译成英语，翻译中...')
                rewritten_query = query_rewritten(query)
            # print(rewritten_query)
            eng_query = translation_chn2eng(rewritten_query)
            print('翻译的英文查询语句：', eng_query)
            # eng_query = 'Transformers is a type of transformer model, which has been widely used in natural language processing (NLP) tasks such as sequence tagging and named entity recognition.'
            print('开始检索相关文章...')
            query_paper_result = query_arxiv(eng_query, max_results=3)
            # query_paper_result = [{'arxiv_id': '2111.00830v2', 'title': 'Deep Learning Transformer Architecture for Named Entity Recognition on\n  Low Resourced Languages: State of the art results',
                                #   'abstract': '  This paper reports on the evaluation of Deep Learning (DL) transformer\narchitecture models for Named-Entity Recognition (NER) on ten low-resourced\nSouth African (SA) languages. In addition, these DL transformer models were\ncompared to other Neural Network and Machine Learning (ML) NER models. The\nfindings show that transformer models substantially improve performance when\napplying discrete fine-tuning parameters per language. Furthermore, fine-tuned\ntransformer models outperform other neural network and machine learning models\non NER with the low-resourced SA languages. For example, the transformer models\nobtained the highest F-scores for six of the ten SA languages and the highest\naverage F-score surpassing the Conditional Random Fields ML model. Practical\nimplications include developing high-performance NER capability with less\neffort and resource costs, potentially improving downstream NLP tasks such as\nMachine Translation (MT). Therefore, the application of DL transformer\narchitecture models for NLP NER sequence tagging tasks on low-resourced SA\nlanguages is viable. Additional research could evaluate the more recent\ntransformer architecture models on other Natural Language Processing tasks and\napplications, such as Phrase chunking, MT, and Part-of-Speech tagging.\n'}]
            print(f'查询到 {len(query_paper_result)} 篇文章，载入中...')
            
            for queried_paper in query_paper_result:
                download_arxiv_pdf(queried_paper['arxiv_id'])
                parsed_pdf = pdf_parser(queried_paper['arxiv_id'] + '.pdf')
                doc = parser.parse_document(queried_paper['title'], queried_paper['abstract'], queried_paper['arxiv_id'], parsed_pdf)
                docs.extend(doc)
                doc_bm25 = [d.page_content for d in doc]
                bm25_corpus.extend(doc_bm25)
                print(f"文章 {queried_paper['arxiv_id']} 处理完毕，待入库")
                
            vector_store = VectorStore('paper_vector_store', 'BAAI/bge-base-en-v1.5')
            
            if not bm25_retriever:
                bm25_retriever = BM25(bm25_corpus)
            else:
                bm25_retriever.add_document(bm25_corpus)
            bm25_retriever.save()
            
            vector_store.add_documents(docs)
            vector_store.save()

            bm25_retrieval = [text for _, text in bm25_retriever.rank_documents(eng_query, 5)]
            vector_store_retrieval = vector_store.query(eng_query, 5)
            
            print('检索完成，进入重排序阶段')
            reranker = Reranker()
            # print(vector_store_retrieval)
            # print(bm25_retrieval)
            rerank_result = reranker.rerank(eng_query, bm25_retrieval + [d.page_content for d in vector_store_retrieval], 5)
            print('重排序完成，生成结果...')
            
            response = ollama.chat(
                model="llama3.2-vision",
                messages=[
                    {"role": "system", "content": "Please answer the following question based on the following context by user:\nQuestion: {eng_query}"},
                    {"role": "user", "content": f"\n".join(rerank_result)}
                ]
            )
            final_response = translation_eng2chn(response.content)
            print(f"最终回复：\n{final_response.content}")
            
            break
            
    # # Step 1: User Query Input
    # query = input("Enter your query: ")
    
    # # Step 2: Trigger Search Action
    # search_results = perform_search(query)  # Function to handle search logic
    
    # # Step 3: Generate Output
    # output = generate_output(search_results)  # Generate final output
    # print(output)

# def perform_search(query):
#     # Step 1: Search Vector Store
#     search_results = search_vector_store(query)  # Search for relevant info in vector store
    
#     if not search_results:  # If no results found
#         # Step 2: Search ArXiv
#         papers = search_arxiv(query)  # Function to search ArXiv
        
#         # Step 3: Download Papers
#         downloaded_papers = download_papers(papers[:3])  # Download top 3 papers
        
#         # Step 4: Ingest into Vector Store
#         ingest_into_vector_store(downloaded_papers)  # Ingest papers into vector store
        
#         # Re-search Vector Store after ingestion
#         search_results = search_vector_store(query)  # Search again after ingesting papers
    
#     return search_results  # Return the search results

# def search_vector_store(query):
#     # ... existing code ...
#     {{ code }}
#     # Function to search the vector store

# def search_arxiv(query):
#     # ... existing code ...
#     {{ code }}
#     # Function to search ArXiv and return papers

# def download_papers(papers):
#     # ... existing code ...
#     {{ code }}
#     # Function to download papers

# def ingest_into_vector_store(papers):
#     # ... existing code ...
#     {{ code }}
#     # Function to convert papers to embeddings and store them

# def generate_output(search_results):
#     # ... existing code ...
#     {{ code }}
#     # Function to generate output from search results

if __name__ == "__main__":
    main()