from utils.chat import MultiRoundDialogueManager

def main():
    chat_manager = MultiRoundDialogueManager()
    chat_manager.multi_turn_query()
    # if os.path.exists('store/bm25_store.pkl'):
    #     bm25_retriever = BM25.bm25_load()
    # else:
    #     bm25_retriever = None
    # docs = []
    # bm25_corpus = []
    # vector_store = None
    # llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
    
    # # master_mode = input('欢迎使用文献助手！是否需要高级模式（y / n）：') or 'n'
    # master_mode = 'n'
    # if master_mode == 'y':
    #     print('进入高级模式，将使用自定义参数')
    #     while True:
    #         query = input("输入你的问题：")
    #         rewritten_query = query_rewritten(query, llm)
    #         eng_query = translation_chn2eng(rewritten_query, llm)
            
    #         max_result = input('问题理解完毕。你需要查找文献的数量（默认为 5 篇，指定不超过 10 篇）：') or 5
    #         assert 0 <= max_result <= 10, '输入的数字不在范围内'
            
    #         query_paper_result = query_arxiv_papers(eng_query)
    #         if not len(query_paper_result):
    #             print('未找到相关文章，请重新输入...')
    #             continue
            
    #         num, cur_idx = 0, 0
    #         print(f'查询到 {len(query_paper_result)} 篇文章，载入中...')
    #         vllm = input('请选择要使用的本地多模态模型（默认为 llama3.2-vision）：') or 'llama3.2-vision'
    #         parser = DocumentParser(vllm)
    #         while cur_idx < len(query_paper_result):
    #             queried_paper = query_paper_result[cur_idx]
    #             try:
    #                 download_arxiv_pdf(queried_paper['arxiv_id'])
    #                 parsed_pdf = pdf_parser(queried_paper['arxiv_id'])
                    
    #                 doc = parser.parse_document(queried_paper['title'], 
    #                                             queried_paper['abstract'], 
    #                                             queried_paper['arxiv_id'], 
    #                                             parsed_pdf)
    #                 docs.append(doc)
    #                 doc_bm25 = [{'text':d.page_content, 'metadata':d.metadata} for d in doc]
    #                 bm25_corpus.extend(doc_bm25)
    #                 print(f"文章 {queried_paper['arxiv_id']} 处理完毕，待入库")
                    
    #                 num += 1
    #                 if num >= max_result:
    #                     break
    #             except Exception as e:
    #                 print(f'文章 {queried_paper["arxiv_id"]} 处理失败，错误信息：{e}')
    #             cur_idx += 1
                
    #         # 载入（初始化）数据库，准备查询
    #         vector_store_name = input('请输入向量库名称（默认为 paper_vector_store）：') or 'paper_vector_store'
    #         embed_model = input('请输入嵌入模型名称（默认为 BAAI/bge-small-en-v1.5）：') or 'BAAI/bge-base-en-v1.5'
    #         vector_store = VectorStore(vector_store_name, embed_model)
            
    #         if not bm25_retriever:
    #             bm25_retriever = BM25(bm25_corpus)
    #         else:
    #             bm25_retriever.bm25_add_document(bm25_corpus)
    #         bm25_retriever.bm25_save()
            
    #         vector_store.add_documents(docs)
    #         vector_store.save()
            
    #         top_k = input('数据库已更新完毕，请输入搜索数量（默认为 5 篇，指定不超过 10 条）：') or 5
    #         assert 0 <= top_k <= 10, '输入的数字不在范围内'
    #         bm25_retrieval = [text for _, text in bm25_retriever.bm25_rank_documents(eng_query, top_k)]
    #         vector_store_retrieval = vector_store.query(eng_query, top_k)
            
    #         print('检索完成，进入重排序阶段')
    #         reranker = Reranker()
    #         result = reranker.rerank(eng_query, bm25_retrieval + vector_store_retrieval, top_k)
    #         print('重排序完成，生成结果...')
            
    #         response = ollama.chat(
    #             model="llama3.2-vision",
    #             messages=[
    #                 {"role": "system", "content": "Please answer the following question based on the following context by user:\nQuestion: {eng_query}"},
    #                 {"role": "user", "content": f"\n".join(result)}
    #             ]
    #         )
    #         final_response = translation_eng2chn(response['message']['content'])
    #         print(f'最终回复：\n{final_response}')
            
    # else:
    #     print('进入普通模式，将使用默认参数')
    #     max_result = 3
        
    #     while True:
    #         rerank_result = []
    #         queried_papers = []
    #         keywords = []
    #         vector_store_retrieval, bm25_retrieval = [], []
    #         arxiv_ids = []
    #         parser = DocumentParser('llama3.2-vision')
    #         original_query = input("输入你的问题：")
    #         # original_query = 'What is the self-attention mechanism?'
            
    #         if has_chn(original_query):
    #             print('这条语句需要翻译成英语，翻译中...')
                
    #             eng_query = translation_chn2eng(original_query, llm)
    #             print('翻译的英文查询语句：', eng_query)
    #             rewritten_query = query_rewritten(eng_query, llm)
    #             print(f'经过重写后的查询：{rewritten_query}')
                
    #             # satisfied = input('对翻译结果是否满意？（y 满意，n 重新翻译，t 人工翻译）')
    #             # while True:
    #             #     if satisfied == 'y':
    #             #         break
    #             #     elif satisfied == 'n':
    #             #         eng_query = translation_chn2eng(query, llm)
    #             #         satisfied = input('对翻译结果是否满意？（y 满意，n 重新翻译，t 人工翻译）')
    #             #     elif satisfied == 't':
    #             #         eng_query = input('请输入人工翻译后的英文查询：')
    #             #         if has_chn(eng_query):
    #             #             print('输入的英文查询语句中包含中文，重新进行翻译')
    #             #             satisfied = 'n'
    #             #     else:
    #             #         print('输入有误，请重新输入')
    #         else:
    #             eng_query = original_query
    #             rewritten_query = query_rewritten(eng_query, llm)
    #             print(f'经过重写后的查询：{rewritten_query}')
                        
            
    #         multiple_queries = [eng_query] + multiple_querie_generation(rewritten_query, llm)
    #         tmp = '\n'.join(multiple_queries)
    #         print(f"经过扩写的新查询：\n{tmp}")
    #         print('查询处理完成，首先尝试在已有数据库中进行检索...')
    #         vector_store_search, bm25_search = False, False
    #         vector_store_root = Path('store')
    #         folders = [item.name for item in vector_store_root.iterdir() if item.is_dir()]

    #         if not folders and not bm25_retriever:
    #             print('数据库为空，开始查询在线论文。')
    #         elif folders and not bm25_retriever:
    #             vector_store_search = True
    #         elif not folders and bm25_retriever:
    #             bm25_search = True
    #         else:
    #             vector_store_search, bm25_search = True, True
                
    #         if vector_store_search:
    #             indexed_foleders = [' '.join([str(idx), folder]) for idx, folder in enumerate(folders, 1)]
    #             if len(indexed_foleders) > 1:
    #                 print('\n'.join(indexed_foleders))
    #                 selected_db = int(input('请选择向量数据库：\n'))
    #                 assert 1 <= selected_db <= len(folders), '输入的数字不在范围内'
    #                 vector_store = VectorStore(folders[selected_db - 1], 'BAAI/bge-base-en-v1.5')
    #             else:
    #                 vector_store = VectorStore(folders[0], 'BAAI/bge-base-en-v1.5')
                
    #             for query in multiple_queries:
    #                 vector_store_retrieval.extend(vector_store.query(query, 5))
    #         if bm25_search:
    #             for query in multiple_queries:
    #                 bm25_retrieval = [text for _, text in bm25_retriever.bm25_rank_documents(eng_query, 5)]
            
    #         if vector_store_retrieval or bm25_retrieval:
    #             # print([d.page_content for d in vector_store_retrieval])
    #             # import time; time.sleep(5)
    #             # print(bm25_retrieval)
    #             # time.sleep(5)
    #             reranker = Reranker()
    #             rerank_result = reranker.rerank(eng_query, bm25_retrieval + [d.page_content for d in vector_store_retrieval], 5)[:5]
    #             for i in range(len(rerank_result) - 1, -1, -1):
    #                 if not is_relavent_check(eng_query, rerank_result[i], llm):
    #                     rerank_result.pop(i)
    #             del reranker
                        
    #         if rerank_result:
    #             print('检索到相关文档，根据检索到的文档生成回答：')
    #             response = llm.invoke(
    #             f"Please answer the following question based on the following context by user:\nQuestion: {eng_query}\nContext:\n" + \
    #             "\n".join(rerank_result)
    #         ).content
    #             final_response = translation_eng2chn(response, llm)
    #             print(f" 问题 {eng_query} 的最终回复：\n{final_response}")
                
    #             satisfied = input('对生成的回复是否满意？（y 满意，输入其它字符 联网搜索新论文）')
    #             assert satisfied in ['y', 'n'], '输入有误'
    #             if satisfied == 'y':
    #                 break
    #             else:
    #                 print('开始联网搜索论文')
    #                 arxiv_ids = set([doc.metadata['arxiv_id'] for doc in vector_store.vector_store.docstore._dict.values()]) if vector_store else set()
            
    #         for query in multiple_queries:
    #             keywords = keywords_extraction(query, llm)
    #             papers = query_arxiv_papers(keywords, 5)
    #             queried_papers.append(papers)
            
    #         print('检索完成，开始进行排序、筛选')
    #         related_papers = rank_by_aggregated_reverse_value(rewritten_query, queried_papers, exclude=arxiv_ids)
    #         print(f'查询到 {len(related_papers)} 篇论文，载入前 {max_result} 篇最相关论文中...')
            
    #         num = 0
    #         for queried_paper in related_papers[:max_result]:
    #             download_arxiv_pdf(queried_paper['arxiv_id'])
    #             parsed_pdf = pdf_parser(queried_paper['arxiv_id'] + '.pdf')
    #             doc = parser.parse_document(queried_paper['title'], queried_paper['abstract'], queried_paper['arxiv_id'], parsed_pdf)
    #             if doc is None:
    #                 continue
    #             docs.extend(doc)
    #             doc_bm25 = [d.page_content for d in doc]
    #             bm25_corpus.extend(doc_bm25)
    #             print(f"文章 {queried_paper['arxiv_id']} 处理完毕，待入库")
                
    #             num += 1
    #             if num >= max_result:
    #                 break
            
    #         print('开始检索...')
    #         del parser
            
    #         if not bm25_retriever:
    #             print('根据语料初始化 BM25 检索器')
    #             bm25_retriever = BM25(bm25_corpus)
    #         else:
    #             print('载入 BM25 检索器')
    #             bm25_retriever.bm25_add_document(bm25_corpus)
    #         bm25_retriever.bm25_save()
            
    #         if not vector_store:
    #             print('根据语料初始化向量数据库')
    #             # 载入（初始化）数据库，准备查询
    #             vector_store_name = input('请输入向量库名称（默认为 paper_vector_store）：') or 'paper_vector_store'
    #             embed_model = input('请输入嵌入模型名称（默认为 BAAI/bge-small-en-v1.5）：') or 'BAAI/bge-base-en-v1.5'
    #             vector_store = VectorStore(vector_store_name, embed_model)
    #         vector_store.vector_store_add_documents(docs)
    #         vector_store.vector_store_save()

    #         for query in multiple_queries:
    #             pass
            
    #         bm25_retrieval = [text for _, text in bm25_retriever.bm25_rank_documents(eng_query, 5)]
    #         vector_store_retrieval = vector_store.vector_store_query(eng_query, 5)
            
    #         print('检索完成，进入重排序阶段')
    #         reranker = Reranker()
    #         # print(vector_store_retrieval)
    #         # print(bm25_retrieval)
    #         rerank_result = reranker.rerank(eng_query, bm25_retrieval + [d.page_content for d in vector_store_retrieval], 5)
    #         print('重排序完成，生成结果...')
            
    #         response = llm.invoke(
    #             f"Please answer the following question based on the following context by user:\nQuestion: {eng_query}\nContext:\n" + \
    #             "\n".join(rerank_result)
    #         ).content
    #         # response = ollama.chat(
    #         #     model="llama3.2-vision",
    #         #     messages=[
    #         #         {"role": "", "content": ""},
    #         #         {"role": "user", "content": f"Please answer the following question based on the following context by user:\nQuestion: {eng_query}\nContext:\n" + \
    #         #             "\n".join(rerank_result)}
    #         #     ]
    #         # )
    #         # print(response['message']['content'])
    #         final_response = translation_eng2chn(response, llm)
    #         print(f" 问题 '{original_query}' 的最终回复：\n----------------------------------\n{final_response}")
            
    #         # break
            

if __name__ == "__main__":
    main()