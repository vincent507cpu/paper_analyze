import os
from pathlib import Path
from typing import List
from collections import deque

from tqdm import tqdm
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama.chat_models import ChatOllama
from langchain_core.documents import Document

from utils.data import (DocumentParser, download_arxiv_pdf, pdf_parser,
                        query_arxiv_papers, rank_by_aggregated_reverse_value)
from utils.query import (keywords_extraction, multiple_query_generation,
                         query_rewritten, translation_chn2eng,
                         translation_eng2chn, is_relavent_check)
from utils.retrieval import BM25, Reranker, VectorStore
from utils.utils import has_chn, flatten_list
from .prompts import conversation_summarization_instruction

class MultiRoundDialogueManager:
    def __init__(self, corpus=None,
                 llm="qwen2.5:3b", 
                 embedding='BAAI/bge-small-en-v1.5', 
                 vector_store='paper_vector_store', 
                 bm25_store='bm25_store.pkl', 
                 rerank_model_name="BAAI/bge-reranker-base",
                 k1=1.5, b=0.75, verbose=False):
        # 初始化大语言模型
        self._llm = ChatOllama(model=llm, temperature=0)  # 设置温度为0保证生成内容更确定
        self._store = {}
        self._verbose = verbose
        self._chat_history = deque([], maxlen=100)
        self._vector_store_search = False
        self._bm25_search = False
        
        # 载入向量数据库和 BM25 检索器
        if os.path.exists(f'store/{bm25_store}'):
            self._bm25_retriever = BM25._bm25_load(f'store/{bm25_store}')
            self._bm25_retriever.bm25_add_documents(corpus)
        else:
            self._bm25_retriever = BM25(corpus=corpus, bm25_store_name=bm25_store, k1=k1, b=b)
        self._vector_store = VectorStore(vector_store_name=vector_store, embed_model=embedding)
        self.rerank_model_name = rerank_model_name
        
        # 初始化重排序器
        self.reranker = Reranker(model_name=rerank_model_name)
        
        if verbose:
            print('初始化（载入 BM25 数据库、向量数据库完成）')
        
    def _get_contextualized_question_prompt(self):
        """
        定义一个用于改写用户问题的提示模板，确保问题在上下文中具有明确性。
        - 如果没有聊天历史，问题保持不变。
        - 如果有历史，尝试基于上下文进行指代消解。
        - 如果不够完整，补充缺乏的内容。
        """
        system_prompt = f'''
            History:
            []
            Current question: How are you?
            Is coreference resolution needed: No => Reasoning: The output question is the same as the current question. => Output question: How are you?
            -------------------
            History:
            [Q: Is Milvus a vector database?
            A: Yes, Milvus is a vector database.]
            Current question: How do I use it?
            Is coreference resolution needed: Yes => Reasoning: I need to replace "it" in the current question with "Milvus." => Output question: How do I use Milvus?
            -------------------
            History:
            []
            Current question: Self-attention mechanism
            Is coreference resolution needed: Yes => Reasoning: The current question is too short and vague. I need to expand the question to make it clear and complete. => Output question: What is the self-attention mechanism?
            -------------------
            History:
            [Q: How to cook lobster?
            A: First, clean the lobster thoroughly, then steam or stir-fry it, and add seasoning based on personal preference.]
            Current question: How does it taste?
            Is coreference resolution needed: Yes => Reasoning: "It" in the current question refers to "lobster." I need to expand the question to make it clear and complete. => Output question: How does lobster taste?
            -------------------
            History:
            [Q: What is the difference between deep learning and traditional machine learning?
            A: Deep learning relies on neural networks to process large-scale data, while traditional machine learning often requires manual feature extraction and is suitable for smaller datasets.]
            Current question: Which is better?
            Is coreference resolution needed: Yes => Reasoning: "Which" in the current question refers to "deep learning and traditional machine learning." I need to expand the question to make it clear and complete. => Output question: Which is better, deep learning or traditional machine learning?
            ------------
        '''
        contextualize_question_prompt = ChatPromptTemplate([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "History: {chat_history}\nCurrent Question: {input}\nDoes it need anaphora resolution:")
        ])

        return contextualize_question_prompt

    def _get_answer_prompt(self):
        """
        定义一个问答任务的提示模板，使用检索出的上下文信息回答问题。
        """
        system_prompt = """\
        You are an assistant for a question-answering task. Please answer the question based on the following retrieved information:
        {context}
        """

        qa_prompt = ChatPromptTemplate([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        return qa_prompt
    
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """
        获取指定会话 ID 的聊天记录。如果会话记录不存在，则创建一个新的。
        """
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]
    
    def _qa_pair_summarization(self, rewritten_query: str, response: str):
        # 定义摘要的提示模板
        conversation_summarization_prompt = (
            'Input:\n'
            '- Question: {question}\n'
            '- Answer: {answer}\n'
            'Output:\n'
        )
        prompt = conversation_summarization_instruction + conversation_summarization_prompt.format(question=rewritten_query, answer=response)
        summary = self._llm.invoke(prompt)
        return summary.content
    
    def intention_detection(self, query: str) -> bool:
        prompt = (
            "Determine if the following text is a causual chat or an academic inquiry. Only response 'True' or 'False'.\n"
            "EXAMPLE:\n-----"
            "INPUT: Shall we go for lunch?\n"
            "OUTPUT: False\n"
            "-----\n"
            "INPUT: Introduce the movie Inception\n"
            "OUTPUT: False\n"
            "-----\n"
            "INPUT: What is perceptron?\n"
            "OUTPUT: True\n"
            "INPUT: Introduce self-attention machansim\n"
            "OUTPUT: True\n"
            "INPUT: Why Llava performed so well?\n"
            "OUTPUT: True\n"
            "INPUT: How do multimodal language model work?\n"
            "OUTPUT: True\n"
            "INPUT: {}\n"
            "OUTPUT: "
        )
        response = self._llm.invoke(prompt.format(query))
        return eval(response.content)
    
    def chn_chat(self, query: str):
        response = self._llm.invoke(f'''
                                    请回答如下问题：{query}\n回答：
                                    ''')
        return response.content
    
    def eng_chat(self, query: str, history=None):
        if not history:
            response = self._llm.invoke(
                "Please answer the following question by user:\n"
                "Question: " + query + "\nAnswer: "
            )
        else:
            response = self._llm.invoke(
                "Please answer the following question based on the following context:\n"
                "Context: " + "\n".join(history) + "\n"
                "Question: " + query + "\nAnswer: "
            )
        return response.content
    
    def multi_turn_query(self, debug=False) -> str:
        self.parser = DocumentParser('llama3.2-vision')
        master_mode = input('欢迎使用文献助手！是否需要高级模式（y / n）：') or 'n'
        if master_mode == 'y':
            print('进入高级模式，将使用自定义参数')
        else:
            print('进入普通模式，将使用默认参数')
        self.reranker = Reranker(self.rerank_model_name)
        
        # 主循环：获取用户输入并处理改写与问答
        while True:
            original_query = input("\n输入你的问题：")
            if original_query.strip().lower() in ['退出', 'exit', 'quit', 'end', '结束']:
                break
            
            if has_chn(original_query):
                print('这条语句需要翻译成英语，翻译中...')
                
                eng_query = translation_chn2eng(original_query, self._llm)
                print('翻译的英文查询语句：', eng_query)
            else:
                eng_query = original_query
            rewritten_query = query_rewritten(eng_query, self._llm)
            print(f'经过重写后的查询：{rewritten_query}')
            
            academic_inttention = self.intention_detection(original_query)
            print('\n是学术问题' if academic_inttention else '\n是普通问题')
            if not academic_inttention:
                if has_chn(original_query):
                    self._chat_history.append('问题：' + original_query)
                    print(f'问题：{original_query}')
                    response = self.chn_chat('\n'.join(self._chat_history) + '\n回答：')
                else:
                    chn_query = translation_eng2chn(original_query, self._llm)
                    self._chat_history.append('问题：' + chn_query)
                    print(f'问题：{chn_query}')
                    response = self.chn_chat('\n'.join(self._chat_history) + '\n回答：')
                print(f"回复：{response}\n")
                self._chat_history.append('回答：' + response)
                # summary = self._qa_pair_summarization(original_query, response)
                # self._chat_history.append(summary)
                continue
            else:
                self.academic_query(original_query, rewritten_query, master_mode)
            # if inputs == "end":
            #     self._vector_store.save_local('../chat/vector_store')  # 保存向量存储
            #     self.bm25_save()
            #     print("数据库保存成功，退出查询")
            #     break

            # # 改写用户问题
            # res = contextualize_question_chain.invoke({
            #     "input": inputs
            # }, config={
            #     "configurable": {"session_id": "test456"}
            # })
            # rewritten_input = res.content.split('输出问题: ')[-1]
            # print("改写后内容：\n" + rewritten_input)

            # # 使用改写后的问题进行问答
            # res = conversational_rag_chain.invoke({
            #     "input": rewritten_input
            # }, config={
            #     "configurable": {"session_id": "test123"}
            # })
            # print("回答：\n" + res["answer"])
            
            # # 将 Q&A 存储到向量存储和 BM25 存储中
            # self.update_stores(rewritten_input, res["answer"])
            
            if debug:
                break
    
    def academic_query(self, original_query: str, rewritten_query: str, master_mode: bool):
        queried_papers = []
        docs = []
        bm25_corpus = []
        
        if self._chat_history:
            print('在聊天历史中查询...')
            history = [text for text in self._chat_history if is_relavent_check(rewritten_query, text, self._llm)]
            if history:
                response = self.eng_chat(rewritten_query, self._chat_history)
                self._chat_history.append(response)
                self.update_stores(response)
                print(f'回答：{response}\n')
                return
            
        print('在聊天记录中没有发现有关内容，在索引库中查询...')
        multiple_queries = multiple_query_generation(rewritten_query, self._llm)
        print(f"经过扩写的新查询：\n" + '\n'.join(multiple_queries) + '\n\n')
        multiple_queries = [rewritten_query] + multiple_queries
        
        if master_mode == 'y':
            top_k = int(input('\n请输入要使用的检索数量：'))
        else:
            top_k = 3
        response = self.query_single_question(multiple_queries,
                                                    self._vector_store,
                                                    self._bm25_retriever,
                                                    top_k=top_k)
        if response:
            print(f'对问题 "{original_query}" 的回复：\n"{response}"')
            
            satisfied = input('\n对生成的回复是否满意？（y 满意，输入其它字符 联网搜索新论文）')
            if satisfied == 'y':
                return

        print('\n开始联网搜索新论文')
        arxiv_ids = set([
            doc.metadata['arxiv_id'] for doc \
                in self._vector_store.vector_store.docstore._dict.values()
                if 'arxiv_id' in doc.metadata
        ])
        for query in multiple_queries:
            keywords = keywords_extraction(query, self._llm)
            papers = query_arxiv_papers(keywords, top_k)
            queried_papers.append(papers)
                
        print('\n检索完成，开始进行排序、筛选')
        if master_mode == 'y':
            top_k = int(input('请输入要载入的新论文数量：'))
        else:
            top_k = 3
        related_papers = rank_by_aggregated_reverse_value(rewritten_query, queried_papers, exclude=arxiv_ids)
        print(f'查询到 {len(related_papers)} 篇新论文，载入前 {top_k} 篇最相关论文中...')
        
        num = 0
        for queried_paper in related_papers:
            download_arxiv_pdf(queried_paper['arxiv_id'])
            parsed_pdf = pdf_parser(queried_paper['arxiv_id'] + '.pdf')
            doc = self.parser.parse_document(queried_paper['title'], queried_paper['abstract'], queried_paper['arxiv_id'], parsed_pdf)
            if doc is None:
                continue
            docs.extend(doc)
            bm25_corpus.extend([d.page_content for d in doc])
            print(f"文章 {queried_paper['arxiv_id']} 处理完毕，待入库")
            
            num += 1
            if num >= top_k:
                break
            
        new_bm25_retriever = BM25(bm25_corpus, 'new_bm25_store.pkl')
        new_vector_store = VectorStore(vector_store_name='new_paper_vector_store', embed_model='BAAI/bge-small-en-v1.5', documents=docs)
        
        final_response = self.query_single_question(multiple_queries, 
                                                    new_vector_store, 
                                                    new_bm25_retriever, 
                                                    top_k)
        
        if final_response:
            chn_response = translation_eng2chn(final_response, self._llm)
            print(f'\n问题 "{original_query}" 的最终回复：\n"{chn_response}"')
        
        self._chat_history.append(final_response)
        self._vector_store.vector_store_add_documents(docs)
        self._vector_store.vector_store_save()
        self._bm25_retriever.bm25_add_documents(bm25_corpus)
        self._bm25_retriever.bm25_save()
        
    def query_single_question(self, multiple_queries: List[str], vector_store, bm25_retriever, top_k: int):
        vector_store_retrieval, bm25_retrieval, retrieval_result = [], [], []
        for query in multiple_queries:
            vector_store_retrieval.extend(vector_store.vector_store_query(query, top_k))
            bm25_retrieval.extend([bm25_retriever.bm25_rank_documents(query, top_k)])
        retrieval_result.extend([p.page_content for p in vector_store_retrieval])
        retrieval_result.extend(bm25_retrieval)
        
        # 展平嵌套列表
        retrieval_result = flatten_list(retrieval_result)

        # print(retrieval_result)  # 调试打印

        retrieval_result = self.reranker.rerank(multiple_queries[0], retrieval_result)
        rerank_result = []
        for i in range(len(retrieval_result)):
            if is_relavent_check(multiple_queries[0], retrieval_result[i], self._llm):
                rerank_result.append(retrieval_result[i])
            if len(rerank_result) >= top_k:
                break
                
        if rerank_result:
            print('\n检索到相关文档，根据检索到的文档生成回答：')
            response = self._llm.invoke(
            f"Please answer the following question based on the following context by user:\nQuestion: {multiple_queries[0]}\nContext:\n" + \
            "\n".join(rerank_result)
        ).content
            response = translation_eng2chn(response, self._llm)
            return response
        else:
            return None
        
    def stores_updating_summary(self, summary: str):
        # 创建一个新的文档
        new_document = Document(page_content=summary, metadata={'arxiv_id': None,
                'title': None,
                'type': 'text',
                'section': 'qa_pair_summary',
                'previous': None,
                'next': None})

        # 添加到向量存储
        self.vector_store_add_documents([[new_document]])

        # 添加到 BM25 存储
        self.bm25_add_documents([{'text': new_document.page_content,
                                  'metadata': new_document.metadata}])
            
if __name__ == "__main__":
    chat = MultiRoundDialogueManager(llm="qwen2.5:7b", 
                                     embedding='BAAI/bge-base-en-v1.5', 
                                     vector_store='../store/vector_store',
                                     bm25_store='../store/bm25_store.pkl',
                                     verbose= True)
    
    chat.multi_turn_query()