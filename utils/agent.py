import os
import sys
import pickle
from uuid import uuid4

from langgraph.graph import StateGraph, END
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

sys.path.append(os.path.abspath(os.path.join('..')))

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

from utils.instructions import (intention_identification_instruction, query_generation_instruction,
                                text_relevant_instruction, conversation_summarization_instruction)
from utils.config import Config
from utils.utils import quit, has_chn, flatten_list
from utils.query import (translation_eng2chn, query_rewritten, translation_chn2eng,
                        keywords_extraction)
from utils.retrieval import BM25, Reranker, VectorStore
from utils.data import (DocumentParser, download_arxiv_pdf, pdf_parser, query_single_question_in_stores,
                        query_arxiv_papers, rank_by_aggregated_reverse_value)

llm = ChatOpenAI(temperature=0,
                 model=Config.gpt_model,
                 openai_api_key=Config.gpt_key,
                 base_url=Config.gpt_url)

# llm = ChatOllama(model='qwen2.5:1.5b', temperature=0.0)

class Agent:
    def __init__(self, llm: ChatOpenAI, checkpointer: MemorySaver, system_prompt: str='', verbose: bool = False):
        self.system_prompt = system_prompt
        self.llm = llm
        self.verbose = verbose

        graph = StateGraph(AgentState)
        graph.add_node('original_query', self.original_query)
        # graph.add_node('printer', self.cur_state_printer)  # 专门的打印节点
        graph.add_node('casual_chat', self.casual_chat)
        graph.add_node('database_search', self.database_search)
        graph.add_node('online_search', self.online_search)
        graph.add_conditional_edges(
            'original_query',
            self.query_classification,
            {True: 'database_search', False: 'casual_chat'}
        )
        graph.add_conditional_edges(
            'database_search',
            self.satisfied,
            {True: 'original_query', False: 'online_search'}
        )
        graph.add_edge('casual_chat', 'original_query')
        graph.add_edge('database_search', 'online_search')
        graph.add_edge('online_search', 'original_query')
        graph.set_entry_point('original_query')
        self.graph = graph.compile(checkpointer=checkpointer)

    def cur_state_printer(self, latest_state: AnyMessage, current_message: AnyMessage):
        # message = state['messages'][-1]
        """根据 verbose 参数决定是否打印当前 state"""
        if self.verbose:
            print("\n--- 当前会话状态 ---")
            # print("-" * 40)
            """打印消息的辅助方法"""
            # for message in messages[-2:]:
            print(f"  类型: {type(latest_state).__name__}")
            print(f"  内容: {latest_state.content}")
            if hasattr(latest_state, 'tool_call_id'):  # 如果是 ToolMessage
                print(f"  Tool Call ID: {latest_state.tool_call_id}")
            print("-" * 40)
            print(f"  类型: {type(current_message).__name__}")
            print(f"  内容: {current_message.content}")
            if hasattr(current_message, 'tool_call_id'):  # 如果是 ToolMessage
                print(f"  Tool Call ID: {current_message.tool_call_id}")
            print("--- 当前会话结束 ---\n")

    def query_classification(self, state: AgentState):
        latest_message = state['messages'][-1].content
        if has_chn(latest_message):
            eng_message = translation_chn2eng(latest_message, self.llm)
        else:
            eng_message = latest_message
        # print('- eng_message:', eng_message)
        rewriten_message = query_rewritten(eng_message, self.llm)
        # print('- rewriten_message:', rewriten_message)
        result = self.llm.invoke(
            intention_identification_instruction + 'INPUT: {}'.format(rewriten_message)
        )
        print(f'- "{latest_message}" 是学术查询' if 'True' in result.content else f'- "{latest_message}" 不是学术查询')
        return 'True' in result.content
    
    def satisfied(self, state: AgentState):
        satisfied = input('\n对生成的回复是否满意？（y 满意，输入其它字符 联网搜索新论文）')
        if satisfied == 'y':
            return True
        else:
            return False
    
    
    def casual_chat(self, state: AgentState):
        latest_message = state['messages'][-1].content
        if not has_chn(latest_message):
            chn_message = translation_eng2chn(latest_message, self.llm)
        else:
            chn_message = latest_message
        print('- message:', chn_message)
        response = self.llm.invoke(
            state['messages'][:-1] + [HumanMessage(content=chn_message)]
        ).content
        message = ToolMessage(content=response, tool_call_id="casual_chat")
        self.cur_state_printer(state['messages'][-1], message)
        return {'messages': [message]}

    def database_search(self, state: AgentState):
        latest_message = state['messages'][-1].content
        self.latest_message = latest_message
        if has_chn(latest_message):
            print('- 这条语句需要翻译成英语，翻译中...')
            
            eng_query = translation_chn2eng(latest_message, self.llm)
            print('- 翻译的英文查询语句：', eng_query)
        else:
            eng_query = latest_message
        rewritten_query = query_rewritten(eng_query, self._llm)
        print(f'- 经过重写后的查询：{rewritten_query}')
        
        multiple_queries = llm.invoke(query_generation_instruction.format(rewritten_query)).content
        print(f"- 经过扩写的新查询：\n" + multiple_queries + '\n\n')
        self.multiple_queries = [rewritten_query] + multiple_queries.split('\n')
        
        self.bm25_retriever = BM25.bm25_load('bm25_store.pkl')
        self.vector_store = VectorStore(vector_store_name='paper_vector_store', embed_model='BAAI/bge-small-en-v1.5', documents=None)
        self.reranker = Reranker()
        
        vector_store_retrieval, bm25_retrieval, retrieval_result = [], [], []
        for query in self.multiple_queries:
            vector_store_retrieval.extend(self.vector_store.vector_store_query(query, 3))
            bm25_retrieval.extend([self.bm25_retriever.bm25_rank_documents(query, 3)])
        retrieval_result.extend([p.page_content for p in vector_store_retrieval])
        retrieval_result.extend(bm25_retrieval)
        
        # 展平嵌套列表
        retrieval_result = flatten_list(retrieval_result)
        retrieval_result = self.reranker.rerank(multiple_queries[0], retrieval_result)
        rerank_result = []
        for i in range(len(retrieval_result)):
            if llm.invoke(text_relevant_instruction.format(rewritten_query, retrieval_result[i])):
                rerank_result.append(retrieval_result[i])
            if len(rerank_result) >= 3:
                break
        
        if rerank_result:
            print('\n- 检索到相关文档，根据检索到的文档生成回答：')
            response = llm.invoke(
            f"Please answer the following question based on the following context by user:\nQuestion: {multiple_queries[0]}\nContext:\n" + \
            "\n".join(rerank_result)
        ).content
            response = translation_eng2chn(response, self.llm)
            qa_summary = self.llm.invoke(conversation_summarization_instruction.format(multiple_queries[0], response)).content
            
             # 添加到向量存储
            self.vector_store.vector_store_add_documents([[qa_summary]])
            self.vector_store.vector_store_save()
            # 添加到 BM25 存储
            self.bm25_retriever.bm25_add_documents([{'text': qa_summary,
                                        'metadata': None}])
            self.bm25_retriever.bm25_save()
            
        message = ToolMessage(content=response, tool_call_id="academic_query")
        self.cur_state_printer(state['messages'][-1], message)
        return {'messages': [message]}
    
    def online_search(self, state: AgentState):
        queried_papers = []
        docs = []
        bm25_corpus = []
        
        parser = DocumentParser('llava-llama3')
        arxiv_ids = set([
            doc.metadata['arxiv_id'] for doc \
                in self.vector_store.docstore._dict.values()
                if 'arxiv_id' in doc.metadata
        ])
        for query in self.multiple_queries:
            keywords = keywords_extraction(query)
            papers = query_arxiv_papers(keywords, 3)
            queried_papers.append(papers)
            
        print('\n检索完成，开始进行排序、筛选')
        related_papers = rank_by_aggregated_reverse_value(self.rewritten_query, queried_papers, exclude=arxiv_ids)
        print(f'查询到 {len(related_papers)} 篇新论文，载入前 3 篇最相关论文中...')
        
        num = 0
        for queried_paper in related_papers:
            download_arxiv_pdf(queried_paper['arxiv_id'])
            parsed_pdf = pdf_parser(queried_paper['arxiv_id'] + '.pdf')
            doc = parser.parse_document(queried_paper['title'], queried_paper['abstract'], queried_paper['arxiv_id'], parsed_pdf)
            if doc is None:
                continue
            docs.extend(doc)
            bm25_corpus.extend([d.page_content for d in doc])
            print(f"文章 {queried_paper['arxiv_id']} 处理完毕，待入库")
            
            num += 1
            if num >= 3:
                break
            
        new_bm25_retriever = BM25(bm25_corpus, 'new_bm25_store.pkl')
        new_vector_store = VectorStore(vector_store_name='new_paper_vector_store', embed_model='BAAI/bge-small-en-v1.5', documents=docs)
        
        final_response = query_single_question_in_stores(self.multiple_queries, 
                                                        new_vector_store, 
                                                        new_bm25_retriever, 
                                                        3)
        
        if final_response:
            chn_response = translation_eng2chn(final_response, self.llm)
            print(f'\n问题 "{self.latest_message}" 的最终回复：\n"{chn_response}"')
        
        self.vector_store.vector_store_add_documents(docs)
        self.vector_store.vector_store_save()
        self.bm25_retriever.bm25_add_documents(bm25_corpus)
        self.bm25_retriever.bm25_save()

    def original_query(self, state: AgentState):
        messages = state['messages']
        original_query = input('Input your query: ')
        quit(original_query)
        thread = {"configurable": {"thread_id": "1"}}
        messages += [HumanMessage(content=original_query, configurable=thread)]
        
        return {'messages': messages}

if __name__ == '__main__':
    system_prompt = ('Your are a helpful assistant. '
                     "Your task is to do your best to solve user's query.")
    chat_bot = Agent(llm, MemorySaver(), system_prompt, verbose=True)
    thread = {"configurable": {"thread_id": "1"}}
    print('init: Init graph...')
    # if not chat_bot.graph.get_state(config=thread)[0]:
    messages = [SystemMessage(content=system_prompt)]
    
    result = chat_bot.graph.invoke({'messages': messages}, config=thread)
    # print(result)
    # print(chat_bot.graph.get_state(config=thread))