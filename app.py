import os
from pathlib import Path
from typing import List
from collections import deque
import io
import sys
import time

import streamlit as st
from langchain_ollama.chat_models import ChatOllama

from utils.data import (DocumentParser, download_arxiv_pdf, pdf_parser,
                        query_arxiv_papers, rank_by_aggregated_reverse_value)
from utils.query import (keywords_extraction, multiple_query_generation,
                         query_rewritten, translation_chn2eng,
                         translation_eng2chn, is_relevant_check,
                         get_contextualized_question_prompt,
                         intention_detection, chn_chat)
from utils.retrieval import BM25, Reranker, VectorStore
from utils.utils import has_chn, flatten_list, capture_print_output

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)
reranker = Reranker("BAAI/bge-reranker-base")

class AcademicQueryHandler:
    def __init__(self, llm, chat_history, vector_store_name, bm25_store_name, parser):
        self._llm = llm
        self._chat_history = chat_history
        self._vector_store = VectorStore(vector_store_name=vector_store_name,
                                         embed_model='BAAI/bge-small-en-v1.5')
        self._bm25_retriever = BM25(bm25_store_name=bm25_store_name, k1=1.5, b=0.75)
        self.parser = parser

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

    def capture_print_output(self, func, *args, **kwargs):
        """Capture print output of a function."""
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
        return result, buffer.getvalue()

    def query_preprocessing(self, query):
        academic_intention = intention_detection(query, self._llm)
        print('\n是学术问题，进入学术检索...' if academic_intention else '\n是普通问题，不进入学术检索')
        if not academic_intention:
            if not has_chn(query):
                chn_query = translation_eng2chn(query, self._llm)
            else:
                chn_query = query
            self._chat_history.append('问题：' + chn_query)
            print(f'问题：{chn_query}')
            response = chn_chat('\n'.join(self._chat_history) + '\n回答：', self._llm)

            print(f"回复：{response}\n")
            self._chat_history.append('回答：' + response)
            return query, None
        else:
            if has_chn(query):
                print('这条语句需要翻译成英语，翻译中...')
                eng_query = translation_chn2eng(query, self._llm)
                print(f'翻译的英文查询语句：{eng_query}')
            else:
                eng_query = query
            return query, eng_query

    def academic_query(self, original_query, eng_query):
        if self._chat_history:
            print('在聊天历史中查询...')
            history = [text for text in self._chat_history if is_relevant_check(eng_query, text, self._llm)]
            if history:
                response = self.eng_chat(eng_query, self._chat_history)
                self._chat_history.append(response)
                self.update_stores(response)
                print(f'回答：{response}\n')
                return response
            else:
                return None

    def store_search(self, original_query, eng_query):
        print('在聊天记录中没有发现有关内容，在索引库中查询...')
        rewritten_query = query_rewritten(eng_query, self._llm)
        multiple_queries = multiple_query_generation(eng_query, self._llm)
        multiple_queries = [eng_query] + multiple_queries

        top_k = 3
        response = self.query_single_question(multiple_queries, self._vector_store, self._bm25_retriever, top_k=top_k)
        if response:
            return response, rewritten_query, multiple_queries
        else:
            return "未找到相关信息", rewritten_query, multiple_queries

    def user_feedback(self, original_query, response):
        feedback_message = f'对问题 "{original_query}" 的回复：\n"{response}"'
        return feedback_message, "用户反馈已记录。"

    def internet_search(self, original_query, eng_query, multiple_queries):
        if not multiple_queries:
            print("No queries available for internet search.")
            return None

        queried_papers = []
        docs = []
        bm25_corpus = []

        print('\n开始联网搜索新论文')
        arxiv_ids = set([
            doc.metadata['arxiv_id'] for doc in self._vector_store.vector_store.docstore._dict.values()
            if 'arxiv_id' in doc.metadata
        ])
        for query in multiple_queries:
            keywords = keywords_extraction(query, self._llm)
            papers = query_arxiv_papers(keywords, 3)
            queried_papers.append(papers)

        print('\n检索完成，开始进行排序、筛选')
        related_papers = rank_by_aggregated_reverse_value(multiple_queries, queried_papers, exclude=arxiv_ids)
        print(f'查询到 {len(related_papers)} 篇新论文，载入前 {3} 篇最相关论文中...')

        num = 0
        for queried_paper in related_papers[:3]:
            download_arxiv_pdf(queried_paper['arxiv_id'])
            parsed_pdf = pdf_parser(queried_paper['arxiv_id'] + '.pdf')
            doc = self.parser.parse_document(queried_paper['title'], queried_paper['abstract'], queried_paper['arxiv_id'], parsed_pdf)
            if doc:
                docs.extend(doc)
                bm25_corpus.extend([d.page_content for d in doc])
                print(f"文章 {queried_paper['arxiv_id']} 处理完毕，待入库")

            num += 1
            if num >= 3:
                break

        new_bm25_retriever = BM25(bm25_corpus, None)
        new_vector_store = VectorStore(vector_store_name=None, embed_model='BAAI/bge-small-en-v1.5', documents=docs)

        if multiple_queries:
            final_response = self.query_single_question(multiple_queries, new_vector_store, new_bm25_retriever, 3)
            if final_response:
                chn_response = translation_eng2chn(final_response, self._llm)
                print(f'\n问题 "{original_query}" 的最终回复：\n"{chn_response}"')
                return chn_response
        else:
            print("No valid queries to process.")
            return None

    def query_single_question(self, multiple_queries: List[str], vector_store, bm25_retriever, top_k: int):
        vector_store_retrieval, bm25_retrieval, retrieval_result = [], [], []
        for query in multiple_queries:
            vector_store_retrieval.extend(vector_store.vector_store_query(query, top_k))
            bm25_retrieval.extend([bm25_retriever.bm25_rank_documents(query, top_k)])
        retrieval_result.extend([p.page_content for p in vector_store_retrieval])
        retrieval_result.extend(bm25_retrieval)

        retrieval_result = flatten_list(retrieval_result)

        retrieval_result = reranker.rerank(multiple_queries[0], retrieval_result)
        rerank_result = []

        for i in range(len(retrieval_result)):
            if is_relevant_check(multiple_queries[0], retrieval_result[i], self._llm):
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

handler = AcademicQueryHandler(
    llm=llm,
    chat_history=[],
    vector_store_name='paper_vector_store',
    bm25_store_name='bm25_store.pkl',
    parser=DocumentParser('llama3.2-vision')
)

# Helper functions for updating logs
def update_logs(current_logs, new_message):
    return f"{current_logs}\n{new_message}".strip()

def append_logs_in_stream(new_message, log_state):
    updated_logs = f"{log_state}\n{new_message}".strip()
    return updated_logs

def preprocess_and_query(query, current_logs, dialog_history):
    dialog_history.append({"user": query})
    
    # Capture query preprocessing
    result, log_output = handler.capture_print_output(handler.query_preprocessing, query)
    original_query, rewritten_query = result
    updated_logs = update_logs(current_logs, log_output)

    if rewritten_query:
        response, log_output_academic = handler.capture_print_output(
            handler.academic_query, original_query, rewritten_query
        )
        updated_logs = update_logs(updated_logs, log_output_academic)

        if response:
            dialog_history.append({"assistant": response})
            feedback_message, _ = handler.user_feedback(original_query, response)
            print(feedback_message)
            return (
                response,
                updated_logs,
                True,
                True,
                updated_logs,
                dialog_history,
            )
        else:
            return store_search_process(rewritten_query, original_query, updated_logs, dialog_history)
    else:
        return "", updated_logs, False, False, updated_logs, dialog_history

def store_search_process(rewritten_query, original_query, current_logs, dialog_history):
    """
    Handles searching the local database and transitions to internet search if needed.
    """
    # Perform the database search
    result, log_output = handler.capture_print_output(handler.store_search, original_query, rewritten_query)
    response, rewritten_query_new, multiple_queries = result

    # Log query adjustments
    log_output += f"\n经过重写后的查询: {rewritten_query_new}"
    log_output += f"\n扩展查询列表:\n" + '\n'.join(multiple_queries)
    updated_logs = update_logs(current_logs, log_output)

    # If database returns meaningful results, stop here
    if response and response != "未找到相关信息":
        dialog_history.append({"assistant": response})
        feedback_message, _ = handler.user_feedback(original_query, response)
        print(feedback_message)
        return (
            response,
            updated_logs,
            True,
            True,
            updated_logs,
            dialog_history,
        )

    # Otherwise, transition to internet search using the existing query context
    return internet_search_process(multiple_queries, rewritten_query_new, original_query, updated_logs, dialog_history)

def internet_search_process(multiple_queries, rewritten_query, original_query, current_logs, dialog_history):
    """
    Handles searching the internet using expanded queries from the database search.
    """
    # Ensure queries are available for the internet search
    if not multiple_queries:
        error_message = "Error: No queries available for internet search."
        updated_logs = update_logs(current_logs, error_message)
        dialog_history.append({"assistant": error_message})
        return (
            "未找到相关信息",
            updated_logs,
            False,
            False,
            updated_logs,
            dialog_history,
        )

    try:
        # Use multiple queries for internet search
        result, log_output = handler.capture_print_output(
            handler.internet_search, original_query, rewritten_query, multiple_queries
        )
    except Exception as e:
        error_message = f"Error during internet search: {str(e)}"
        updated_logs = update_logs(current_logs, error_message)
        dialog_history.append({"assistant": error_message})
        return (
            "未找到相关信息",
            updated_logs,
            False,
            False,
            updated_logs,
            dialog_history,
        )

    # Default fallback response
    if not result:
        result = "未找到相关信息"
        log_output += "\nNo meaningful data was retrieved from the internet search."

    updated_logs = update_logs(current_logs, log_output)
    dialog_history.append({"assistant": result})

    # Attempt to capture feedback
    try:
        feedback_message, _ = handler.user_feedback(original_query, result)
        print(feedback_message)
    except Exception as e:
        feedback_message = f"Error generating feedback: {str(e)}"
        updated_logs = update_logs(updated_logs, feedback_message)

    return (
        result,
        updated_logs,
        False,
        False,
        updated_logs,
        dialog_history,
    )

def reset_to_new_query(logs, dialog_history):
    return "", logs, True, "", dialog_history

def store_search_process(rewritten_query, original_query, current_logs, dialog_history):
    result, log_output = handler.capture_print_output(handler.store_search, original_query, rewritten_query)
    response, rewritten_query_new, multiple_queries = result  # This should return multiple queries

    if not multiple_queries:
        multiple_queries = [rewritten_query_new]  # Fallback to the rewritten query if no others are generated

    log_output += f"\n经过重写后的查询: {rewritten_query_new}"
    log_output += f"\n扩展查询列表:\n" + '\n'.join(multiple_queries)
    updated_logs = f"{current_logs}\n{log_output}"

    # Update the session state with the logs before widget creation
    st.session_state['logs'] = updated_logs

    # If response is found, return it
    if response and response != "未找到相关信息":
        st.session_state['dialog_history'].append({"assistant": response})
        feedback_message, _ = handler.user_feedback(original_query, response)
        st.write(feedback_message)
        return response, multiple_queries  # Return both response and multiple_queries

    # If no relevant results found, proceed to internet search
    return internet_search_process(multiple_queries, rewritten_query_new, original_query, updated_logs, dialog_history)

def internet_search_process(multiple_queries, rewritten_query, original_query, current_logs, dialog_history):
    if not multiple_queries:
        error_message = "Error: No queries available for internet search."
        updated_logs = f"{current_logs}\n{error_message}"
        st.session_state['logs'] = updated_logs
        st.session_state['dialog_history'].append({"assistant": error_message})
        return "未找到相关信息"

    try:
        result, log_output = handler.capture_print_output(
            handler.internet_search, original_query, rewritten_query, multiple_queries
        )
    except Exception as e:
        error_message = f"Error during internet search: {str(e)}"
        updated_logs = f"{current_logs}\n{error_message}"
        st.session_state['logs'] = updated_logs
        st.session_state['dialog_history'].append({"assistant": error_message})
        return "未找到相关信息"

    if not result:
        result = "未找到相关信息"
        log_output += "\nNo meaningful data was retrieved from the internet search."

    updated_logs = f"{current_logs}\n{log_output}"
    st.session_state['logs'] = updated_logs
    st.session_state['dialog_history'].append({"assistant": result})

    try:
        feedback_message, _ = handler.user_feedback(original_query, result)
        st.write(feedback_message)
    except Exception as e:
        feedback_message = f"Error generating feedback: {str(e)}"
        updated_logs = f"{updated_logs}\n{feedback_message}"

    return result

def main_loop():
    # Set the title of the app
    st.title("arXiv 知识问答系统")

    # Initialize session state for logs, dialog history, and other variables if not already done
    if 'logs' not in st.session_state:
        st.session_state['logs'] = ""
        st.session_state['dialog_history'] = []
        st.session_state['academic_response'] = ""  # Initialize academic response as empty

    # Display prompt message
    st.markdown("**请发起新的查询**")

    # User query input
    query_input = st.text_input("输入查询")
    
    # Button to submit the query
    submit_query_button = st.button("开始查询")

    # Initialize the academic response and logs widget with session state value
    academic_response = st.text_area("学术查询结果", value=st.session_state['academic_response'], height=150, disabled=True)
    logs_placeholder = st.empty()  # Use an empty placeholder for logs

    # Confirm and deny buttons
    confirm_button = st.button("对结果满意，继续")
    deny_button = st.button("搜索互联网寻找信息")

    if submit_query_button:
        # Preprocess the query and generate logs and response
        response, multiple_queries = store_search_process(query_input, query_input, st.session_state['logs'], st.session_state['dialog_history'])

        # Update the logs placeholder dynamically
        logs_placeholder.text_area("实时日志", value=st.session_state['logs'], height=200, disabled=True)

        # Display the academic response (make sure this is where the response is being displayed)
        st.session_state['academic_response'] = response  # Store in session_state
        st.text_area("学术查询结果", value=st.session_state['academic_response'], height=150, disabled=True)

    if confirm_button:
        st.session_state['logs'] = ""  # Clear logs if user confirms
        st.session_state['dialog_history'] = []  # Reset dialog history

    if deny_button:
        st.session_state['logs'] = ""  # Clear logs if user denies
        st.session_state['dialog_history'] = []  # Reset dialog history
        
if __name__ == "__main__":
    main_loop()
