import os
import re
import sys
import io

stopwords = set()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stopwords.txt'),
          encoding='utf-8') as f:
    for line in f.readlines():
        stopwords.add(line.strip())
        
def has_chn(string):
    re_zh = re.compile('([\u4E00-\u9FA5]+)')
    return re_zh.search(string)

def filter_stop(words):
    return [w for w in words if w not in stopwords]

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

def flatten_list(nested_list):
    """
    递归展平嵌套列表。
    
    :param nested_list: 嵌套列表
    :return: 展平后的列表
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            # 如果元素是列表，则递归展平
            flat_list.extend(flatten_list(item))
        else:
            # 如果元素不是列表，直接添加到结果中
            flat_list.append(item)
    return flat_list

# 捕获打印输出
def capture_print_output(func, *args, **kwargs):
    '''
    Captures the output of a function that prints to standard output.

    :param func: The function whose output is to be captured.
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    :return: A tuple containing the result of the function and the captured output as a string.
    '''
    output_buffer = io.StringIO()
    sys.stdout = output_buffer  # 重定向标准输出到 StringIO
    try:
        result = func(*args, **kwargs)  # 执行原始函数
    finally:
        sys.stdout = sys.__stdout__  # 恢复标准输出
    return result, output_buffer.getvalue()

def quit(query):
    '''
    退出程序的函数。

    :param query: 用户输入的查询字符串。该字符串用于判断用户是否希望退出程序。
    '''
    if query in ['exit', 'quit', 'q', 'end', '退出', '取消', '停止']:
        print('退出中')
        sys.exit(0)
        
def continue_process():
    while True:
        query = input('是否继续？(y/n)')
        if query in ['y', 'yes', '是', '确定', '继续']:
            return True
        elif query in ['n', 'no', '否', '取消', '停止']:
            return False
        
def print_states(messages):
    """
    Prints the current state of messages.

    :param messages: A list of message objects to be printed. Each message should have a 'content' attribute,
                        and may optionally have a 'tool_call_id' attribute.
    """
    for i, message in enumerate(messages):
        print("-" * 40)
        print(f"Message {i + 1}:")
        print(f"  Type: {type(message).__name__}")
        print(f"  Content: {message.content}")
        if hasattr(message, 'tool_call_id'):  # 如果是 ToolMessage
            print(f"  Tool Call ID: {message.tool_call_id}")
        