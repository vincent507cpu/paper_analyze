keyword_extract_prompt = """你是一名资深编辑，需要对给定的查询语句进行以下处理：

1. 规范化查询语句：使其表达简洁、清晰。
2. 提取查询主体：确定查询的主要对象（可以是名词或名词性短语）。

请输出：规范化后的查询主体。

示例：
INPUT: Transformers 的编码器是如何设计的？
"OUTPUT": Transformers 的编码器

INPUT: LLaVa 的线性层起到了什么作用？
"OUTPUT": LLaVa 的线性层

INPUT: Attention 好像很厉害，请问它是干嘛的？
"OUTPUT": Attention

INPUT: {}
OUTPUT: 
"""

qa_instruction = '''
You are an expert in information evaluation and critical thinking. Your task is to find the answer to a given question from a passage of text. You must carefully read every word and think through each step without overlooking any details. Your output should contain two fields: `Reasoning` and `Response`. In `Reasoning`, document your logical thought process in a clear, concise manner. If you find the answer, write it in the `Response` field; if not, try your best to guess one. The `Reasoning` should end with '*' to indicate completion.

Objective: The task is to carefully analyze a passage of text to determine whether it contains the answer to a given question. The evaluation must be detailed, with clear reasoning, and identify the correct answer if present, or confirm its absence.

You are provided with the following inputs:

1. Context: {The text provided to you.}
2. Question: {The question that is being asked.}

Based on these inputs, provide a step-by-step explanation to identify the correct answer from the content. If you cannot find the answer in the passage, try to guess the answer. Your response should only contain the answer itself. Do not explain, provide notes, or include any additional text, punctuation, or preposition (e.g., 'on', 'at'), or articles (e.g., 'a', 'an', 'the') unless absolutely necessary.

Output format: 

-----
SCHEMA
-----

{
    "Reasoning": "Step-by-step reasoning explaining how the answer is inferenced to satisfy the question.",
    "Response": "The answer itself, as simple as possible."
}

-----

1. Context: ```Pilotwings 64\nPilotwings 64 (Japanese: パイロットウイングス64, Hepburn: Pairottouingusu Rokujūyon) is a video game for the Nintendo 64, originally released in 1996 along with the debut of the console. The game was co-developed by Nintendo and the American visual technology group Paradigm Simulation. It was one of three launch titles for the Nintendo 64 in Japan as well as Europe and one of two launch titles in North America. Pilotwings 64 is a follow-up to Pilotwings for the Super Nintendo Entertainment System (SNES), which was a North American launch game for its respective console in 1991. Also like that game, Pilotwings 64 received production input from Nintendo producer Shigeru Miyamoto.```
2. Question: Who is a Japanese video game designer and producer, currently serving as the co-Representative Director of Nintendo, who gave production input to a video game for the Nintendo 64, originally released in 1996 along with the debut of the console?

-----

output:

{
    "Reasoning": "The context mentions that 'Pilotwings 64' was a video game released in 1996 for the Nintendo 64. The game received production input from Nintendo producer Shigeru Miyamoto. This aligns with the question, which asks for a Japanese video game designer and producer who gave production input to a Nintendo 64 game released in 1996. Additionally, Shigeru Miyamoto is well known as a prominent figure at Nintendo and is currently serving as the co-Representative Director of the company. Therefore, the content fully supports that Shigeru Miyamoto is the correct answer to the question.*", 
    "Response": "Shigeru Miyamoto" 
}

-----

'''