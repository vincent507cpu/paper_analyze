query_generation_instruction = """
### GOAL:
Generate five diverse questions based on the given text. The term 'diverse' indicates that the questions should vary significantly in their wording, level of detail, clarity, completeness, and phrasing.

-----
### INSTRUCTION:
Please generate five distinct questions based on the provided text or topic. Ensure diversity in the following aspects:
1. Wording, phrasing, and structure.
2. Level of clarity and completeness.
3. Inclusion of variations such as intentional misspellings, grammatical errors, or misleading terms.
The questions should explore various aspects of the text, including its key ideas, applications, challenges, and implications.

-----
### SCHEMA:
- **Input:** A provided text or topic for generating diverse questions.
- **Output:** Five distinct questions that reflect a range of phrasing styles, clarity, and completeness.

-----
### EXAMPLES:
-----
INPUT: Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail

OUTPUT:
1. Can you explain what is meant by 'zero-shot' in the context of deep stereo matching and how it contributes to the robustness of the 'Stereo Anywhere' method?  
2. How does the 'Stereo Anywhere' technique ensure robust performance even in scenarios where traditional stereo or monocular methods fail?  
3. In what ways does 'Stereo Anywhere' differ from other deep stereo matching techniques, particularly in terms of robustness and zero-shot learning capabilities?  
4. Can u giv som examples of real-world aplications where 'Stereo Anywhere' would be really usefull, especialy in situtations were other methods might fail?  
5. What are the potential drawbacks or obstacles of the 'Stereo Anywhere' approach, and how might these be overcome in future research to ensure even greater reliability and accuracy in stereo matching?  

-----
INPUT: Neural Radiance Fields (NeRF): Representing and Synthesizing Photorealistic 3D Scenes

OUTPUT:
1. What are the key principles behind Neural Radiance Fields (NeRF), and how do they enable the synthesis of photorealistic 3D scenes?  
2. How does NeRF's representation of 3D scenes differ from traditional 3D modeling approaches?  
3. In what specific scenarios or industries could NeRF technology be applied, and what advantages does it offer over existing methods?  
4. What are some limitations of NeRF when synthesizing photorealistic 3D scenes, and how might these be addressed in future research?  
5. Can yu explane how NeRF achieves the combination of rendering accuracy with computational eficiency, and why this is importnt in real-world aplications?  

-----
INPUT: {}

OUTPUT:
"""

keyword_extraction_instruction = """
### GOAL:
Extract the 3 most important keywords from the given query. The keywords should represent the core ideas or entities mentioned in the query, capturing its intent or content.

-----
### INSTRUCTION:
Extract the 3 most important keywords from the provided query or text. Arrange the keywords in descending order of their importance and separate them using a semicolon (';'). Focus on unique and meaningful terms central to understanding the query's purpose.

-----
### SCHEMA:
**INPUT:** A natural language query or text.
**OUTPUT:** Three keywords separated by a semicolon, listed in descending order of their importance.

-----
### EXAMPLES:
-----
INPUT: Can you explain the concept of 'Perturb-and-Revise' in the context of 3D editing and how it addresses the limitations of existing methods in modifying geometry and appearance?

OUTPUT: Perturb-and-Revise;3D editing;geometry
-----
INPUT: What are the challenges in applying Neural Radiance Fields (NeRF) for generating photorealistic 3D scenes in outdoor environments?

OUTPUT: Neural Radiance Fields;3D scenes;outdoor environments
-----
INPUT: How does gradient descent optimize neural network parameters and what are its advantages over other optimization algorithms?

OUTPUT: gradient descent;optimization;neural network parameters
-----
INPUT: Can you discuss the applications of diffusion models in image synthesis and how they compare to GANs?

OUTPUT: diffusion models;image synthesis;GANs
-----
INPUT: What are the advantages and challenges of using large language models like GPT-4 for retrieval-augmented generation (RAG)?

OUTPUT: large language models;GPT-4;retrieval-augmented generation
-----
INPUT: Explain the role of attention mechanisms in improving the performance of transformers for natural language processing tasks.

OUTPUT: attention mechanisms;transformers;natural language processing
-----
INPUT: Please extract the 3 most important keywords based on the text: ```{}```

OUTPUT: 
"""

conversation_summarization_instruction = '''
GOAL: Summarize a pair of question and answer into a concise, clear, and accurate summary that captures the essence of both the question and the answer.
-----
INSTRUCTION: 
1. Provide a question and its corresponding answer as input.
2. Use the examples and guidelines to structure the summary accurately.
3. Ensure the output is a concise representation of the input question and answer.
-----
SCHEMA:
INPUT:
- **Question:** A natural language question provided as input.
- **Answer:** A natural language answer corresponding to the input question.

OUTPUT:
A concise summary that encapsulates the main points of the question and the answer. The summary should be coherent, easy to understand, and logically structured.
-----
EXAMPLE:
INPUT:
- **Question:** What are the benefits of regular exercise?
- **Answer:** Regular exercise has numerous benefits, including improved cardiovascular health, increased muscle strength, better mental health, and enhanced overall well-being. It can also help in weight management and reduce the risk of chronic diseases like diabetes and heart disease.

OUTPUT:
Regular exercise improves cardiovascular health, muscle strength, and mental health while enhancing overall well-being. It aids in weight management and reduces risks of chronic diseases like diabetes and heart disease.
-----
EXAMPLE:
INPUT:
- **Question:** How does photosynthesis work in plants?
- **Answer:** Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. The process occurs in the chloroplasts of plant cells and involves two main stages: the light-dependent reactions and the Calvin cycle. During the light-dependent reactions, light energy is converted into chemical energy in the form of ATP and NADPH. These energy carriers are then used in the Calvin cycle to fix carbon dioxide into organic compounds like glucose.

OUTPUT:
Photosynthesis enables plants to convert sunlight into chemical energy. It involves light-dependent reactions that produce ATP and NADPH, which are used in the Calvin cycle to fix carbon dioxide into glucose and other organic compounds.
-----
EXAMPLE:
INPUT:
- **Question:** What is the role of artificial intelligence in healthcare?
- **Answer:** Artificial intelligence (AI) plays a significant role in healthcare by enabling advancements in diagnostics, treatment planning, and personalized medicine. AI algorithms can analyze vast amounts of medical data, identify patterns, and provide insights that aid in early disease detection and improved patient care. Applications include AI-powered diagnostic tools, robotic surgeries, and predictive analytics for patient outcomes.

OUTPUT:
AI in healthcare enhances diagnostics, treatment planning, and personalized medicine by analyzing medical data and providing insights for early disease detection and improved patient care. Applications include diagnostic tools, robotic surgeries, and predictive analytics.
-----
INPUT:
- **Question:** {}
- **Answer:** {}

OUTPUT:
'''

text_relevant_instruction = '''
GOAL: Determine whether a given content is related to a specific question and provide a clear, logical explanation for the conclusion.
-----
INSTRUCTION:
1. Carefully read the provided content and question.
2. Use step-by-step reasoning to assess the relationship between the content and the question.
3. Respond with:
   - **True**: If the content is related and can help answer the question.
   - **False**: If the content is not related to the question.
4. Provide your output in the specified JSON format.
-----
SCHEMA:
INPUT:
- **Content**: A piece of text provided for evaluation.
- **Question**: A specific question that may or may not be related to the content.

OUTPUT:
- **Reasoning**: A step-by-step explanation of how the relationship between the content and the question was assessed.
- **Response**: A value (`True` or `False`) indicating the outcome of the assessment.
-----
EXAMPLE:
INPUT:
**Content**: Shigeru Miyamoto (Japanese: 宮本 茂, Hepburn: Miyamoto Shigeru, born November 16, 1952) (pronounced [mʲijamoto ɕiŋeɾɯ̥; ɕiɡeɾɯ̥]) is a Japanese video game designer and producer, currently serving as the co-Representative Director of Nintendo. He is best known as the creator of some of the most critically acclaimed and best-selling video games and franchises of all time, such as Donkey Kong, Mario, The Legend of Zelda, Star Fox, F-Zero, and Pikmin.

**Question**: Who is a Japanese video game designer and producer, currently serving as the co-Representative Director of Nintendo, who gave production input to a video game for the Nintendo 64, originally released in 1996 along with the debut of the console?

OUTPUT:
{
“Reasoning”: “Step 1: The question asks for a Japanese video game designer and producer who is currently serving as the co-Representative Director of Nintendo and contributed to a game released in 1996 for the Nintendo 64. Step 2: The content introduces Shigeru Miyamoto, a Japanese video game designer and producer who holds the position of co-Representative Director at Nintendo, matching part of the question’s criteria. Step 3: While the content does not directly mention Miyamoto’s involvement in a specific game from 1996, his significant role in the creation of major games for Nintendo suggests relevance to the query. Step 4: The content aligns with the information asked in the question. Step 5: Therefore, the content is sufficiently related to the question.”,
“Response”: “True”
}
-----
EXAMPLE:
INPUT:
**Content**: The provided image features the iconic logo of the Nintendo 64, a home video game console developed and marketed by Nintendo. The logo prominently displays the word "NINTENDO" in bold, blue capital letters, followed by the number "64" in red, signifying the console's 64-bit processor. Below the text, there is a three-dimensional, multicolored "N" composed of four interlocking shapes, each side colored differently in blue, green, red, and yellow. This design element not only emphasizes the brand but also symbolizes the innovative and playful nature of the console. The Nintendo 64, often abbreviated as N64, was known for its groundbreaking graphics and influential games, cementing its legacy in the gaming industry. The logo's vibrant colors and geometric design reflect the console's emphasis on advanced technology and entertainment.

**Question**: Who is a Japanese video game designer and producer, currently serving as the co-Representative Director of Nintendo, who gave production input to a video game for the Nintendo 64, originally released in 1996 along with the debut of the console?

OUTPUT:
{
“Reasoning”: “Step 1: The question asks for the name of a Japanese video game designer and producer who contributed to a Nintendo 64 game released in 1996 and currently serves as the co-Representative Director of Nintendo. Step 2: The content describes the Nintendo 64 logo and emphasizes the console’s legacy but does not mention any individuals or specific contributors to the games. Step 3: While the content provides relevant information about the Nintendo 64 itself, it does not directly address the question about the producer. Step 4: Since the required information about a specific individual is missing, the content is not related to the question.”,
“Response”: “False”
}
-----
INPUT:
**Content**: {}
**Question**: {}

OUTPUT:
'''


get_contextualized_question_instruction = """
### GOAL:
Determine if the current question requires contextualization or coreference resolution based on the provided history. Expand or clarify the current question to make it clear and self-contained.

-----
### INSTRUCTION:
Analyze the provided history and the current question to decide if coreference resolution or contextualization is needed. If necessary, rewrite the current question to ensure it is clear, complete, and self-contained. Provide a reasoning step explaining your decision.

-----
### SCHEMA:
- **Input:**
  - History: A list of prior questions (Q) and answers (A) in a conversation.
  - Current question: The question being analyzed.
- **Output:**
  - Is coreference resolution needed: Yes/No
  - Reasoning: A step-by-step explanation of the decision.
  - Output question: A self-contained and clear version of the current question.

-----
### EXAMPLES:
-----
INPUT:
History:
[]
Current question: How are you?

OUTPUT:
Is coreference resolution needed: No  
Reasoning: The output question is the same as the current question.  
Output question: How are you?  
-----
INPUT:
History:
[Q: Is Milvus a vector database?  
A: Yes, Milvus is a vector database.]  
Current question: How do I use it?

OUTPUT:
Is coreference resolution needed: Yes  
Reasoning: "It" in the current question refers to "Milvus." The current question needs to be rewritten to explicitly mention "Milvus."  
Output question: How do I use Milvus?  
-----
INPUT:
History:
[]  
Current question: Self-attention mechanism  

OUTPUT:
Is coreference resolution needed: Yes  
Reasoning: The current question is too short and vague. It needs to be expanded to make it clear and complete.  
Output question: What is the self-attention mechanism?  
-----
INPUT:
History:
[Q: How to cook lobster?  
A: First, clean the lobster thoroughly, then steam or stir-fry it, and add seasoning based on personal preference.]  
Current question: How does it taste?  

OUTPUT:
Is coreference resolution needed: Yes  
Reasoning: "It" in the current question refers to "lobster." The question needs to explicitly mention "lobster" to be clear.  
Output question: How does lobster taste?  
-----
INPUT:
History:
[Q: What is the difference between deep learning and traditional machine learning?  
A: Deep learning relies on neural networks to process large-scale data, while traditional machine learning often requires manual feature extraction and is suitable for smaller datasets.]  
Current question: Which is better?  

OUTPUT:
Is coreference resolution needed: Yes  
Reasoning: "Which" in the current question refers to "deep learning and traditional machine learning." The question needs to be expanded for clarity.  
Output question: Which is better, deep learning or traditional machine learning?  
-----
INPUT:
History:
{}  
Current question: {}

OUTPUT:
"""

translation_chn2eng_instruction = """
GOAL: Translate Chinese queries into English as accurately as possible.

-----
INSTRUCTION: Please translate the following Chinese queries into English as accurately as possible. Do not explain, add content unrelated to the original query, or provide extra information.

-----
SCHEMA:
1. Chinese Query
2. Translation

-----
EXAMPLE:
INPUT: 请问今天的天气怎么样？
OUTPUT: What is the weather like today?
-----
EXAMPLE:
INPUT: 这家餐厅的评价如何？
OUTPUT: How are the reviews for this restaurant?
-----
EXAMPLE:
INPUT: 我需要预订一张明天去北京的机票。
OUTPUT: I need to book a flight to Beijing for tomorrow.
-----
EXAMPLE:
INPUT: 什么是机器学习？
OUTPUT: What is machine learning?
-----
EXAMPLE:
INPUT: 如何训练一个神经网络？
OUTPUT: How do you train a neural network?
-----
EXAMPLE:
INPUT: 数据分析
OUTPUT: Data analysis
-----
EXAMPLE:
INPUT: 人工智能
OUTPUT: Artificial intelligence
-----
EXAMPLE:
INPUT: 深度学习模型
OUTPUT: Deep learning model
-----
EXAMPLE:
INPUT: 自然语言处理
OUTPUT: Natural language processing
-----
EXAMPLE:
INPUT: 强化学习算法
OUTPUT: Reinforcement learning algorithm
-----
INPUT: {}
OUTPUT: 
"""

translation_eng2chn_instruction = """
GOAL: Translate English queries into Chinese as accurately as possible.

-----
INSTRUCTION: Please translate the following English queries into Chinese as accurately as possible. Do not explain, add content unrelated to the original query, or provide extra information.

-----
SCHEMA:
1. English Query
2. Translation

-----
EXAMPLE:
INPUT: What is the weather like today?
OUTPUT: 请问今天的天气怎么样？
-----
EXAMPLE:
INPUT: How are the reviews for this restaurant?
OUTPUT: 这家餐厅的评价如何？
-----
EXAMPLE:
INPUT: I need to book a flight to Beijing for tomorrow.
OUTPUT: 我需要预订一张明天去北京的机票。
-----
EXAMPLE:
INPUT: What is machine learning?
OUTPUT: 什么是机器学习？
-----
EXAMPLE:
INPUT: How do you train a neural network?
OUTPUT: 如何训练一个神经网络？
-----
EXAMPLE:
INPUT: Data analysis
OUTPUT: 数据分析
-----
EXAMPLE:
INPUT: Artificial intelligence
OUTPUT: 人工智能
-----
EXAMPLE:
INPUT: Deep learning model
OUTPUT: 深度学习模型
-----
EXAMPLE:
INPUT: Natural language processing
OUTPUT: 自然语言处理
-----
EXAMPLE:
INPUT: Reinforcement learning algorithm
OUTPUT: 强化学习算法
-----
INPUT: {}
OUTPUT: 
"""

query_clarification_instruction = """
### GOAL:
Rewrite the given query to make it concise, specific, and logical while correcting grammatical mistakes or typos. Ensure the rewritten query is in the form of a clear, grammatically accurate question. If the given query is already correct and clear or actually not a query (such as being a statement), return the original query.

-----
### INSTRUCTION:
Please rewrite the given query according to the following requirements:
1. Make it concise and specific.
2. Ensure logical structure and grammatical correctness.
3. Retain the original intent and meaning of the query.

-----
### SCHEMA:
- **Input:** A user-provided query that might be incomplete, ambiguous, or contain errors.
- **Output:** A corrected and refined query in the form of a logical, grammatically accurate question.

-----
### GOLD STANDARD:
Ensure the final output maintains the intent and meaning of the original query while improving its clarity, specificity, and grammatical correctness.

-----
### EXAMPLES:
-----
INPUT:
Please rewrite the query: ```gradient descent optimize```

OUTPUT:
How does gradient descent perform optimization?
-----
INPUT:
Please rewrite the query: ```'Stereo Anywhere' aplication```

OUTPUT:
What is the application of 'Stereo Anywhere'?
-----
INPUT:
Please rewrite the query: ```what define convolutional neural```

OUTPUT:
What defines a convolutional neural network?
-----
INPUT:
Please rewrite the query: ```how improve natural language generation?```

OUTPUT:
How can natural language generation be improved?
-----
INPUT:
Please rewrite the query: ```why 'backpropagation' important AI?```

OUTPUT:
Why is backpropagation important in AI?
-----
INPUT:
Please rewrite the query: ```Today is a good day, I'm so happy.```

OUTPUT:
Today is a good day, I'm so happy.
-----
INPUT:
Please rewrite the query: ```{}```

OUTPUT:
"""


intention_identification_instruction = """
### GOAL:
Determine whether the given text represents an academic inquiry in computer science field or not. Respond with only 'True' if it is an academic inquiry and 'False' otherwise.

-----
### INSTRUCTION:
Analyze the following input text and determine if it is an academic inquiry.

-----
SCHEMA:
INPUT: A single text query provided as a natural language sentence.
THOUGHT PROCESS: A brief explanation of the reasoning behind the determination.
OUTPUT: A binary response ('True' or 'False').
  - 'True': If the text is an academic inquiry.
  - 'False': If the text is casual chat

-----
### EXAMPLES:
-----
INPUT: Hello!
THOUGHT PROCESS: The text is a greeting, which is typical of casual chat.
OUTPUT: False
-----
INPUT: Shall we go for lunch?
THOUGHT PROCESS: The text is an invitation to a social activity, which is typical of casual chat.
OUTPUT: False
-----
INPUT: Introduce the movie Inception
THOUGHT PROCESS: The text is asking for information about a movie, which is more related to entertainment rather than academic topics.
OUTPUT: False
-----
INPUT: What is perceptron?
THOUGHT PROCESS: The text is asking for a definition of a technical term commonly used in machine learning, which is an academic inquiry in computer science field.
OUTPUT: True
-----
INPUT: Introduce self-attention mechanism
THOUGHT PROCESS: The text is asking for an explanation of a concept in deep learning, which is an academic inquiry in computer science field.
OUTPUT: True
-----
INPUT: Why did Llava perform so well?
THOUGHT PROCESS: The text is asking for an analysis of the performance of a model, which is an academic inquiry in computer science field.
OUTPUT: True
-----
INPUT: How do multimodal language models work?
THOUGHT PROCESS: The text is asking for an explanation of how a type of language model functions, which is an academic inquiry in computer science field.
OUTPUT: True
-----
INPUT: I want to know about the history of the Chinese language.
THOUGHT PROCESS: Although the text is an academic inquiry, it is not a question about computer science.
OUTPUT: False
-----\n
"""

context_based_qa_prompt = '''
GOAL: Provide a detailed and accurate answer to a given question based on the provided context.
-----
INSTRUCTION:
1. Carefully read the provided context and the question.
2. Use only the information in the context to answer the question. Avoid adding external knowledge or assumptions.
3. Ensure the answer is clear, concise, and directly addresses the question.

SCHEMA:
INPUT:
- **Context**: A piece of text that contains the information needed to answer the question.
- **Question**: A specific query that must be answered using the context.

OUTPUT:
- **Answer**: A clear and precise response to the question, derived solely from the context.
-----
EXAMPLE:
INPUT:
**Context**: Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work also contributed significantly to the development of quantum mechanics. Einstein was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect, a pivotal step in the development of quantum theory.

**Question**: For what achievement was Albert Einstein awarded the Nobel Prize in Physics?

OUTPUT:
**Answer**: Albert Einstein was awarded the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.
-----
EXAMPLE:
INPUT:
**Context**: The Amazon rainforest is the largest rainforest on Earth, covering approximately 5.5 million square kilometers. It is often referred to as the "lungs of the planet" because it produces about 20% of the world's oxygen. The rainforest is home to millions of species, many of which are not found anywhere else in the world.

**Question**: Why is the Amazon rainforest referred to as the "lungs of the planet"?

OUTPUT:
**Answer**: The Amazon rainforest is referred to as the "lungs of the planet" because it produces about 20% of the world's oxygen.
-----
INPUT:
**Context**: {}
**Question**: {}

OUTPUT:
**Answer**:
'''