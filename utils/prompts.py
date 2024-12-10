query_clearification_instruction = (
    "Rewrite the given query to make it more concise, specific and logical while correct grammatical mistakes or typos in style of question: ```{{query}}```\n\n"
    "EXAMPLE:\n------------------------\nINPUT:\n"
    "Please rewrite the query: ```gradient descent optimize```\n"
    "\nOUTPUT:\n"
    "How does gradient descent perform optimization?\n\n"
    
    "Please rewrite the query: ```'Stereo Anywhere' aplication```\n"
    "\nOUTPUT:\n"
    "What is the application of 'Stereo Anywhere'?"
)

query_generation_instruction = (
    f"Please generate five diverse questions based on the given text: ```{{text}}```. "
    '"Diverse" means the questions should be different from each other in terms of wording, level of description clarity, completeness, and phrasing. For example, some questions can be very detailed and specific, while others can be more general and open-ended. '
    "The questions may even include intentional misspellings, grammatical errors, or misleading words to add to the diversity. "
    "Ensure that the questions are distinct and cover a range of aspects related to the text.\n\n"
    "EXAMPLE:\n------------------------\nINPUT:\n"
    "Please generate five diverse questions based on the text: ```Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail```\n"
    
    "\nOUTPUT:\n"
    "Can you explain what is meant by 'zero-shot' in the context of deep stereo matching and how it contributes to the robustness of the 'Stereo Anywhere' method?\n"
    "How does the 'Stereo Anywhere' technique ensure robust performance even in scenarios where traditional stereo or monocular methods fail?\n"
    "In what ways does 'Stereo Anywhere' differ from other deep stereo matching techniques, particularly in terms of robustness and zero-shot learning capabilities?\n"
    "Can u giv som examples of real-world aplications where 'Stereo Anywhere' would be really usefull, especialy in situtations were other methods might fail?\n"
    "What are the potential drawbacks or obstacles of the 'Stereo Anywhere' approach, and how might these be overcome in future research to ensure even greater reliability and accuracy in stereo matching?\n\n"
)

keyword_extraction_instruction = (
    "Extract the 3 most important keywords from a given query, place them in descending order of their importance, and separate them by ';': ```text```\n\n"
    "EXAMPLE:\n------------------------\nINPUT:\n"
    "Please extract 3 most important keywords based on the text: ```Can you explain the concept of 'Perturb-and-Revise' in the context of 3D editing and how it addresses the limitations of existing methods in modifying geometry and appearance?```\n"
    
    "\nOUTPUT:\n"
    "Perturb-and-Revise;3D editing;geometry\n\n"
)

text_isRel_instruction = '''
You are an expert in reading comprehension and logical reasoning. Your task is to give a clear, logical, and step-by-step explanation for the following conclusion about whether a given content is related to a specific question. Please think carefully and provide a concise but logical explanation.

Objective: Determine whether a given content is related to a specific question.

You are provided with the following inputs:
1. Content: {A piece of text provided to you.}
2. Question: {A specific question that may or may not be related to the text.}

Based on this, provide a clear explanation with logical steps. Your response should include:
1. Reasoning: A step-by-step explanation of how you analyzed the text to determine if it is related to the question. It should end with '*' as a complete indication.
2. Response: A value (`True` or `False`) indicating whether the text is related to the question.

- **True**: The text is related and can help answer the question.
- **False**: The text is not related to the question.

Format your output as a JSON object:

-----
SCHEMA
-----
{
    "Reasoning": "Step-by-step reasoning explaining why the text is or isn't related to the question.",
    "Response": "True"/"False"
}

-----
EXAMPLE
-----

1. Content: Shigeru Miyamoto (Japanese: 宮本 茂, Hepburn: Miyamoto Shigeru, born November 16, 1952) (pronounced [mʲijamoto ɕiŋeɾɯ̥; ɕiɡeɾɯ̥]) is a Japanese video game designer and producer, currently serving as the co-Representative Director of Nintendo. He is best known as the creator of some of the most critically acclaimed and best-selling video games and franchises of all time, such as Donkey Kong, Mario, The Legend of Zelda, Star Fox, F-Zero, and Pikmin.
2. Question: Who is a Japanese video game designer and producer, currently serving as the co-Representative Director of Nintendo, who gave production input to a video game for the Nintendo 64, originally released in 1996 along with the debut of the console?

Output:
{
    "Reasoning": "Step 1: The question asks for a Japanese video game designer and producer who is currently serving as the co-Representative Director of Nintendo and contributed to a game released in 1996 for the Nintendo 64. Step 2: The content introduces Shigeru Miyamoto, a Japanese video game designer and producer who holds the position of co-Representative Director at Nintendo, matching part of the question's criteria. Step 3: While the content does not directly mention Miyamoto's involvement in a specific game from 1996, his significant role in the creation of major games for Nintendo suggests relevance to the query. Step 4: The content aligns with the information asked in the question. Step 5: Therefore, the content is sufficiently related to the question.*", 
    "Response": "True" 
}

-----

1. Content: The provided image features the iconic logo of the Nintendo 64, a home video game console developed and marketed by Nintendo. The logo prominently displays the word ""NINTENDO"" in bold, blue capital letters, followed by the number ""64"" in red, signifying the console's 64-bit processor. Below the text, there is a three-dimensional, multicolored ""N"" composed of four interlocking shapes, each side colored differently in blue, green, red, and yellow. This design element not only emphasizes the brand but also symbolizes the innovative and playful nature of the console. The Nintendo 64, often abbreviated as N64, was known for its groundbreaking graphics and influential games, cementing its legacy in the gaming industry. The logo's vibrant colors and geometric design reflect the console's emphasis on advanced technology and entertainment.
2. Question: Who is a Japanese video game designer and producer, currently serving as the co-Representative Director of Nintendo, who gave production input to a video game for the Nintendo 64, originally released in 1996 along with the debut of the console?

Output:
{
    "Reasoning": "Step 1: The question asks for the name of a Japanese video game designer and producer who contributed to a Nintendo 64 game released in 1996 and currently serves as the co-Representative Director of Nintendo. Step 2: The content describes the Nintendo 64 logo and emphasizes the console's legacy but does not mention any individuals or specific contributors to the games. Step 3: While the content provides relevant information about the Nintendo 64 itself, it does not directly address the question about the producer. Step 4: Since the required information about a specific individual is missing, the content is not related to the question.*", 
    "Response": "False"
}

-----

'''

