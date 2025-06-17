import os
from pydantic import SecretStr
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, FewShotPromptTemplate, \
    MessagesPlaceholder, SemanticSimilarityExampleSelector
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

openai_api_key = SecretStr("sk-6cc44ac6764046bbb5520195e40b14aa")
open_ai_url = "https://api.deepseek.com"

os.environ["OPENAI_API_KEY"] = 'sk-VHB442318ebe5ed35b009d91d1f48c571c46384c7b8LAiB5'
os.environ["OPENAI_API_BASE"] = 'https://api.gptsapi.net'

examples = [
    {
        "question": "谁的寿命更长,汉武帝和秦始皇",
        "answer": "汉武帝的寿命更长"

    },
    {
        "question": "地球公转一周需要多久",
        "answer": "地球公转一周需要365天"
    }]

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(
        content='你是一个ai问答机器人'
    ),
    HumanMessage(
        content="You are a helpful assistant",
    ),
    # HumanMessagePromptTemplate.from_template("translates {input_language} to {output_language}."),
    # HumanMessagePromptTemplate.from_template("{text}"),
    MessagesPlaceholder("msgs")

])

example_prompt = PromptTemplate(input_variables=["question", "answer"], template="问题:{question}\r\n{answer}")

print(example_prompt.format(**examples[0]))

prompt_1 = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="问题:{input}",
    input_variables=["input"]
)

# print(prompt_1.format(input="地球公转一周需要多久"))

messages = prompt.invoke({"msgs": [HumanMessage(content='翻译官'), HumanMessage(content='翻译官1')]})

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    Chroma,
    k=1
)
question = '汉武帝和秦始皇'
selected = example_selector.select_examples({"question": question})
print(f"最相似的例子:{question}")
for example in selected:
    print('\r\n')
    for k, v in example.items():
        print(f"{k}:{v}")
