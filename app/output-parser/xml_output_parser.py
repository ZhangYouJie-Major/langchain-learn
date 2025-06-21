from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser,XMLOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()

model = ChatOpenAI(model="deepseek-chat")

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")




json_output_parser = XMLOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template='回答用户的查询。\n{format_instructions}\n{question}',
    input_variables=['question'],
    partial_variables={'format_instructions': json_output_parser.get_format_instructions()}
)

chain = prompt | model 

result = chain.invoke({'question': '给我中文讲一个冷笑话'})
print(result.content)