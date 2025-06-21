from langchain_openai import ChatOpenAI
from langchain.output_parsers import YamlOutputParser
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model="deepseek-chat")

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# The output parser can be used to get format instructions and also to parse the
# output.
yaml_output_parser = YamlOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="回答用户的查询。\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": yaml_output_parser.get_format_instructions()},
)

# Pipe the parser to the chain.
chain = prompt | model

result = chain.invoke({"question": "给我中文讲一个冷笑话"})

print(result.content) 