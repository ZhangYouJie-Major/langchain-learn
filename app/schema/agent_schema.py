from pydantic import BaseModel


class AgentInput(BaseModel):
    question: str  # 与 AgentExecutor.input_keys 对应


class AgentOutput(BaseModel):
    output: str  # 与 AgentExecutor.outpu t_keys 对应


