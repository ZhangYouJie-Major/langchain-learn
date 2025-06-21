from langchain_core.tools import StructuredTool, ToolException
from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
import os
import requests
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 加载.dev
load_dotenv()

llm = ChatOpenAI(model="deepseek-chat", temperature=0)


def get_current_time() -> str:
    """获取当前时间"""
    logging.info("get_current_time called")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def get_current_time_async() -> str:
    """获取当前时间"""
    logging.info("get_current_time_async called")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_hefeng_weather(city: str) -> str:
    """获取一个城市的实时天气"""
    api_key = os.getenv("HEFENG_WEATHER_API_KAY")
    base_url = os.getenv("HEFENG_WEATHER_API_URL")

    if not api_key or not base_url:
        logging.error(
            "HEFENG_WEATHER_API_KAY or HEFENG_WEATHER_API_URL not set in .env file."
        )
        raise ToolException(
            "请确保在.env文件中设置了 HEFENG_WEATHER_API_KAY 和 HEFENG_WEATHER_API_URL"
        )

    if "://" in base_url:
        logging.warning(
            f"HEFENG_WEATHER_API_URL ({base_url}) seems to contain a protocol, usually only the domain is needed."
        )

    # 1. 获取城市ID
    lookup_url = f"https://{base_url}/geo/v2/city/lookup?location={city}&key={api_key}"
    try:
        logging.info(f"Looking up city ID for {city}")
        response = requests.get(lookup_url)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != "200" or not data.get("location"):
            logging.error(f"Could not find city: {city}. API response: {data}")
            raise ToolException(f"找不到城市: {city}。API响应: {data}")

        location_id = data["location"][0]["id"]
        logging.info(f"Found location ID for {city}: {location_id}")

        # 2. 获取天气信息
        weather_url = (
            f"https://{base_url}/v7/weather/now?location={location_id}&key={api_key}"
        )
        logging.info(f"Fetching weather for location ID: {location_id}")
        response = requests.get(weather_url)
        response.raise_for_status()
        weather_data = response.json()

        if weather_data.get("code") != "200":
            logging.error(f"Failed to get weather for {city}: {weather_data}")
            raise ToolException(
                f"获取 {city} 天气失败: {weather_data.get('message', '未知错误')}"
            )

        now = weather_data.get("now", {})
        temp = now.get("temp")
        text = now.get("text")
        wind_dir = now.get("windDir")

        result = f"{city}的实时天气：{text}，气温：{temp}℃，风向：{wind_dir}。"
        logging.info(f"Successfully fetched weather for {city}: {result}")
        return result

    except requests.exceptions.RequestException as e:
        logging.error(f"Error requesting weather API: {e}", exc_info=True)
        raise ToolException(f"请求天气API时出错: {e}")
    except (KeyError, IndexError) as e:
        logging.error(f"Error parsing weather API response: {e}", exc_info=True)
        raise ToolException(f"解析天气API响应时出错: {e}")


tools = [
    StructuredTool.from_function(
        func=get_current_time,
        name="get_current_time",
        description="获取当前时间",
        handle_tool_error=True,
        coroutine=get_current_time_async,
    ),
    StructuredTool.from_function(
        func=get_hefeng_weather,
        name="get_hefeng_weather",
        description="获取一个城市的实时天气",
        handle_tool_error=True,
    ),
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can use tools to answer questions.",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


async def main():
    response = agent_executor.invoke({"input": "北京的天气怎么样？"})
    logging.info(f"Final response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
