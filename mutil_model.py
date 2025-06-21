import base64
import httpx
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量


def image_url_to_base64(image_url):
    """
    将图片URL转换为base64编码
    """
    response = httpx.get(image_url)
    response.raise_for_status()  # 确保请求成功
    return base64.b64encode(response.content).decode("utf-8")

def main():
    # 图片地址
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1280px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    # 将图片转换为base64编码
    base64_image = image_url_to_base64(image_url)

    # 初始化大模型
    # 需要在 .env 文件中设置 OPENAI_API_KEY
    # 如果使用了代理，还需要设置 OPENAI_BASE_URL
    model = ChatOpenAI(model="gpt-4o", temperature=0,base_url='https://api.gptsapi.net/v1')

    # 构建消息
    message = HumanMessage(
        content=[
            {"type": "text", "text": "这张图片里有什么内容？"},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
            },
        ]
    )

    # 调用大模型
    result = model.invoke([message])

    # 打印结果
    print("大模型识别内容如下：")
    print(result.content)

if __name__ == "__main__":
    load_dotenv()
    main()


