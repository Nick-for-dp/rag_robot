from langchain_deepseek import ChatDeepSeek
from robot.utils import get_api_key


def init_llm() -> ChatDeepSeek:
    deepseek_api_key = get_api_key()
    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    llm = ChatDeepSeek(
        model="deepseek-reasoner",
        temperature=0,
        max_tokens=2048,
        timeout=None,
        max_retries=2,
        api_key=deepseek_api_key # pyright: ignore[reportArgumentType]
    )
    return llm


if __name__ == '__main__':
    llm = init_llm()
    print(type(llm))
