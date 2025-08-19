import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from robot.llm import init_llm
from toolbox import tavily_search


def build_runnable_llm():
    llm = init_llm()
    tools = [tavily_search]
    llm_with_tool = llm.bind_tools(tools)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior international travel guide, well-versed in the local customs and cultures of countries across the globe. You excel at planning travel routes, with particular expertise in travel to Japan. You will provide the best travel itineraries based on the number of travelers, travel theme, travel duration, and travel budget.",
            ),
            ("user", "{query}"),
        ]
    )
    runnable = prompt | llm_with_tool | StrOutputParser()
    return runnable


@cl.on_chat_start
async def on_chat_start():
    runnable = build_runnable_llm()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable: Runnable = cl.user_session.get("runnable")  # type: ignore
    assert runnable is not None, "Runnable not found in user session"

    msg = cl.Message(content="")

    async for chunk in runnable.astream(  # type: ignore
        {"query": message.content},
        config=RunnableConfig(),
    ):
        await msg.stream_token(chunk)

    await msg.send()
