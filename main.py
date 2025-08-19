from langchain.chains.retrieval_qa.base import RetrievalQA

from robot.document import init_vector_store
from robot.llm import init_llm
from robot.toolbox import tavily_search


def main():
    # vector_store = init_vector_store(pdf_file_path=document_file_path)
    llm = init_llm()
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, 
    #                                        chain_type="stuff",
    #                                        retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    #                                        )
    tools = [tavily_search]
    llm_with_tool = llm.bind_tools(tools=tools)
    query = "推荐日本大阪好玩的三个景点。"
    result = llm_with_tool.invoke(query)
    print(result)


if __name__ == "__main__":
    main()
