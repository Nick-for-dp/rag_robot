from langchain.chains.retrieval_qa.base import RetrievalQA

from robot.document import init_vector_store
from robot.llm import init_llm


def main(document_file_path: str):
    vector_store = init_vector_store(pdf_file_path=document_file_path)
    llm = init_llm()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                           chain_type="stuff",
                                           retriever=vector_store.as_retriever(search_kwargs={"k": 2})
                                           )
    query = "How many distribution centers does Nike have in the US?"
    result = qa_chain.invoke({"query": query})
    print(result)


if __name__ == "__main__":
    pdf_file_path = r'D:/projects/rag_robot/robot/document/nke-10k-2023.pdf'
    main(document_file_path=pdf_file_path)
