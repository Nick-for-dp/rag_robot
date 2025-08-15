from typing import Optional, List
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_core.documents import Document

from robot.document import load_pdf, init_embedding_model


# TODO：向量库改为硬盘存储
def init_vector_store(pdf_file_path: str, embedding_model_name: Optional[str] = None) -> InMemoryVectorStore:
    all_spilts = load_pdf(pdf_file_path=pdf_file_path)
    embedding_model = init_embedding_model(model_name=embedding_model_name)
    vector_store = InMemoryVectorStore(embedding=embedding_model)
    _ = vector_store.add_documents(documents=all_spilts)
    return vector_store


def semantic_search(user_input: str, vector_store: VectorStore) -> List[Document]:
    if not user_input or user_input == '':
        raise ValueError("Input should not be none or empty string.")
    results = vector_store.similarity_search(user_input, k=3)
    return results


if __name__ == '__main__':
    pdf_file_path = r'document/nke-10k-2023.pdf'
    vector_store = init_vector_store(pdf_file_path=pdf_file_path)
    user_input = "How many distribution centers does Nike have in the US?"
    results = semantic_search(user_input=user_input, vector_store=vector_store)
    print(results)
