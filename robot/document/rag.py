from uuid import uuid4
from typing import Optional, List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from robot.document.pdf_loader import load_pdf, init_embedding_model

def init_vector_store(pdf_file_path: str,
                    embedding_model_name: Optional[str] = None,
                    collection_name: str = 'nike_knowledge_lib',
                    chroma_db_path: str = './chroma_langchain_db') -> Chroma:
    """Initialize Chroma vector store with PDF documents"""
    all_splits = load_pdf(pdf_file_path=pdf_file_path)
    # load embedding model
    embedding_model = init_embedding_model(model_name=embedding_model_name)
    # create vector store
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=chroma_db_path
    )
    ids = [str(uuid4()) for _ in range(len(all_splits))]
    _ = vector_store.add_documents(documents=all_splits, ids=ids)
    return vector_store

def semantic_search(user_input: str, vector_store: Chroma) -> List[Document]:
    """Perform semantic search on vector store"""
    if not user_input:
        raise ValueError("Input cannot be empty")
    return vector_store.similarity_search(user_input, k=3)

if __name__ == '__main__':
    pdf_file_path = r'../document/nke-10k-2023.pdf'
    print("Starting RAG pipeline...")
    try:
        vector_store = init_vector_store(pdf_file_path=pdf_file_path, chroma_db_path='../vec_db')
        user_input = "How many distribution centers does Nike have in the US?"
        results = semantic_search(user_input=user_input, vector_store=vector_store)
        print(results)
    except Exception as e:
        print(f"Error: {str(e)}")
