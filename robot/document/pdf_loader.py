import os
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf(pdf_file_path: str):
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"Do not found pdf file in path:{pdf_file_path}")
    # 读取PDF文档
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    # 文档分段
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits


def init_embedding_model(model_name: Optional[str] = None):
    if not model_name:
        model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    # huggingface 镜像地址
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    return HuggingFaceEmbeddings(model_name=model_name)


if __name__ == '__main__':
    pdf_file_path = r'document/nke-10k-2023.pdf'
    splits = load_pdf(pdf_file_path=pdf_file_path)
    print(f"There has {len(splits)} splits.")
    print(f"{splits[0].page_content}\n")
    embedding_model = init_embedding_model()
    vector_1 = embedding_model.embed_query(splits[0].page_content)
    vector_2 = embedding_model.embed_query(splits[1].page_content)
    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])
