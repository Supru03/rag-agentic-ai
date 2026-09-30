import time
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src import config


def ensure_index():
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(config.INDEX_NAME).status["ready"]:
            time.sleep(1)


def run_ingestion():
    ensure_index()
    docs = PyPDFLoader(config.PDF_PATH).load()
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)
    PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=config.INDEX_NAME
    )
    print("Ingestion complete")


if __name__ == "__main__":
    run_ingestion()