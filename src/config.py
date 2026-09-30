import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index-384")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
LLM_MODEL = "openai/gpt-oss-20b"
PDF_PATH = "data/Ebook-Agentic-AI.pdf"
TOP_K = 6
MIN_RELEVANCE = 0.25