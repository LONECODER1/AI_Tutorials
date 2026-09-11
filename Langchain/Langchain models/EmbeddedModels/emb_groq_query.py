from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI Embeddings model
emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # dimensions=1536  # text-embedding-3-small supports custom dimensions (e.g., 256, 512, 1536)
)

# Embed a single query
result = emb.embed_query("What is the capital of India?")

print("Embedding vector length:", len(result))
print("Vector preview (first 5 dimensions):", result[:5])
