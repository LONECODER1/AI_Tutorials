from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI Embeddings model
emb = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# embed_query embeds a single query string
query = "What is the capital of India?"
result = emb.embed_query(query)

print("Embedding vector length:", len(result))
print("Vector preview (first 5 dimensions):", result[:5])
