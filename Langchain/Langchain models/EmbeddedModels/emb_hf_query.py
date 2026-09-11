import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize Hugging Face Endpoint Embeddings
emb = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=hf_token
)

# embed_query embeds a single query string
query = "What is the capital of India?"
result = emb.embed_query(query)

print("Embedding vector length:", len(result))
print("Vector preview (first 5 dimensions):", result[:5])
