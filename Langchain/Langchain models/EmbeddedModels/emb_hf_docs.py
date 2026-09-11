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

# List of documents to embed
documents = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Paris is the capital of France."
]

# embed_documents takes a list of strings and returns a list of embedding vectors (List[List[float]])
result = emb.embed_documents(documents)

print(f"Number of documents embedded: {len(result)}")
print(f"Embedding dimensions per document: {len(result[0])}")
print("First document vector preview (first 5 dimensions):", result[0][:5])
