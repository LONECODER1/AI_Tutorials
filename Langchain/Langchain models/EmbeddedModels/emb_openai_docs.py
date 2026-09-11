from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI Embeddings model
emb = OpenAIEmbeddings(
    model="text-embedding-3-small"
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
