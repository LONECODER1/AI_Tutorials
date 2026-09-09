# LangChain Component: Embeddings

Embeddings create vector representations of a piece of text. In LangChain, the **`Embeddings`** class provides a consistent, standardized interface for interacting with different text embedding models (OpenAI, HuggingFace, Google, Cohere, Ollama, etc.).

---

## 1. The Core Methods: `embed_documents` vs. `embed_query`

The `Embeddings` base class defines two distinct methods for generating vectors:

```python
# 1. Used for embedding multiple document chunks (batch ingestion)
doc_vectors = embedding_model.embed_documents(["chunk 1 text", "chunk 2 text"])
# Returns: List[List[float]]

# 2. Used for embedding a single user query at runtime
query_vector = embedding_model.embed_query("User's question")
# Returns: List[float]
```

### Why are there two separate methods?
Certain embedding models (e.g., Cohere, Instructor, E5) use different prompting architectures under the hood for *documents to be indexed* versus *search queries to retrieve them*. Having dedicated methods allows LangChain to automatically apply provider-specific prefixes (like `"search_query:"` or `"search_document:"`).

---

## 2. Supported Embedding Providers

### A. OpenAI Embeddings (Cloud Standard)
Uses models like `text-embedding-3-small` (1536 dimensions) or `text-embedding-3-large` (3072 dimensions):

```python
from langchain_openai import OpenAIEmbeddings

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Generate embedding for a query
query_vector = embeddings.embed_query("How many runs has Virat scored?")
print(f"Vector Length: {len(query_vector)}")  # 1536
print(f"Sample values: {query_vector[:5]}")
```

---

### B. HuggingFace / Local Open-Source Embeddings (Free, No API Keys)
Run embeddings locally on your own CPU or GPU using open-source models like `all-MiniLM-L6-v2`:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Download and run model locally
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector = embeddings.embed_query("Machine learning algorithms")
print(f"Vector dimensions: {len(vector)}")  # 384
```

---

### C. Google Generative AI Embeddings
Uses Google Gemini embedding models:

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector = embeddings.embed_query("Cricket batting records")
```

---

## 3. Practical Example: Computing Vector Similarity Manually

This example demonstrates the exact mathematics behind Image 1:

```python
import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 1. Embed query
query = "How many runs has Virat scored?"
query_vec = np.array(embeddings.embed_query(query))

# 2. Embed document chunks
docs = [
    "Virat Kohli has scored over 26,000 international runs across all formats.",
    "Jasprit Bumrah took 5 wickets in the final test match with his yorkers.",
    "Rohit Sharma scored a double hundred against Sri Lanka."
]
doc_vecs = [np.array(vec) for vec in embeddings.embed_documents(docs)]

# 3. Cosine similarity function
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 4. Compare query against all documents
print(f"Query: '{query}'\n")
for doc, doc_vec in zip(docs, doc_vecs):
    similarity = cosine_similarity(query_vec, doc_vec)
    print(f"Similarity: {similarity:.4f} -> Doc: '{doc[:50]}...'")

# The Virat Kohli chunk will score the highest similarity!
```

---

## 4. Performance Optimization: `CacheBackedEmbeddings`

Re-computing embeddings for identical documents on every app run wastes money and slows down startup. LangChain offers **CacheBackedEmbeddings** to store vectors in a local key-value store (e.g., SQLite, Redis, or filesystem):

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

underlying_embeddings = OpenAIEmbeddings()
store = LocalFileStore("./cache_dir/")

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)

# First run calculates and caches; subsequent calls load from disk instantly!
vectors = cached_embeddings.embed_documents(["Paragraph 1", "Paragraph 2"])
```

---

## 5. Summary Checklist

- [x] Use `OpenAIEmbeddings` for state-of-the-art managed cloud performance.
- [x] Use `HuggingFaceEmbeddings` for privacy-sensitive or offline/free local deployments.
- [x] Always use the **exact same embedding model** for both document ingestion and query retrieval. (Mismatched models break similarity search).
