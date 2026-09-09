# LangChain Component: Vector Stores & Retrievers

Once document chunks are converted into numerical embedding vectors, they must be stored in specialized databases that support efficient nearest-neighbor searches across high-dimensional spaces. 

In LangChain, this capability is divided into two closely related components:
1. **Vector Stores**: Databases that index and store vectors and metadata.
2. **Retrievers**: Lightweight Runnable interfaces that accept a string query and return a list of matching `Document` objects.

---

## 1. Vector Stores

A **Vector Store** (or Vector Database) is optimized for Approximate Nearest Neighbor (ANN) search algorithms (such as HNSW, IVF-PQ) rather than traditional SQL keyword indexing.

### Popular Vector Stores in LangChain:
- **FAISS (Facebook AI Similarity Search)**: Extremely fast, lightweight, in-memory or on-disk vector index. Perfect for local prototyping.
- **Chroma**: Embeddable, open-source vector store with native persistence and filtering.
- **Pinecone / Qdrant / Milvus**: Enterprise cloud-native vector databases capable of scaling to billions of vectors with high concurrency.

### Code Example: Creating and Querying a FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

texts = [
    "Virat Kohli scored 76 runs in the T20 World Cup final in 2024.",
    "Jasprit Bumrah was awarded the Player of the Tournament for his bowling.",
    "Rohit Sharma retired from T20 internationals after captaining India to victory.",
    "Sunil Narine won MVP for Kolkata Knight Riders in IPL 2024."
]

embeddings = OpenAIEmbeddings()

# 1. Create vector store directly from text chunks
vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)

# 2. Similarity search with raw query
results = vectorstore.similarity_search("Who bowled well in the T20 World Cup?", k=2)

for doc in results:
    print("-", doc.page_content)
```

---

## 2. Retrievers: The Universal Retrieval Interface

While a vector store requires vector manipulation and returns specific database types, a **Retriever** is a higher-level LangChain component designed to plug cleanly into LCEL chains.

Any vector store can be converted into a Retriever using `.as_retriever()`:

```python
# Convert vector store to retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# A retriever is a Runnable: accepts a string, returns List[Document]
matching_docs = retriever.invoke("Who was the captain in 2024?")
print(matching_docs[0].page_content)
```

---

## 3. Advanced Retrieval Strategies

Simple cosine similarity often retrieves redundant chunks that repeat identical sentences. LangChain retrievers offer several advanced algorithms:

### A. Maximal Marginal Relevance (MMR)
MMR seeks to optimize both **relevance** to the query and **diversity** among the retrieved results. It prevents retrieving 3 nearly identical sentences from the same page:

```python
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # Number of documents to return
        "fetch_k": 10,    # Number of documents to fetch before reranking
        "lambda_mult": 0.5 # 0 = max diversity, 1 = max similarity
    }
)

docs = mmr_retriever.invoke("cricket records")
```

---

### B. Similarity Score Threshold
Only returns documents whose similarity score exceeds a specified confidence cut-off, avoiding low-quality irrelevant context:

```python
threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.75}
)
```

---

### C. Multi-Query Retriever
Uses an auxiliary LLM to rephrase a single user query into multiple alternative perspectives, retrieving documents for all variations to overcome phrasing bias:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# User asks "Kohli achievements", LLM generates 3 variations:
# 1. "Virat Kohli major cricket records"
# 2. "Centuries and awards won by Virat Kohli"
# 3. "Kohli career milestones"
docs = multi_retriever.invoke("Kohli achievements")
```

---

## 4. Summary Matrix

| Retrieval Technique | When to Use | Advantage |
| :--- | :--- | :--- |
| **Standard Similarity (`similarity`)** | Simple RAG applications | Fast, low latency |
| **MMR (`mmr`)** | Repetitive or large documents | Minimizes redundant context |
| **Score Threshold (`similarity_score_threshold`)** | Strict accuracy requirements | Filters out low-confidence hallucinations |
| **Multi-Query** | Complex or ambiguously worded questions | High recall across different vocabularies |
