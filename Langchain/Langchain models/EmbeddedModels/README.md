# LangChain Embedding Models

In LangChain, embedding models convert text strings into dense numerical vectors that capture semantic meaning.

## Key Methods

Every LangChain embedding model provides two primary methods:

1. **`embed_query(text: str) -> List[float]`**
   - Used to embed a single query or search prompt.
   - Example: Embedding a user's question before searching a vector database.

2. **`embed_documents(texts: List[str]) -> List[List[float]]`**
   - Used to embed multiple text chunks or documents at once in batch.
   - Example: Embedding parsed PDF pages or knowledge-base documents to store into a vector store.

---

## File Structure

* **[emb_openai_query.py](file:///c:/Users/asus/Desktop/AI_Tutorials/Langchain/Langchain%20models/EmbeddedModels/emb_groq_query.py)**: Demonstrates `embed_query` with OpenAI.
* **[emb_openai_docs.py](file:///c:/Users/asus/Desktop/AI_Tutorials/Langchain/Langchain%20models/EmbeddedModels/emb_openai_docs.py)**: Demonstrates `embed_documents` with OpenAI.
* **[emb_hf_docs.py](file:///c:/Users/asus/Desktop/AI_Tutorials/Langchain/Langchain%20models/EmbeddedModels/emb_hf_docs.py)**: Demonstrates `embed_documents` with Hugging Face (Free / No credit card required).

---

## Dimensions Comparison

| Provider | Model | Dimensions | Notes |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `text-embedding-3-small` | 1536 (customizable) | High accuracy, requires paid API key |
| **Hugging Face** | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast, free serverless endpoint |
