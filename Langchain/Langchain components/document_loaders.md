# LangChain Component: Document Loaders

Language models are only as good as the information they have access to. Most enterprise knowledge is locked inside PDFs, Word documents, CSV spreadsheets, websites, Notion pages, and SQL databases.

**Document Loaders** are LangChain's data ingestion gateways. They extract data from hundreds of different sources and normalize them into a uniform `Document` object that the rest of LangChain can process.

---

## 1. The Core `Document` Object

Every document loader returns a list of standard LangChain `Document` objects containing two attributes:

```python
Document(
    page_content="The raw extracted textual contents of the document...",
    metadata={
        "source": "cricket_rules.pdf",
        "page": 4,
        "author": "ICC",
        "created_at": "2024-01-15"
    }
)
```

- **`page_content`**: The raw text that will be chunked, embedded, and passed to LLMs.
- **`metadata`**: A dictionary storing provenance data (file source, page number, title, URLs), allowing filtered searches and source citations in final answers.

---

## 2. Common Document Loaders

### A. `TextLoader` (Plain Text Files)
Loads `.txt`, `.md`, or code files into memory.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("notes.txt", encoding="utf-8")
docs = loader.load()

print(f"Loaded {len(docs)} document.")
print(f"Content preview: {docs[0].page_content[:100]}")
print(f"Metadata: {docs[0].metadata}")
```

---

### B. `PyPDFLoader` (PDF Documents)
Extracts text from PDF files, automatically splitting the output into one `Document` per PDF page:

```python
from langchain_community.document_loaders import PyPDFLoader

# Initialize loader with local file or path
loader = PyPDFLoader("annual_report.pdf")

# Returns a list of Document objects (one per page)
pages = loader.load()

print(f"Total pages extracted: {len(pages)}")
print(f"Page 1 Content:\n{pages[0].page_content[:200]}")
print(f"Page 1 Metadata:\n{pages[0].metadata}")  # e.g., {'source': 'annual_report.pdf', 'page': 0}
```

---

### C. `WebBaseLoader` (HTML Web Scraping)
Extracts and parses clean text content from public web URLs using BeautifulSoup:

```python
from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://en.wikipedia.org/wiki/Virat_Kohli",
    "https://en.wikipedia.org/wiki/Rohit_Sharma"
]

loader = WebBaseLoader(urls)
web_docs = loader.load()

print(f"Extracted {len(web_docs)} web pages.")
print(f"Title: {web_docs[0].metadata.get('title')}")
```

---

### D. `CSVLoader` (Tabular Data)
Loads CSV spreadsheets by treating each row as an individual `Document` with column-value formatting:

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="cricketers.csv", source_column="Player_Name")
row_docs = loader.load()

# Example output for a single row:
# Player_Name: Virat Kohli
# Runs: 13906
# Average: 58.18
print(row_docs[0].page_content)
```

---

### E. `DirectoryLoader` (Batch Ingestion)
Scans an entire directory tree and loads multiple file formats matching glob patterns:

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load all markdown files in a folder recursively
dir_loader = DirectoryLoader(
    path="./docs",
    glob="**/*.md",
    loader_cls=TextLoader
)

all_docs = dir_loader.load()
print(f"Loaded {len(all_docs)} files from directory.")
```

---

## 3. Cloud Storage & S3 Ingestion (As in RAG Architecture)

As illustrated in RAG architectures (e.g., `image2.png`), files often live in cloud object storage like **AWS S3**:

```python
from langchain_community.document_loaders import S3FileLoader

# Load directly from an AWS S3 bucket
s3_loader = S3FileLoader(
    bucket="my-enterprise-ai-bucket",
    key="research_papers/transformers.pdf"
)

documents = s3_loader.load()
```

---

## 4. Best Practices

1. **Lazy Loading (`.lazy_load()`)**: When processing thousands of files or gigabytes of PDFs, use `.lazy_load()` instead of `.load()`. This yields an iterator of documents to avoid out-of-memory (OOM) crashes.
2. **Preserve Metadata**: Always ensure your loader captures relevant metadata (e.g., page numbers, URLs) to enable accurate source attribution in RAG responses.
3. **Handle Encodings**: Specify `encoding="utf-8"` for text files to prevent decoding errors across different operating systems.
