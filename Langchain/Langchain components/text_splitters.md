# LangChain Component: Text Splitters (Chunking)

Raw documents loaded by document loaders are frequently too large to be embedded effectively or passed into LLM prompts in their entirety. For example, a 50-page PDF contains far too many ideas to be represented by a single embedding vector without washing out specific details.

**Text Splitters** (or **Chunkers**) break large continuous documents into smaller, semantically coherent passages called **chunks**, preparing them for vectorization and retrieval.

---

## 1. Why is Chunking Necessary?

1. **Embedding Precision**: An embedding model produces a vector of fixed length. If you embed an entire 10,000-word document, nuanced details (like a single cricket statistic or date) become diluted. Small chunks preserve granular semantic meaning.
2. **Context Window Efficiency**: Retrieving 3 targeted 200-word paragraphs uses a fraction of the LLM context window compared to loading entire chapters.
3. **Cost Reduction**: Fewer tokens sent to LLMs equals dramatically lower API costs and faster generation speed.

---

## 2. Core Splitting Parameters: Size & Overlap

Every text splitter relies on two primary tuning levers:

```
Full Text: [ ------------------------------------------------------------- ]

Chunk 1:   [ ========== ] (chunk_size = 500)
Chunk 2:          [ ========== ] (chunk_overlap = 100)
Chunk 3:                 [ ========== ]
```

- **`chunk_size`**: The maximum size (in characters or tokens) of each resulting chunk.
- **`chunk_overlap`**: The number of characters/tokens shared between adjacent chunks. Overlap prevents critical context from being split in half across chunk boundaries (e.g., a sentence cut down the middle).

---

## 3. Core Text Splitters

### A. `RecursiveCharacterTextSplitter` (The Industry Standard)
The most recommended splitter for generic text. It tries to split text hierarchically using a prioritized list of natural separators:
1. `"\n\n"` (Paragraph breaks)
2. `"\n"` (Line breaks)
3. `" "` (Word spaces)
4. `""` (Characters as a last resort)

This hierarchy keeps paragraphs and complete sentences together as long as possible before resorting to mid-sentence breaks.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Virat Kohli is an Indian international cricketer and former captain of the India national team.
He is widely regarded as one of the greatest batsmen in the history of the sport.

Kohli holds numerous records, including the fastest batsman to 10,000 runs in One Day Internationals.
In 2023, he broke Sachin Tendulkar's record by scoring his 50th ODI century during the World Cup.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk.strip())
```

---

### B. `CharacterTextSplitter` (Simple Delimiter Splitter)
Splits text based on a single explicit separator (e.g., `"\n\n"`):

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=200,
    chunk_overlap=30
)

chunks = splitter.split_text(text)
```

---

### C. `MarkdownHeaderTextSplitter` (Structure-Aware)
Splits markdown documents based on header hierarchy (`#`, `##`, `###`), preserving the header structure in each chunk's metadata:

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = """
# Cricket Guide

## Batting
Batting requires hand-eye coordination, timing, and footwork.

## Bowling
Bowling involves pace, spin, and seam movement.
### Spin Bowling
Spin bowling relies on finger or wrist revolutions to deviate the ball off the pitch.
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_chunks = markdown_splitter.split_text(markdown_document)

for chunk in md_chunks:
    print("Content:", chunk.page_content)
    print("Metadata:", chunk.metadata)
```

---

### D. Splitting `Document` Objects with Metadata Preservation

When working with loaders (like `PyPDFLoader`), use `.split_documents()` so that file names and page numbers are copied to every child chunk:

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("sample.txt")
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_docs = text_splitter.split_documents(raw_docs)

print(f"Created {len(chunked_docs)} chunks from {len(raw_docs)} original document.")
# Each chunked_doc has page_content AND the original metadata!
```

---

## 4. Best Practices & Rule of Thumb

| Use Case | Recommended `chunk_size` | Recommended `chunk_overlap` |
| :--- | :--- | :--- |
| **Factual QA / Semantic Search** | 300 – 600 characters | 50 – 100 characters |
| **Comprehensive Summarization** | 1000 – 2000 characters | 150 – 300 characters |
| **Code / Programming Files** | Language-specific AST splitters (`PythonCodeTextSplitter`) | 50 – 100 characters |

> [!TIP]
> Always prefer `RecursiveCharacterTextSplitter` over `CharacterTextSplitter` because it minimizes fractured sentences and semantic distortion.
