# LangChain Components: Complete Comprehensive Guide

LangChain is an open-source framework designed to simplify the development of applications powered by Large Language Models (LLMs). Its core philosophy is **modularity** and **composability**: breaking complex AI systems into reusable, standardized building blocks called **Components**.

---

## Architecture Overview: How LangChain Components Connect

```
                        +-----------------------------------------------------------+
                        |                       User Query                          |
                        +-----------------------------+-----------------------------+
                                                      |
                                                      v
                                      +---------------+---------------+
                                      |   Prompt / PromptTemplate     |
                                      +---------------+---------------+
                                                      |
                  +-----------------------------------+-----------------------------------+
                  |                                                                       |
                  v                                                                       v
+-----------------+-----------------+                                   +-----------------+-----------------+
|     RAG Pipeline Components       |                                   |    Autonomous Agent Components    |
|                                   |                                   |                                   |
| [ Document Loaders ] (PDF, Web)   |                                   | [ Tools ] (APIs, Python, SQL)     |
|          |                        |                                   |          ^                        |
|          v                        |                                   |          |                        |
| [ Text Splitters ] (Chunks)       |                                   | [ Tool Calling Loop (ReAct) ]     |
|          |                        |                                   |                                   |
|          v                        |                                   |                                   |
| [ Embeddings ] (Text -> Vectors)  |                                   |                                   |
|          |                        |                                   |                                   |
|          v                        |                                   |                                   |
| [ Vector Stores & Retrievers ]    |                                   |                                   |
+-----------------+-----------------+                                   +-----------------+-----------------+
                  |                                                                       |
                  +-----------------------------------+-----------------------------------+
                                                      |
                                                      v
                                      +---------------+---------------+
                                      |    Chat Models / LLMs         |
                                      |  (OpenAI, Claude, Gemini)     |
                                      +---------------+---------------+
                                                      |
                                                      +---------> [ Memory / State Management ]
                                                      |
                                                      v
                                      +---------------+---------------+
                                      |       Output Parsers          |
                                      |  (String, JSON, Pydantic)     |
                                      +---------------+---------------+
                                                      |
                                                      v
                                      +---------------+---------------+
                                      |         Final Output          |
                                      +-------------------------------+
                                                      |
                 [ Observability: Callbacks & Tracing monitor all stages ]
```

---

## Quick Component Matrix

| # | Component | Primary Purpose | Key Classes / Functions | Dedicated Guide |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Models** | Reasoning engine / text generation | `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI` | [models.md](./models.md) |
| **2** | **Prompts** | Parameterized input templating & roles | `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder` | [prompts.md](./prompts.md) |
| **3** | **Output Parsers** | Transforming LLM text into structured types | `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser` | [output_parsers.md](./output_parsers.md) |
| **4** | **Chains & LCEL** | Orchestrating steps via pipe `\|` syntax | `Runnable`, `RunnableParallel`, `RunnablePassthrough` | [chains_lcel.md](./chains_lcel.md) |
| **5** | **Memory** | Preserving conversation state across turns | `ConversationBufferMemory`, `RunnableWithMessageHistory` | [memory.md](./memory.md) |
| **6** | **Document Loaders** | Ingesting raw data from files/APIs into `Document`s | `PyPDFLoader`, `TextLoader`, `WebBaseLoader`, `CSVLoader` | [document_loaders.md](./document_loaders.md) |
| **7** | **Text Splitters** | Chunking large texts into semantic passages | `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter` | [text_splitters.md](./text_splitters.md) |
| **8** | **Embeddings** | Converting natural text into dense numerical vectors | `OpenAIEmbeddings`, `HuggingFaceEmbeddings` | [embeddings.md](./embeddings.md) |
| **9** | **Vector Stores & Retrievers** | Storing vectors & similarity search | `FAISS`, `Chroma`, `Pinecone`, `as_retriever()` | [vector_stores_retrievers.md](./vector_stores_retrievers.md) |
| **10** | **Tools** | Executable functions exposed to the LLM | `@tool`, `StructuredTool`, `.bind_tools()` | [tools.md](./tools.md) |
| **11** | **Agents** | LLM-directed autonomous decision-making loops | `create_tool_calling_agent`, `AgentExecutor`, `LangGraph` | [agents.md](./agents.md) |
| **12** | **Callbacks** | Logging, token tracking, streaming, and observability | `BaseCallbackHandler`, `StreamingStdOutCallbackHandler` | [callbacks.md](./callbacks.md) |

---

## 1. Models (LLMs & Chat Models)

### Explanation:
Models are the core computation engines. LangChain standardizes the interface across dozens of AI providers. Modern applications use **Chat Models**, which consume structured message lists (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`) and output an `AIMessage`.

### Example:
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
messages = [
    SystemMessage(content="You are a senior cricket historian."),
    HumanMessage(content="Name the player with the most international centuries.")
]
response = model.invoke(messages)
print(response.content)  # Sachin Tendulkar (100 centuries)
```

---

## 2. Prompts & Prompt Templates

### Explanation:
Prompt Templates separate the static instructions from dynamic user inputs. They validate inputs, handle role assignment, and support dynamic history insertion with `MessagesPlaceholder`.

### Example:
```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert software tutor in {language}."),
    ("human", "Explain {concept} in 2 sentences.")
])

prompt_value = template.invoke({"language": "Python", "concept": "Decorators"})
```

---

## 3. Output Parsers

### Explanation:
By default, models output unstructured strings. Output Parsers translate responses into strict Python types (e.g., `str`, `dict`, or Pydantic models). Modern models can also enforce schemas using `.with_structured_output()`.

### Example:
```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class Cricketer(BaseModel):
    name: str = Field(description="Name of player")
    role: str = Field(description="Batter, Bowler, or All-Rounder")

model = ChatOpenAI(model="gpt-4o-mini")
structured_llm = model.with_structured_output(Cricketer)

player = structured_llm.invoke("Jasprit Bumrah is India's premier fast bowler.")
print(player.name, "|", player.role)  # Jasprit Bumrah | Bowler
```

---

## 4. Chains & LCEL (LangChain Expression Language)

### Explanation:
Chains bind components together. LCEL uses the Unix pipe operator (`|`) to pass the output of one component into the input of the next. Every runnable natively supports `.invoke()`, `.stream()`, `.batch()`, and async calls.

### Example:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("Summarize {book} in 10 words.")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

print(chain.invoke({"book": "Atomic Habits"}))
```

---

## 5. Memory

### Explanation:
LLMs are stateless. Memory components store conversation turns and re-inject them into subsequent prompts. Options range from simple buffers (`ConversationBufferMemory`) to sliding windows (`ConversationBufferWindowMemory`) and persistent session stores (`RunnableWithMessageHistory`).

### Example:
```python
from langchain.memory import ConversationBufferWindowMemory

# Remember only the last 2 interactions
memory = ConversationBufferWindowMemory(k=2, return_messages=True)
memory.save_context({"input": "I like blue"}, {"output": "Noted."})
memory.save_context({"input": "I live in Delhi"}, {"output": "Delhi is great."})

print(memory.load_memory_variables({})["history"])
```

---

## 6. Document Loaders

### Explanation:
Document Loaders connect external data (PDFs, Web pages, CSVs, AWS S3, Notion) and normalize them into a standard `Document` object with `page_content` and `metadata`.

### Example:
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("cricket_rules.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages. Source: {pages[0].metadata['source']}")
```

---

## 7. Text Splitters (Chunking)

### Explanation:
Large documents must be divided into smaller chunks so that embedding models can capture granular semantic meaning without exceeding token windows. `RecursiveCharacterTextSplitter` splits hierarchically by paragraphs, sentences, and words while preserving a configurable overlap.

### Example:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_text("Long text content goes here...")
```

---

## 8. Embeddings

### Explanation:
Embeddings map human language into high-dimensional numerical vectors. Similar concepts produce vectors that are geometrically close in vector space (measured via Cosine Similarity).

### Example:
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("How many runs has Virat scored?")
print("Vector Dimensions:", len(vector))  # 1536
```

---

## 9. Vector Stores & Retrievers

### Explanation:
A Vector Store (e.g., FAISS, Chroma, Pinecone) indexes vectors for ultra-fast nearest-neighbor searches. Calling `.as_retriever()` turns a vector store into an LCEL Runnable that fetches relevant documents given a query string.

### Example:
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

texts = [
    "Virat Kohli scored 76 runs in the T20 World Cup final.",
    "Jasprit Bumrah was player of the tournament."
]

vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

matched = retriever.invoke("Who was the top bowler in the tournament?")
print(matched[0].page_content)
```

---

## 10. Tools

### Explanation:
Tools give LLMs the ability to execute code, query live databases, or hit external web APIs. They are declared with type signatures and docstrings so the LLM understands when and how to call them.

### Example:
```python
from langchain_core.tools import tool

@tool
def calculate_run_rate(runs: int, overs: float) -> float:
    """Calculate the cricket run rate given total runs and overs bowled."""
    return round(runs / overs, 2)

print(calculate_run_rate.invoke({"runs": 180, "overs": 20}))  # 9.0
```

---

## 11. Agents

### Explanation:
An Agent uses an LLM as a reasoning engine to dynamically decide which tools to execute, inspect the tool's output, and iterate until the user's objective is completed.

### Example:
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

@tool
def check_weather(city: str) -> str:
    """Check current weather for a city."""
    return f"The weather in {city} is 28°C and Sunny."

tools = [check_weather]
model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with tools."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

res = executor.invoke({"input": "What's the weather in Mumbai?"})
print(res["output"])
```

---

## 12. Callbacks & Observability

### Explanation:
Callbacks hook into execution milestones (`on_llm_start`, `on_llm_new_token`, `on_tool_end`) to enable real-time token streaming, latency tracking, cost calculation, and logging.

### Example:
```python
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

with get_openai_callback() as cb:
    res = llm.invoke("What is LangChain?")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost:.6f}")
```

---

## End-to-End Synergy: Complete Working Example

Below is a complete, production-ready script showing how **Loaders**, **Splitters**, **Embeddings**, **Vector Stores**, **Retrievers**, **Prompts**, **Models**, **LCEL**, and **Parsers** work together as one cohesive system:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Ingestion (Raw Text -> Chunks)
raw_text = """
The Indian cricket team won the ICC Men's T20 World Cup 2024 in Barbados.
Virat Kohli was awarded Player of the Match in the final for his 76 runs off 59 balls.
Jasprit Bumrah was named Player of the Tournament for taking 15 wickets at an economy rate of 4.17.
Following the victory, both Rohit Sharma and Virat Kohli announced retirement from T20 internationals.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_text(raw_text)

# 2. Vector Store & Retriever (Chunks -> Vectors -> Index)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a cricket expert. Answer the question using ONLY the provided context:
Context: {context}

Question: {question}
""")

# 4. Chain (LCEL Pipeline)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 5. Execute
question = "Why was Jasprit Bumrah named Player of the Tournament?"
answer = rag_chain.invoke(question)

print("Q:", question)
print("A:", answer)
```
