# LangChain Component: Chains & LangChain Expression Language (LCEL)

A **Chain** is an orchestrated sequence of calls—combining prompt templates, models, output parsers, retrievers, or custom Python functions into a unified, executable pipeline.

In modern LangChain (v0.1+ and onwards), chains are constructed using **LangChain Expression Language (LCEL)**, which uses the Unix-style pipe operator (`|`) to connect components adhering to the **Runnable** protocol.

---

## 1. Why LCEL?

- **Declarative & Readable**: Build complex workflows with intuitive `component_a | component_b` syntax.
- **Unified Interface**: Every LCEL runnable automatically supports:
  - `invoke()`: Synchronous single execution.
  - `stream()`: Stream tokens as they generate.
  - `batch()`: Parallel processing of multiple inputs.
  - `ainvoke()` / `astream()` / `abatch()`: Native asynchronous support.
- **Built-in Parallelism**: Run steps simultaneously using `RunnableParallel` without complex threading code.
- **Streaming & Tracing Out-of-the-Box**: Minimal latency with token streaming and instant integration with observability tools like LangSmith.

---

## 2. Core LCEL Primitives

### A. The Basic Pipe (`|`)
Connects the output of one component directly to the input of the next.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Basic LCEL Chain
chain = prompt | model | parser

# Execute
result = chain.invoke({"country": "Japan"})
print(result)  # "The capital of Japan is Tokyo."
```

---

### B. `RunnableParallel` (Executing Steps Concurrently)
Executes multiple operations in parallel and combines their results into a dictionary.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatOpenAI(model="gpt-4o-mini")

joke_chain = (
    ChatPromptTemplate.from_template("Tell a joke about {topic}")
    | model
    | StrOutputParser()
)

poem_chain = (
    ChatPromptTemplate.from_template("Write a 2-line poem about {topic}")
    | model
    | StrOutputParser()
)

# Run both chains in parallel
combined_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

output = combined_chain.invoke({"topic": "coffee"})
print("Joke:", output["joke"])
print("Poem:", output["poem"])
```

---

### C. `RunnablePassthrough` (Passing Original Inputs)
Frequently used in RAG to pass the original user question through to the prompt while simultaneously retrieving relevant documents.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Simulated retriever function
def fake_retriever(query):
    return "LangChain was created by Harrison Chase in late 2022."

prompt = ChatPromptTemplate.from_template("""
Answer the question based strictly on the context:
Context: {context}
Question: {question}
""")

model = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {"context": fake_retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print(rag_chain.invoke("Who created LangChain and when?"))
```

---

### D. `RunnableLambda` (Custom Python Logic)
Wraps any standard Python function or lambda so it can participate in an LCEL pipeline.

```python
from langchain_core.runnables import RunnableLambda

def reverse_string(text: str) -> str:
    return text[::-1]

def uppercase_string(text: str) -> str:
    return text.upper()

chain = RunnableLambda(reverse_string) | RunnableLambda(uppercase_string)

print(chain.invoke("langchain"))  # Output: "NIAHCGNAL"
```

---

## 3. Real-World RAG Chain Architecture

Here is how modern LCEL builds a complete Retrieval-Augmented Generation (RAG) system:

```
[ User Input (query) ]
         |
         +----------------------------------+
         |                                  |
         v                                  v
[ Vector Store Retriever ]        [ RunnablePassthrough ]
         |                                  |
         v                                  v
     { context }                      { question }
         \                                  /
          +----------------+---------------+
                           |
                           v
                 [ ChatPromptTemplate ]
                           |
                           v
                   [ ChatOpenAI (LLM) ]
                           |
                           v
                  [ StrOutputParser ]
                           |
                           v
                    [ Final Answer ]
```

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Knowledge Base
documents = [
    "Virat Kohli has scored 50 ODI centuries, the most by any player in history.",
    "Rohit Sharma holds the record for highest individual ODI score of 264 runs.",
    "Jasprit Bumrah is known for his unusual bowling action and lethal yorkers."
]

vectorstore = FAISS.from_texts(documents, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 2. Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context provided:
Context: {context}

Question: {question}
""")

# 3. Complete LCEL Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# 4. Invoke
response = rag_chain.invoke("Who has scored the most ODI centuries?")
print(response)
```

---

## 4. Key Takeaways

1. **Legacy Chains (Deprecated)**: Old classes like `LLMChain`, `SimpleSequentialChain`, or `RetrievalQA` have been superseded by LCEL pipelines.
2. **Streaming & Async by Default**: Every LCEL chain automatically works with `.stream()` and `asyncio` without extra configuration.
3. **Modularity**: Individual runnables can be tested, reused, or mocked independently in unit tests.
