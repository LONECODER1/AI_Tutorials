# LangChain Component: Memory

By default, Large Language Models are **stateless**. Each API call to an LLM is completely isolated and independent; the model retains zero recollection of previous questions or interactions.

**Memory** in LangChain provides the mechanisms to persist state, manage conversational history, and inject context from prior dialogue turns into current prompts.

---

## 1. Why is Memory Needed?

Without memory, conversational flow breaks down:
- **User Turn 1**: *"Hi, my name is Alex."*
- **Assistant**: *"Hello Alex, nice to meet you!"*
- **User Turn 2**: *"What is my name?"*
- **Assistant (Stateless)**: *"I'm sorry, I don't know your name because I don't have access to past messages."*

With memory, the system records each message exchange and automatically re-injects past context into the prompt sent to the LLM.

---

## 2. Core Memory Strategies

LangChain categorizes memory into several common patterns depending on token budgets and conversational longevity:

```
+-----------------------------------------------------------------------+
|                         Memory Strategies                             |
+-----------------------------------------------------------------------+
|  1. Full Transcript (ConversationBufferMemory)                        |
|     Stores all past turns verbatim.                                   |
+-----------------------------------------------------------------------+
|  2. Sliding Window (ConversationBufferWindowMemory)                   |
|     Stores only the last K interactions (drops oldest turns).         |
+-----------------------------------------------------------------------+
|  3. Summarization (ConversationSummaryMemory)                         |
|     Uses an LLM to progressively summarize past history into a paragraph|
+-----------------------------------------------------------------------+
|  4. Hybrid (ConversationSummaryBufferMemory)                          |
|     Buffers recent turns verbatim + summarizes earlier turns.         |
+-----------------------------------------------------------------------+
|  5. Modern RunnableWithMessageHistory / LangGraph                     |
|     Decoupled session store (Redis, SQLite, Postgres, in-memory).     |
+-----------------------------------------------------------------------+
```

---

## 3. Explanations and Code Examples

### A. `ConversationBufferMemory` (Full History)
* **How it works**: Preserves the complete transcript of the conversation verbatim.
* **Pros**: Simple, zero loss of fidelity.
* **Cons**: Consumes token limits rapidly; becomes expensive and eventually exceeds the context window on long chats.

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)

# Save context from user and AI
memory.save_context({"input": "Hi, I am planning a trip to Tokyo."}, 
                    {"output": "Tokyo is wonderful! When are you planning to go?"})
memory.save_context({"input": "Next month in April."}, 
                    {"output": "April is great for cherry blossoms."})

# Load current memory state
history = memory.load_memory_variables({})
print(history["history"])
```

---

### B. `ConversationBufferWindowMemory` (Sliding Window)
* **How it works**: Keeps only the most recent $K$ interactions and discards older interactions.
* **Pros**: Prevents token overflow and bounds API costs.
* **Cons**: Forgets facts mentioned early in the conversation once they slide out of the window.

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only the last 2 interactions (k=2)
window_memory = ConversationBufferWindowMemory(k=2, return_messages=True)

window_memory.save_context({"input": "I like blue."}, {"output": "Noted, blue is a great color."})
window_memory.save_context({"input": "I have 2 dogs."}, {"output": "Dogs are lovely pets."})
window_memory.save_context({"input": "I live in New York."}, {"output": "New York is a bustling city."})

# Interaction 1 ("I like blue") has now been evicted!
print(window_memory.load_memory_variables({})["history"])
```

---

### C. `ConversationSummaryMemory` (Summarizer-Based)
* **How it works**: Runs an auxiliary LLM call in the background to condense dialogue turns into an evolving summary paragraph.
* **Pros**: Keeps memory footprint minimal while retaining core context from long dialogues.
* **Cons**: Incurs extra LLM token usage and latency for every summarization step.

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

summary_memory = ConversationSummaryMemory(llm=llm, return_messages=True)

summary_memory.save_context(
    {"input": "Hi, I'm building an e-commerce platform using Django and PostgreSQL."},
    {"output": "That's a solid stack! Are you handling authentication yourself or using Auth0?"}
)
summary_memory.save_context(
    {"input": "We are using Auth0 and deploying on AWS ECS."},
    {"output": "Got it. ECS Fargate is great for containerized Django apps."}
)

# Returns condensed summary
print(summary_memory.load_memory_variables({})["history"])
```

---

### D. Modern Standard: `RunnableWithMessageHistory` (Session-Based Persistence)

In modern LCEL architectures, memory is managed via session IDs backed by persistent storage (e.g., Redis, SQLite, Mongo, or memory dictionary):

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. Global storage mapping session_id -> chat history
session_store = {}

def get_session_history(session_id: str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# 2. Prompt with MessagesPlaceholder for dynamic history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly personal assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

model = ChatOpenAI(model="gpt-4o-mini")
chain = prompt | model | StrOutputParser()

# 3. Wrap chain with session history manager
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 4. Interact with specific session
config = {"configurable": {"session_id": "user_101"}}

res1 = conversational_chain.invoke({"input": "Hello! My favorite food is Biryani."}, config=config)
print("Bot:", res1)

res2 = conversational_chain.invoke({"input": "What is my favorite food?"}, config=config)
print("Bot:", res2)
# Output: "Your favorite food is Biryani!"
```

---

## 4. Summary & Best Practices

| Memory Type | Token Usage | Context Retention | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Buffer** | High (grows linearly) | 100% exact fidelity | Short Q&A sessions, quick support tickets |
| **Buffer Window** | Capped & predictable | Retains only last $K$ turns | Moderate chats where only recent context matters |
| **Summary** | Flat / bounded | Semantic essence retained | Long multi-session customer relationships |
| **Message History** | Configurable backend | Persistent across server restarts | Production web applications, multi-user apps |
