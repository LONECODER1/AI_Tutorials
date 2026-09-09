# LangChain Components

Welcome to the **LangChain Components** module. This directory provides dedicated guides and runnable code examples for every core building block of the LangChain ecosystem.

---

## Complete Overview Guide

For a single consolidated document covering every component, architecture diagrams, and an end-to-end RAG script:
👉 **[Read All Components Master Guide (all_components.md)](./all_components.md)**

---

## Component Index & Dedicated Guides

| # | Component | Focus Areas | Guide Link |
| :-: | :--- | :--- | :--- |
| **01** | **Models** | Chat Models vs. LLMs, `SystemMessage`, streaming, batching, multi-provider support | [models.md](./models.md) |
| **02** | **Prompts** | `PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`, `FewShotPromptTemplate` | [prompts.md](./prompts.md) |
| **03** | **Output Parsers** | `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`, `.with_structured_output()` | [output_parsers.md](./output_parsers.md) |
| **04** | **Chains & LCEL** | Pipe `\|` syntax, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, RAG chain | [chains_lcel.md](./chains_lcel.md) |
| **05** | **Memory** | `ConversationBufferMemory`, `ConversationBufferWindowMemory`, `ConversationSummaryMemory`, sessions | [memory.md](./memory.md) |
| **06** | **Document Loaders** | `PyPDFLoader`, `TextLoader`, `WebBaseLoader`, `CSVLoader`, `DirectoryLoader`, metadata | [document_loaders.md](./document_loaders.md) |
| **07** | **Text Splitters** | Chunking strategies, `chunk_size`, `chunk_overlap`, `RecursiveCharacterTextSplitter` | [text_splitters.md](./text_splitters.md) |
| **08** | **Embeddings** | `OpenAIEmbeddings`, `HuggingFaceEmbeddings`, cosine similarity, `CacheBackedEmbeddings` | [embeddings.md](./embeddings.md) |
| **09** | **Vector Stores & Retrievers** | `FAISS`, `Chroma`, `Pinecone`, `as_retriever()`, MMR search, similarity thresholds | [vector_stores_retrievers.md](./vector_stores_retrievers.md) |
| **10** | **Tools** | `@tool` decorator, `StructuredTool`, schema validation, `.bind_tools()`, `ToolMessage` | [tools.md](./tools.md) |
| **11** | **Agents** | ReAct pattern, `create_tool_calling_agent`, `AgentExecutor`, loops, guardrails | [agents.md](./agents.md) |
| **12** | **Callbacks** | Real-time streaming, token usage tracking, cost calculation, custom handlers, LangSmith | [callbacks.md](./callbacks.md) |

---

## Recommended Learning Path

1. **Foundations**: Start with **[Models](./models.md)** and **[Prompts](./prompts.md)**.
2. **Pipelines**: Connect them with **[Output Parsers](./output_parsers.md)** and **[Chains & LCEL](./chains_lcel.md)**.
3. **Information Retrieval (RAG)**: Study **[Document Loaders](./document_loaders.md)** $\to$ **[Text Splitters](./text_splitters.md)** $\to$ **[Embeddings](./embeddings.md)** $\to$ **[Vector Stores & Retrievers](./vector_stores_retrievers.md)**.
4. **State & Conversation**: Learn conversational retention with **[Memory](./memory.md)**.
5. **Autonomy & Agents**: Implement external actions with **[Tools](./tools.md)** and autonomous loops with **[Agents](./agents.md)**.
6. **Production & Observability**: Add telemetry with **[Callbacks](./callbacks.md)**.
