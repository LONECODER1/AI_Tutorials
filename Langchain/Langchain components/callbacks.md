# LangChain Component: Callbacks & Observability

In production AI applications, it is crucial to understand what is happening inside complex LLM chains and agents. You need to monitor latency, measure token consumption, stream tokens to the UI, log intermediate steps, and debug errors.

**Callbacks** provide a hook system allowing you to tap into various stages of your LLM application's lifecycle.

---

## 1. How Callbacks Work

LangChain fires event notifications at distinct milestones throughout execution:

```
[ User Request ]
       |
       +---> Event: on_chain_start
       |
       +---> Event: on_llm_start
       |        |
       |        +---> Event: on_llm_new_token (fired for each streamed token)
       |        |
       +---> Event: on_llm_end
       |
       +---> Event: on_tool_start
       |        |
       |        +---> Event: on_tool_end
       |
       +---> Event: on_chain_end
```

---

## 2. Built-in Callbacks

### A. Real-Time Token Streaming (`StreamingStdOutCallbackHandler`)
Streams tokens directly to stdout as they arrive from the model API:

```python
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Tokens will print progressively to terminal as they arrive
llm.invoke("Write a 3-line motivational quote for developers.")
```

---

### B. Tracking Token Usage and Costs (`get_openai_callback`)
Monitors exact token usage (prompt tokens, completion tokens) and financial cost for OpenAI API calls:

```python
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

with get_openai_callback() as cb:
    response1 = llm.invoke("Explain quantum computing in 20 words.")
    response2 = llm.invoke("Explain machine learning in 20 words.")
    
    print("\n--- Usage Telemetry ---")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost:.6f}")
```

---

## 3. Creating Custom Callbacks

You can build custom logging, analytics, or UI push notifications by subclassing `BaseCallbackHandler`:

```python
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict, List
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

class AuditLoggingHandler(BaseCallbackHandler):
    """Custom callback handler to log prompt queries and response lengths."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        print(f"[AUDIT LOG] LLM started with prompt: {prompts[0][:60]}...")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        generation_text = response.generations[0][0].text
        print(f"[AUDIT LOG] LLM finished. Generated length: {len(generation_text)} characters.")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        print(f"[AUDIT LOG] Tool invoked: {serialized.get('name')} with args: {input_str}")

# Attach to model or chain
audit_handler = AuditLoggingHandler()
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[audit_handler])

llm.invoke("What are the 3 laws of robotics?")
```

---

## 4. Passing Callbacks in LCEL Chains

Callbacks can be passed at two levels:

1. **Constructor Level**: Configured when instantiating the model or component (applies to every execution).
   ```python
   model = ChatOpenAI(callbacks=[my_handler])
   ```
2. **Invocation Level**: Passed dynamically during `.invoke()`, affecting only that specific run without modifying global state:
   ```python
   chain.invoke({"input": "Hello"}, config={"callbacks": [my_handler]})
   ```

---

## 5. Production Observability: LangSmith

While manual callbacks are great for debugging locally, production applications benefit from automated tracing with **LangSmith**:

```python
import os

# Set environment variables for automatic zero-code tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "cricket-assistant-prod"

# Now every LCEL chain, retriever call, and tool execution is automatically
# visualized with complete trace trees, latencies, and token costs in the LangSmith UI.
```
