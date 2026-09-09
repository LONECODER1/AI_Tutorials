# LangChain Component: Agents

While a **Chain** follows a predetermined, hardcoded sequence of steps (e.g., Step A $\to$ Step B $\to$ Step C), an **Agent** uses a Large Language Model as a **reasoning engine** to determine which actions to take, in what order, and when the task is complete.

---

## 1. Chains vs. Agents

```
CHAIN (Deterministic Workflow):
[ User Query ] ---> [ Retrieve Chunks ] ---> [ LLM Summary ] ---> [ Output ]
(The path never changes)

AGENT (Dynamic Decision-Making Loop):
[ User Query ]
     |
     v
[ LLM Think: "What tool do I need?" ]
     |
     +---> [ Call Web Search ] ---> (Reads result)
     |
     +---> [ Call Calculator ] ---> (Computes math)
     |
     v
[ LLM Think: "I have all info needed" ] ---> [ Output Final Answer ]
```

---

## 2. The ReAct Pattern (Reason + Act)

Most agents follow the **ReAct** (Reasoning + Acting) paradigm:
1. **Thought**: The LLM reasons about the user request and what information is missing.
2. **Action**: The LLM chooses a tool and specifies input arguments.
3. **Observation**: The system executes the tool and returns the result to the LLM.
4. **Repeat**: Steps 1–3 repeat iteratively until the LLM decides it has sufficient information to formulate the final answer.

---

## 3. Building a Modern Tool-Calling Agent

In modern LangChain, agents are powered by native tool calling using `create_tool_calling_agent` and executed with `AgentExecutor`:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 1. Define Tools for the Agent
@tool
def get_live_stock_price(symbol: str) -> float:
    """Fetch the latest live trading stock price for a ticker symbol."""
    mock_prices = {"AAPL": 224.50, "GOOGL": 178.20, "MSFT": 448.10}
    return mock_prices.get(symbol.upper(), 0.0)

@tool
def calculate_portfolio_value(shares: int, price: float) -> float:
    """Calculate the total financial value given number of shares and share price."""
    return shares * price

tools = [get_live_stock_price, calculate_portfolio_value]

# 2. Select Model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Create Agent Prompt (Must include agent_scratchpad for intermediate steps)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an autonomous financial assistant. Use available tools to answer."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Construct Agent & Executor
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,      # Displays agent reasoning and tool executions in console
    max_iterations=5   # Safety limit to prevent infinite loops
)

# 5. Run the Agent
response = agent_executor.invoke({
    "input": "How much is my 15 shares of Apple (AAPL) worth right now?"
})

print("\nResult:")
print(response["output"])
```

### What happens behind the scenes during execution:
1. **Agent receives query**: *"How much is my 15 shares of Apple worth?"*
2. **LLM decides**: Needs the price of AAPL $\to$ calls `get_live_stock_price(symbol="AAPL")`.
3. **Tool returns**: `224.50`.
4. **LLM decides**: Needs to multiply 15 by 224.50 $\to$ calls `calculate_portfolio_value(shares=15, price=224.50)`.
5. **Tool returns**: `3367.50`.
6. **LLM decides**: Task complete $\to$ generates natural language answer: *"Your 15 shares of Apple (AAPL) are currently worth $3,367.50."*

---

## 4. Agent Safety & Guardrails

When building agents with access to real tools (sending emails, executing code, modifying databases), always implement safety guards:

- **`max_iterations`**: Caps the number of tool-execution cycles to avoid runaway infinite loops and billing spikes.
- **`max_execution_time`**: Timeout threshold (in seconds).
- **Human-in-the-Loop (HITL)**: Require human confirmation before the agent triggers destructive actions (e.g., executing SQL `DROP TABLE` or making financial transactions).
- **LangGraph**: For complex, stateful, multi-agent workflows with branching, cyclical graphs, and checkpoints, LangChain recommends **LangGraph**.

---

## 5. Summary

| Feature | Chains | Agents |
| :--- | :--- | :--- |
| **Execution Flow** | Pre-determined, rigid sequence | Dynamic, self-directed loop |
| **Tool Selection** | Hand-coded by developer | Decided dynamically by LLM |
| **Best For** | Predictable pipelines (RAG, summarization) | Open-ended problems, assistants, multi-step research |
