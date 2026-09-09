# LangChain Component: Prompts & Prompt Templates

Prompts are the instructions, context, and queries provided to language models to guide their generation. Hardcoding raw prompt strings creates brittle, repetitive code. LangChain provides **Prompt Templates** to parameterize, reuse, validate, and compose prompt logic cleanly.

---

## 1. Why Use Prompt Templates?

- **Dynamic Inputs**: Insert variables (user questions, fetched documents, user IDs) into consistent prompt structures.
- **Role Separation**: Structure inputs into distinct system, user, and assistant turns for Chat Models.
- **Few-Shot Learning**: Provide exemplar input-output pairs to dramatically improve model accuracy without fine-tuning.
- **Composability**: Combine and partially format prompts inside LangChain Expression Language (LCEL) chains.

---

## 2. Core Prompt Template Types

### A. `PromptTemplate` (For String-based Prompts)
Used for single-string prompts where variables are interpolated using Python format syntax `{variable}`.

```python
from langchain_core.prompts import PromptTemplate

# Create a prompt template with input variables
template = PromptTemplate(
    template="Explain the concept of {concept} in simple terms for a {audience}.",
    input_variables=["concept", "audience"]
)

# Format the template into a concrete string
formatted_prompt = template.format(concept="Recursion", audience="10-year-old child")
print(formatted_prompt)
# Output: "Explain the concept of Recursion in simple terms for a 10-year-old child."
```

---

### B. `ChatPromptTemplate` (For Chat Models - Recommended)
Constructs structured conversations by pairing system instructions, user queries, and assistant messages.

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI language tutor specializing in {language}. Correct errors politely."),
    ("human", "Translate this sentence: '{sentence}'"),
])

# Format into message objects
formatted_messages = chat_template.format_messages(
    language="Spanish",
    sentence="Where is the train station?"
)

for msg in formatted_messages:
    print(f"[{msg.type.upper()}]: {msg.content}")
```

---

### C. `MessagesPlaceholder` (For Dynamic Chat History)
When building chatbots, the number of previous conversation turns is dynamic. `MessagesPlaceholder` reserves a slot in the prompt template for an arbitrary list of previous messages.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant for CloudDesk."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# Simulate past conversation
history = [
    HumanMessage(content="My order number is #98765."),
    AIMessage(content="Thank you! I found order #98765. How can I assist you with it?")
]

prompt_value = chat_template.invoke({
    "chat_history": history,
    "user_input": "When will it be delivered?"
})

print(prompt_value.to_messages())
```

---

### D. `FewShotPromptTemplate` (Few-Shot In-Context Learning)
Provides real-world examples to the model before presenting the actual query.

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# 1. Define few-shot examples
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"}
]

# 2. Define how each example is formatted
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template="Word: {word}\nAntonym: {antonym}"
)

# 3. Assemble the FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of the provided word:\n",
    suffix="Word: {input_word}\nAntonym:",
    input_variables=["input_word"]
)

print(few_shot_prompt.format(input_word="brilliant"))
```

---

## 3. Partial Formatting

You can bind some variables early (e.g., system configuration, timestamps, API settings) while leaving user-facing variables open for runtime:

```python
from langchain_core.prompts import PromptTemplate
from datetime import datetime

prompt = PromptTemplate.from_template(
    "Current date: {current_date}\nUser Query: {query}"
)

# Pre-bind current_date
partial_prompt = prompt.partial(current_date=datetime.now().strftime("%Y-%m-%d"))

# At runtime, only pass query
final_prompt = partial_prompt.format(query="What holidays are coming up this month?")
print(final_prompt)
```

---

## 4. Best Practices

1. **Explicit Roles**: Always use `ChatPromptTemplate` with distinct `system` and `human` roles to maintain strict prompt boundaries and prevent prompt injection attacks.
2. **Template Validation**: Use `validate_template=True` to catch missing or misspelled input variables early during initialization.
3. **Piping with LCEL**: In modern LangChain, prompt templates connect directly to models via the pipe `|` operator:
   ```python
   chain = prompt_template | model
   ```
