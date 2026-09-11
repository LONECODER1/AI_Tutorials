from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_completion_tokens=100
)

response = chat_model.invoke("What is the capital of India?")
print(response.content)