from langchain_groq import Groq
from dotenv import load_dotenv

load_dotenv()

llm =Groq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
)

response = llm.invoke("What is the capital of India?")
print(response)

