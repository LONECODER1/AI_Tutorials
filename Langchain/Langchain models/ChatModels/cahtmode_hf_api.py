# pyrefly: ignore [missing-import]
from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatHuggingFace(
    repo_id="oChuanming/Tiny-Llama-2.2B-slerp",
   task="conversational"
)

response = chat_model.invoke("What is the capital of India?")
print(response.content)
