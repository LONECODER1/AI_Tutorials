#using api 
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Initialize the underlying Hugging Face LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

# ChatHuggingFace wraps the LLM to provide the ChatModel interface
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is the capital of India?")
print(response.content)
