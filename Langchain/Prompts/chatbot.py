from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Specify the model argument
model = ChatGroq(model="openai/gpt-oss-120b")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    result = model.invoke(user_input)
    print("AI: ", result.content)
