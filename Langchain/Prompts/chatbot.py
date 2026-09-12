from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Specify the model argument
model = ChatGroq(model="openai/gpt-oss-120b")
history=[]
while True:
    user_input = input("You: ") 
    if user_input.lower() == "quit":
        break
    history.append({"role":"user","content":user_input})
    result = model.invoke(history)
    history.append({"role":"assistant","content":result.content})
    print("AI: ", result.content)
