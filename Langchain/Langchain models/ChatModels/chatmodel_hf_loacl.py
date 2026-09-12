from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm =HuggingFacePipeline.from_model_id(
    model_id="oChuanming/Tiny-Llama-2.2B-slerp",
    task="text-generation"
)

chat = ChatHuggingFace(llm=llm)
response = chat.invoke("What is the capital of India?")
print(response.content) 