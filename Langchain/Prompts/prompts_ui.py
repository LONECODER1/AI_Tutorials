import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq

# Automatically locate and load .env from the project root or current directory
load_dotenv(find_dotenv())

st.header("Research UI")
user_input = st.text_input("Enter your query")

if st.button("Submit"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            model = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0,
                api_key=os.getenv("GROQ_API_KEY")
            )
            result = model.invoke(user_input)
            st.write(result.content)
    else:
        st.warning("Please enter a query first!")