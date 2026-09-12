from langchain_groq import GroqEmbeddings

from dotenv import load_dotenv


load_dotenv()

emb = GroqEmbeddings(
    model=''
)