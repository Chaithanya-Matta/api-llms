from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
# from typing import TYPE_CHECKING
from pydantic import BaseModel

load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true" # Landsmith tracking
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="API Server"
)

class TopicInput(BaseModel):
    topic: str

add_routes(
    app,
    ChatOpenAI(model="gpt-4o-mini"),
    path="/openAI"
)

llm_openai = ChatOpenAI(model="gpt-4o-mini")
llm_ollama=ChatOllama(model="llama3.2")

pt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words") # OpenAI
pt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words") # Ollama

chain1 = pt1 | llm_openai
chain2 = pt2 | llm_ollama

add_routes(
    app,
    chain1,
    path="/essay"
)

add_routes(
    app,
    chain2,
    path="/poem"
)

if __name__=="__main__":
    uvicorn.run(app, host="localhost",port=8000)