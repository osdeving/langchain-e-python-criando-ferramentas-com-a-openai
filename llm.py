from dotenv import load_dotenv
import os

from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


load_dotenv()

groq_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = os.getenv("GROQ_API_KEY")


#set_llm_cache(InMemoryCache())
set_llm_cache(SQLiteCache())


def get_openai_llm(model: str = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=openai_api_key,
    )

def get_groq_llm(model: str = "llama-3.3-70b-versatile", temperature: float = 0) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=groq_api_key,
    )

def get_ollama_llm(model: str = "deepseek-r1:7b", temperature: float = 0, base_url: str | None = "http://localhost:11434") -> ChatOllama:
    kwargs = {"model": model, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url  # ex: "http://localhost:11434"
    return ChatOllama(**kwargs)