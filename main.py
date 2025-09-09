from llm import get_groq_llm, get_openai_llm, get_ollama_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = get_groq_llm()

prompt = PromptTemplate(
    template="""
    Crie um roteiro de viagem de {dias} dias, 
    para uma família com {numero_criancas} crianças,
    que gostam de {atividade}
    """,
    input_variables=["dias", "numero_criancas", "atividade"]
)

cadeia =  prompt | llm | StrOutputParser()

print(cadeia.invoke({
    "dias": 7, 
    "numero_criancas": 2,
    "atividade": "praia"
}))







