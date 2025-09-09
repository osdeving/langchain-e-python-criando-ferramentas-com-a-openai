from llm import get_groq_llm, get_openai_llm, get_ollama_llm
from langchain_core.prompts import PromptTemplate

numero_dias = 7
numero_criancas = 2
atividade = "praia"

prompt_template = PromptTemplate(
    template="""
    Crie um roteiro de viagem de {dias} dias, 
    para uma família com {numero_criancas} crianças,
    que gostam de {atividade}
    """
)

prompt = prompt_template.format(
    dias=numero_dias,
    numero_criancas=numero_criancas,
    atividade=atividade
)

print("Prompt : \n", prompt)

llm = get_groq_llm()

resposta = llm.invoke(prompt)

print(resposta.content)





