from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

llm = OllamaLLM(model="mistral:7b-instruct", temperature=0.2)

prompt = PromptTemplate.from_template(
    """You are an expert summarizer. Read the following document and return a short, clear summary of its main theme.

Document:
{text}

Theme Summary:"""
)

summarizer = prompt | llm

def get_theme_summary(text):
    return summarizer.invoke({"text": text}).strip()