from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from config import OLLAMA_MODEL

def run_llm(question, template):
    """Runs a question through the Ollama model."""
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=OLLAMA_MODEL)
    chain = prompt | model
    response = chain.invoke({"question": question})
    return response


if __name__ == "__main__":
    print("Testing Ollama...")
    print(run_llm("What is MoE?", "Answer this question"))
