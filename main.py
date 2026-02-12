from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# Initialize model
llm = OllamaLLM(model="llama3")

# Prompt template
prompt = PromptTemplate(
    input_variables=["paragraph"],
    template="""
You are an AI text transformer.

Given the paragraph below:

{paragraph}

1. Provide a short summary (3-4 lines)
2. Detect the tone (Formal / Casual / Technical)
3. Provide an improved version

Return response as plain text only.
"""
)

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser

if __name__ == "__main__":
    user_input = input("Enter paragraph:\n")
    result = chain.invoke({"paragraph": user_input})
    print("\n===== OUTPUT =====\n")
    print(result)
