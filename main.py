# app.py
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM
import json

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Resume Extractor", page_icon="📄", layout="wide")

st.title("📄 AI Resume Information Extractor")
st.markdown("Extract structured information from resume text using LangChain + JsonOutputParser")

# -------------------------------
# Initialize Model
# -------------------------------
llm = OllamaLLM(model="llama3")
parser = JsonOutputParser()

prompt = PromptTemplate(
    input_variables=["resume_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="""
You are an AI resume information extractor.

Extract the following fields:
- name
- email
- skills (as a list)
- experience_years (integer only)
- education (as a list)

IMPORTANT:
Return ONLY valid JSON. Do not add explanations.

{format_instructions}

Resume Text:
{resume_text}
"""
)

chain = prompt | llm | parser

# -------------------------------
# User Input Section
# -------------------------------
resume_input = st.text_area("Paste Resume Text Here:", height=300)

if st.button("Extract Information"):
    if resume_input.strip() == "":
        st.warning("Please enter resume text.")
    else:
        with st.spinner("Analyzing resume..."):
            try:
                result = chain.invoke({"resume_text": resume_input})
                st.success("Extraction Successful!")

                st.subheader("📊 Extracted Data")
                st.json(result)

            except Exception as e:
                st.error("Failed to parse JSON output.")
                st.write(e)


