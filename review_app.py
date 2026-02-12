# review_app.py

import streamlit as st
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM
# -------------------------------
# Pydantic Model (VERY IMPORTANT)
# -------------------------------
class ReviewAnalysis(BaseModel):
    sentiment: str = Field(description="Overall sentiment: Positive, Negative or Neutral")
    rating: int = Field(description="Rating out of 5")
    key_features: List[str] = Field(description="Important product features mentioned")
    improvement_suggestions: List[str] = Field(description="Suggestions for improvement")


# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Product Review Analyzer", page_icon="⭐", layout="wide")

st.title("⭐ AI Product Review Analyzer")
st.markdown("Extract structured insights using PydanticOutputParser")

# -------------------------------
# Initialize Model & Parser
# -------------------------------
llm = OllamaLLM(model="llama3")
parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

prompt = PromptTemplate(
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="""
You are an AI product review analyzer.

Analyze the review and extract structured insights.

IMPORTANT:
- Sentiment must be Positive, Negative, or Neutral.
- Rating must be an integer between 1 and 5.
- Return ONLY valid structured output.

{format_instructions}

Review:
{review}
"""
)

chain = prompt | llm | parser

# -------------------------------
# User Input
# -------------------------------
review_input = st.text_area("Enter Product Review:", height=250)

if st.button("Analyze Review"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing review..."):
            try:
                result = chain.invoke({"review": review_input})
                st.success("Analysis Successful!")

                st.subheader("📊 Structured Output")
                st.json(result.model_dump())

            except Exception as e:
                st.error("Validation Failed or Incorrect Output Format.")
                st.write(e)
