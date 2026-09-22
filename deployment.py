import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables (.env should contain HUGGINGFACEHUB_API_TOKEN=hf_xxx)
load_dotenv()

st.title("Fee Management Q&A")

df = pd.read_excel("Fee Management.xlsx")
st.dataframe(df.head())

# 1. Use a model supported on HF Serverless API
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",  # Supported serverless model
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1,
)

model = ChatHuggingFace(llm=llm)

question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if question:
        with st.spinner("Searching dataset..."):
            prompt = f"""You are a helpful assistant. Answer the user's question ONLY from the dataset below. If the answer is not available in the dataset, reply exactly: Sorry, I don't have an answer about this.

Dataset:
{df.to_string(index=False)}

Question: {question}"""

            response = model.invoke(prompt)
            st.write(response.content)
    else:
        st.warning("Please enter a question.")
