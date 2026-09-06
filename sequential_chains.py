import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain, SequentialChain

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Article Generator", page_icon="✍️")
st.title("Article Generator")

if not api_key:
    st.error("Missing GOOGLE_API_KEY. Create a .env file with your Google API key and restart the app.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

topic = st.text_input("Enter your topic:")
language = st.selectbox("Select language:", ["English", "Spanish", "French", "German", "Chinese"])

# Define prompt templates
title_template = PromptTemplate(
    input_variables=["topic", "language"],
    template="Give me a catchy title about {topic} in {language}.",
)
body_template = PromptTemplate(
    input_variables=["title"],
    template="Write a detailed article based on this title: {title}",
)

# Initialize models
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7)
llm2 = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.9)

# Set up individual chains with explicit output keys
title_chain = LLMChain(llm=llm, prompt=title_template, output_key="title", verbose=True)
body_chain = LLMChain(llm=llm2, prompt=body_template, output_key="body", verbose=True)

# Combine using SequentialChain to handle multiple inputs and outputs
overall_chain = SequentialChain(
    chains=[title_chain, body_chain],
    input_variables=["topic", "language"],
    output_variables=["title", "body"],
    verbose=True,
)

if topic:
    with st.spinner("Generating article..."):
        response = overall_chain({"topic": topic, "language": language})
        st.subheader("Generated Article")
        st.markdown(f"### **Title:** {response['title']}")
        st.markdown(f"**Body:**\n{response['body']}")