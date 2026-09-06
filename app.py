import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain

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

# Define prompt template
title_template = PromptTemplate(
    input_variables=["topic", "language"],
    template="Write a detailed article about {topic} in {language}.",
)

# Initialize model
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.9)

# Set up the LLMChain using the classic module
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

if topic:
    with st.spinner("Generating article..."):
        response = title_chain.predict(topic=topic, language=language)
        st.subheader("Generated Article")
        st.markdown(response)