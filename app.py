import os

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Article Generator", page_icon="✍️")
st.title("Article Generator")

if not api_key:
    st.error("Missing GOOGLE_API_KEY. Create a .env file with your Google API key and restart the app.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
prompt = st.text_area("Enter your topic:", height=120)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7)

if st.button("Generate article", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a topic first.")
    else:
        article_prompt = f"""
        Write a clear, engaging, and well-structured article about: {prompt}

        Requirements:
        - Use clear headings and paragraphs
        - Keep the writing polished and easy to read
        - Include a strong introduction and conclusion
        - Avoid bullet points unless needed
        - Do not mention that you are an AI
        """

        try:
            with st.spinner("Generating your article..."):
                response = llm.invoke(article_prompt)
                content = getattr(response, "content", str(response))

                if isinstance(content, list):
                    content = "\n".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )

            st.subheader("Your article")
            st.markdown(content)
        except Exception as e:
            st.error(f"Something went wrong while generating the article: {e}")