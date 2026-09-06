import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  # Use PyPDFLoader here
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Document Chat", page_icon="💬")
st.title("Document Chat")

# Load the PDF file (each page becomes a Document object)
loader = PyPDFLoader("./LangChain.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# Split the documents into smaller chunks
splits = text_splitter.split_documents(documents)

# Create embeddings for the document chunks
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", 
    google_api_key=api_key
)

# Create a vector store from the document chunks and their embeddings
vector_store = Chroma.from_documents(splits, embeddings)
st.write("Vector store created successfully!")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

question = st.text_input("Ask a question about the document:")

if question:
    with st.spinner("Generating answer..."):
        answer = chain.run(question)
        st.subheader("Answer")
        st.write(answer)
        
