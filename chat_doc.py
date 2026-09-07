import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Document Chat", page_icon="💬")
st.title("Document Chat")

@st.cache_resource
def get_vector_store():
    loader = PyPDFLoader("./LangChain.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    # Back to Google embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", 
        google_api_key=api_key
    )
    
    return Chroma.from_documents(splits, embeddings)

vector_store = get_vector_store()
st.write("Vector store loaded successfully!")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

question = st.text_input("Ask a question about the document:")

if question:
    with st.spinner("Generating answer..."):
        # Initialize history if it doesn't exist
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # 1. Format the history into a list of tuples for LangChain
        formatted_history = [(msg["question"], msg["answer"]) for msg in st.session_state.history]
        
        # 2. Pass the formatted history to the chain
        answer = chain.run({"question": question, "chat_history": formatted_history})
        
        # 3. Save the new interaction to Streamlit's state
        st.session_state.history.append({"question": question, "answer": answer})
        
        # Display current answer
        st.subheader("Answer")
        st.write(answer)
        
        # Display full conversation history
        st.divider()
        st.write("### Conversation History")
        for msg in st.session_state.history:
            st.write(f"**You:** {msg['question']}")
            st.write(f"**Assistant:** {msg['answer']}")