import os
import re
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def clear_history():
    if 'history' in st.session_state:
        st.session_state.history = []
        st.success("Chat history cleared!")

st.set_page_config(page_title="Document Chat", page_icon="💬")
st.title("Document Chat")

# --- 1. File Upload Section ---
upload_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Keep track of the current active file in session state
if "file_path" not in st.session_state:
    st.session_state.file_path = None

if upload_file:
    if st.button("Process File", on_click=clear_history):
        # Save the uploaded file locally
        file_name = os.path.join("./", upload_file.name)
        with open(file_name, "wb") as f:
            f.write(upload_file.getbuffer()) # getbuffer() is safer for Streamlit
        
        # Save the path to session state so the app knows a file is ready
        st.session_state.file_path = file_name
        st.success(f"File '{upload_file.name}' processed successfully!")

# --- 2. Vector Store (Only runs if a file is uploaded) ---
@st.cache_resource(show_spinner=False)
def get_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", 
        google_api_key=api_key
    )
    
    return Chroma.from_documents(splits, embeddings)

# --- 3. Chat Interface ---
if st.session_state.file_path:
    with st.spinner("Loading document into memory..."):
        # Pass the dynamic file path to the cached function
        vector_store = get_vector_store(st.session_state.file_path)
    
    st.write("Vector store loaded successfully!")

    # Fixed model name to a valid Gemini version
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=1)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

    # Initialize history if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Sidebar controls: Show/Hide history and Clear history
    st.sidebar.title("History")
    st.sidebar.checkbox("Show History", value=st.session_state.get('show_history', False), key="show_history")
    st.sidebar.button("Clear History", on_click=clear_history)
    # Session request counter and cooldown display
    if 'requests_made' not in st.session_state:
        st.session_state['requests_made'] = 0
    cooldown_until = st.session_state.get('cooldown_until', 0)
    now = time.time()
    if cooldown_until and cooldown_until > now:
        remaining = int(cooldown_until - now)
        st.sidebar.warning(f"Rate limit hit — please wait {remaining}s before asking again")
    st.sidebar.markdown(f"**Requests this session:** {st.session_state['requests_made']}")

    # Use a form so other widget changes (like toggling history) don't retrigger generation
    with st.form(key="ask_form"):
        q_input = st.text_input("Ask a question about the document:", key="question_input")
        submit = st.form_submit_button("Ask")

    if submit:
        # Prevent submission during cooldown
        cooldown_until = st.session_state.get('cooldown_until', 0)
        now = time.time()
        if cooldown_until and cooldown_until > now:
            remaining = int(cooldown_until - now)
            st.warning(f"Please wait {remaining}s due to rate limiting before trying again.")
        else:
            question = q_input.strip()
            if question:
                with st.spinner("Generating answer..."):
                    # Format history and get answer
                    formatted_history = [(msg["question"], msg["answer"]) for msg in st.session_state.history]
                    try:
                        answer = chain.run({"question": question, "chat_history": formatted_history})
                    except Exception as e:
                        # Try to parse retry delay from the exception message
                        msg = str(e)
                        delay = None
                        m = re.search(r"Please retry in (\d+(?:\.\d+)?)s", msg)
                        if not m:
                            m = re.search(r'retryDelay["\']?\s*:\s*["\']?(\d+(?:\.\d+)?)s', msg)
                        if m:
                            try:
                                delay = float(m.group(1))
                            except Exception:
                                delay = None
                        if delay:
                            st.error(f"Rate limit reached. Please retry in {int(delay)} seconds.")
                            st.session_state['cooldown_until'] = time.time() + delay
                        else:
                            st.error(f"Error generating answer: {e}")
                        answer = None

                    if answer is not None:
                        st.write(f"**Answer:** {answer}")
                        # Save to history
                        st.session_state.history.append({"question": question, "answer": answer})
                        # increment session counter
                        st.session_state['requests_made'] = st.session_state.get('requests_made', 0) + 1

    # Render chat history in the right sidebar when toggled on
    if st.session_state.get('show_history', False):
        st.sidebar.markdown("### Chat History")
        if not st.session_state.history:
            st.sidebar.info("No chat history yet.")
        else:
            for idx, msg in enumerate(reversed(st.session_state.history), 1):
                st.sidebar.markdown(f"**{idx}. Q:** {msg['question']}")
                st.sidebar.markdown(f"**A:** {msg['answer']}")
                st.sidebar.write("---")

else:
    st.info("Please upload and process a PDF to start chatting.")