import os
from typing import Annotated, TypedDict
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


# 1. State Schema with 'add_messages' Reducer
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 2. Updated Free Tier Models (Fixed Deprecated 2.5 Architecture)
primary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",  # Upgraded to current standard fast model
    temperature=0.7,
    max_retries=5
)

fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",       # Upgraded to current standard reasoning model
    temperature=0.7,
    max_retries=2
)

# Combine into fallback chain
llm = primary_llm.with_fallbacks([fallback_llm])


# 3. Node Function
def call_model(state: AgentState) -> dict:
    """Passes conversation history to Gemini and cleans up complex JSON outputs."""
    response = llm.invoke(state["messages"])
    
    # Check if the output content is wrapped inside a list of dictionaries
    if isinstance(response.content, list):
        extracted_text = ""
        for block in response.content:
            if isinstance(block, dict) and block.get("type") == "text":
                extracted_text += block.get("text", "")
            elif isinstance(block, str):
                extracted_text += block
        
        # Override the content property with clean, markdown-friendly string text
        response.content = extracted_text.strip()
        
    return {"messages": [response]}



# 4. Build Graph
builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

agent = builder.compile()


# 5. Streamlit Chat Interface
st.set_page_config(page_title="LangGraph Chatbot", page_icon="💬")
st.title("💬 LangGraph Chatbot (Free Tier)")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Accept input
if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.write(prompt)

    new_user_message = HumanMessage(content=prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                input_messages = st.session_state.messages + [new_user_message]
                result = agent.invoke({"messages": input_messages})
                
                st.session_state.messages = result["messages"]
                st.write(result["messages"][-1].content)
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    st.warning("Free rate limit reached. Please wait ~1 minute and try again.")
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    st.warning("Google free servers are temporarily busy. Retrying in a few seconds...")
                else:
                    st.error(f"Error: {e}")