import os
from typing import List, TypedDict
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# 1. Define Agent State Schema
class AgentState(TypedDict):
    message: List[HumanMessage]
    response: str


# 2. Initialize Model
llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.7)


# 3. Define Node
def process_message(state: AgentState) -> dict:
    """Processes the incoming message and generates a response."""
    messages = state.get("message", [])
    
    # FIX: Use llm.invoke() instead of calling llm(...) directly
    ai_message = llm.invoke(messages)
    
    return {"response": ai_message.content}


# 4. Build Graph
graph = StateGraph(AgentState)
graph.add_node("process_message", process_message)
graph.add_edge(START, "process_message")
graph.add_edge("process_message", END)

agent = graph.compile()


# 5. Streamlit Interface
st.title("LangGraph Agent")

user_input = st.text_input("Wikipedia Research Task:")

# Execute graph only when the user submits input
if user_input:
    with st.spinner("Processing request..."):
        result = agent.invoke({"message": [HumanMessage(content=user_input)]})
        st.subheader("Response:")
        st.write(result.get("response"))