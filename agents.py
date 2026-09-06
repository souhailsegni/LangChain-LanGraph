import os
from dotenv import load_dotenv

import wikipedia
wikipedia.set_user_agent("LangChainAgentApp/1.0 (john@example.com)")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents.initialize import initialize_agent
from langchain_classic.agents import AgentType
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.7)

tools = load_tools(["wikipedia"], llm=llm)

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True,
    handle_parsing_errors=True
)

prompt = input("Wikipedia Research Task: ")

agent.run(prompt)