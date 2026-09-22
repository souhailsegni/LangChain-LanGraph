from typing import TypedDict
from langgraph.graph import StateGraph , END

class AgentState(TypedDict):
    
    name: str
    
def ask_name(state: AgentState) -> AgentState:
    
    """
    Asks the user for their name and returns it.
    
    Args:
        state (AgentState): The agent's state.
    """
    name = input("What is your name? ")
    state["name"] = name
    return state
def greet(state: AgentState) -> AgentState:
    
    """
    Greets the user by name.
    
    Args:
        state (AgentState): The agent's state.
    """
    print(f"Hello, {state['name']}!")
    
    return state
graph = StateGraph(AgentState)

graph.add_node("ask_name", ask_name)
graph.add_node("greet", greet)
graph.set_entry_point("ask_name")
graph.add_edge("ask_name", "greet")
graph.add_edge("greet", END)

compiled_graph = graph.compile()
print(compiled_graph.get_graph().draw_ascii())
compiled_graph.invoke({})