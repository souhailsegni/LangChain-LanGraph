

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from sqlalchemy import values

class AgentState(TypedDict):
    name: str
    values: List[int]
    result: int
    
def ask_name(state: AgentState) -> AgentState:
    """
    Asks the user for their name and returns it.
    
    Args:
        state (AgentState): The agent's state.
    """
    name = input("What is your name? ")
    state["name"] = name
    return state
def ask_values(state: AgentState) -> AgentState:
    """
    Asks the user for a list of integers and returns it.
    
    Args:
        state (AgentState): The agent's state.
    """
    values = input("Enter a list of integers separated by spaces: ")
    # Inside ask_values() in multiple_inputs.py:

# Replace commas with spaces before splitting
    state["values"] = [int(x) for x in values.replace(',', ' ').split()]
    return state

def compute_sum(state: AgentState) -> AgentState:
    """
    Computes the sum of the list of integers and returns it.
    
    Args:
        state (AgentState): The agent's state.
    """
    state["result"] = sum(state["values"])
    print(f"The sum of the values is: {state['result']}")
    return state

graph = StateGraph(AgentState)

graph.add_node("ask_name", ask_name)
graph.add_node("ask_values", ask_values)
graph.add_node("compute_sum", compute_sum)
graph.set_entry_point("ask_name")
graph.add_edge("ask_name", "ask_values")
graph.add_edge("ask_values", "compute_sum")
graph.add_edge("compute_sum", END)

compiled_graph = graph.compile()
print(compiled_graph.get_graph().draw_ascii())
compiled_graph.invoke({})