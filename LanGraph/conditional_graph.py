from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# 1. Define the Graph State schema
class State(TypedDict):
    input_text: str
    category: str
    result: str


# 2. Define Node Functions
def classify_input(state: State) -> dict:
    """Classifies the user input to set state['category']."""
    text = state["input_text"].lower()
    
    if any(op in text for op in ["+", "-", "*", "/", "sum", "add"]):
        category = "math"
    elif any(word in text for word in ["hello", "hi", "hey"]):
        category = "greeting"
    else:
        category = "general"
        
    print(f"  └─ [classify_input] Detected category: '{category}'")
    return {"category": category}


def handle_math(state: State) -> dict:
    """Handler node for math requests."""
    print("  └─ [handle_math] Executing math node...")
    return {"result": "Routed to Math Handler."}


def handle_greeting(state: State) -> dict:
    """Handler node for greetings."""
    print("  └─ [handle_greeting] Executing greeting node...")
    return {"result": "Routed to Greeting Handler."}


def handle_general(state: State) -> dict:
    """Fallback handler node for general queries."""
    print("  └─ [handle_general] Executing general node...")
    return {"result": "Routed to General Handler."}


# 3. Define the Router Function (Returns the target node name)
def route_by_category(state: State) -> Literal["handle_math", "handle_greeting", "handle_general"]:
    """Determines which node to execute next based on the state."""
    category = state.get("category")
    
    if category == "math":
        return "handle_math"
    elif category == "greeting":
        return "handle_greeting"
    else:
        return "handle_general"


# 4. Construct the Graph
builder = StateGraph(State)

# Add Nodes
builder.add_node("classify_input", classify_input)
builder.add_node("handle_math", handle_math)
builder.add_node("handle_greeting", handle_greeting)
builder.add_node("handle_general", handle_general)

# Define standard edge from START to classification
builder.add_edge(START, "classify_input")

# Add CONDITIONAL Edge:
# Calls `route_by_category` after `classify_input` to decide destination node
builder.add_conditional_edges(
    source="classify_input",
    path=route_by_category,
    path_map={
        "handle_math": "handle_math",
        "handle_greeting": "handle_greeting",
        "handle_general": "handle_general",
    }
)

# Connect handler nodes to END
builder.add_edge("handle_math", END)
builder.add_edge("handle_greeting", END)
builder.add_edge("handle_general", END)

# Compile graph
graph = builder.compile()


# 5. Execute with test inputs
if __name__ == "__main__":
    test_queries = [
        "Hello there!",
        "Can you add 20 + 30?",
        "Tell me something about Linux.",
    ]

    for query in test_queries:
        print(f"\n---> Input: '{query}'")
        output = graph.invoke({"input_text": query})
        print(f"Final State Output: {output['result']}")