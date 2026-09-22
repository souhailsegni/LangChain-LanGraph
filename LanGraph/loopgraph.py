from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# 1. Define the Graph State
class State(TypedDict):
    topic: str
    draft: str
    quality_score: int
    attempts: int


# 2. Define Node Functions
def generate_or_refine_draft(state: State) -> dict:
    """Generates a new draft or refines the existing one."""
    attempts = state.get("attempts", 0) + 1
    topic = state["topic"]
    
    # Simulate draft refinement on each iteration
    draft = f"Draft v{attempts} for '{topic}'"
    print(f"  ├─ [generate_draft] Attempt #{attempts}: Created '{draft}'")
    
    return {"draft": draft, "attempts": attempts}


def evaluate_draft(state: State) -> dict:
    """Evaluates the draft and produces a score."""
    attempts = state["attempts"]
    
    # Simulated scoring logic: improves by +35 points per attempt
    score = min(100, attempts * 35)
    print(f"  ├─ [evaluate_draft] Draft Quality Score: {score}/100")
    
    return {"quality_score": score}


def finalize_result(state: State) -> dict:
    """Final node executed once the loop condition passes."""
    print(f"  └─ [finalize_result] Success! Final approved output: '{state['draft']}'")
    return {}


# 3. Define the Router Function (The Loop / Exit Decision)
def should_continue(state: State) -> Literal["generate_or_refine_draft", "finalize_result"]:
    """Determines whether to loop back to refinement or proceed to finalize."""
    score = state.get("quality_score", 0)
    attempts = state.get("attempts", 0)
    
    # Loop Exit Condition: Score >= 80 OR Max Attempts Reached
    if score >= 80 or attempts >= 5:
        print("  └─ [Conditional Router] Quality threshold met! Exiting loop -> finalize_result")
        return "finalize_result"
    
    print("  └─ [Conditional Router] Quality too low! Looping back -> generate_or_refine_draft\n")
    return "generate_or_refine_draft"


# 4. Construct the Looping Graph
builder = StateGraph(State)

# Add Nodes
builder.add_node("generate_or_refine_draft", generate_or_refine_draft)
builder.add_node("evaluate_draft", evaluate_draft)
builder.add_node("finalize_result", finalize_result)

# Add Standard Edges
builder.add_edge(START, "generate_or_refine_draft")
builder.add_edge("generate_or_refine_draft", "evaluate_draft")

# Add CONDITIONAL Edge (Creates the Loop)
builder.add_conditional_edges(
    source="evaluate_draft",
    path=should_continue,
    path_map={
        "generate_or_refine_draft": "generate_or_refine_draft",  # Loops back to start
        "finalize_result": "finalize_result",                  # Exits loop
    }
)

# Connect final node to END
builder.add_edge("finalize_result", END)

# Compile Graph
graph = builder.compile()


# 5. Run the Script
if __name__ == "__main__":
    print("--- Executing Looping LangGraph ---")
    initial_input = {"topic": "Async Programming in Python"}
    graph.invoke(initial_input)
    print("\n--- Execution Complete ---")