from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode
from openai import OpenAI

from tools import tools

# Initialize OpenAI client
client = OpenAI()
model = "gpt-4o-mini"

# Define the state type
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next: Annotated[str, "The next node to route to"]

# Initialize tool node with built-in routing
tool_node = ToolNode(tools)

model_with_tools = ChatOpenAI(
    model=model,
).bind_tools(tools)

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# Build the graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()

def run_agent(question: str) -> str:
    """Run the agent with a given question"""
    # Run the agent
    for chunk in app.stream(
        {"messages": [("human", question)]}, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()