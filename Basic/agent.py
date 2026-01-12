from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def country_info(country: str) -> str:
    """Get basic info about a country."""
    if country.lower() == "bangladesh":
        return "Bangladesh is a South Asian country with a population of over 170 million."
    return "Country not found."

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

llm_with_tools = llm.bind_tools([country_info])

def agent_node(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response]
    }

def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    if tool_name == "country_info":
        result = country_info.invoke(tool_args)
        return {
            "messages": [
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            ]
        }

def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
graph.add_edge("tools", "agent")

app = graph.compile()

result = app.invoke({"messages": [HumanMessage(content="Tell me about Bangladesh")]})
print(result["messages"][-1].content)
