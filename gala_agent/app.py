from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool
from dotenv import load_dotenv
import os

from langchain.agents import initialize_agent, AgentType


load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))




# Initialize the web search tool
search_tool = DuckDuckGoSearchRun()

# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=os.environ.get("HF_HUB_TOKEN"),
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [search_tool, weather_info_tool, hub_stats_tool]
agent_executor = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state):
    prompt = "\n".join([m.content for m in state["messages"]])
    response = agent_executor.run(
        prompt
    )  # agent will call search_tool/weather_tool itself
    return {"messages": [AIMessage(content=response)]}


## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

prompt = "Who is Facebook and what's their most popular model?"
prompt = "One of our guests is from Qwen. What can you tell me about their most popular model?"
prompt = 'I need to speak with \'Dr. Nikola Tesla\' about recent advancements in wireless energy. Can you help me prepare for this conversation?'


response = alfred.invoke({"messages": [{"role": "user", "content": prompt}]})

print("🎩 Alfred's Response:")
print(response["messages"][-1].content)