from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
# from huggingface_hub import login

# login()

from dotenv import load_dotenv
import os

load_dotenv()
token = os.environ.get("HF_HUB_TOKEN")


# Initialize the search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the model
model = InferenceClientModel(token=token)

agent = CodeAgent(
    model=model,
    tools=[search_tool],
)

# Example usage
response = agent.run(
    "Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering."
)
print(response)
