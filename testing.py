import inspect
from smolagents import InferenceClientModel
print(inspect.signature(InferenceClientModel))

model = InferenceClientModel(provider="google", model="gemini-1.5", api_key=None)