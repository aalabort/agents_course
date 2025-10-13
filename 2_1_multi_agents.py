import math
from typing import Optional, Tuple
from smolagents import InferenceClientModel
import shapely
import geopandas
import plotly

from smolagents import tool
from smolagents.utils import encode_image_base64, make_image_url
from smolagents import OpenAIServerModel
from smolagents import CodeAgent
from smolagents import DuckDuckGoSearchTool
from smolagents import VisitWebpageTool

from PIL import Image


from dotenv import load_dotenv
import os

load_dotenv()
token = os.environ.get("HF_HUB_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

@tool
def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 750.0,  # Average speed for cargo planes
) -> float:
    """
    Calculate the travel time for a cargo plane between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 750 km/h for typical cargo planes)

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # Chicago (41.8781° N, 87.6298° W) to Sydney (33.8688° S, 151.2093° E)
        >>> result = calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093))
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # Add 10% to account for non-direct routes and air traffic controls
    actual_distance = distance * 1.1

    # Calculate flight time
    # Add 1 hour for takeoff and landing procedures
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    # Format the results
    return round(flight_time, 2)

# model = OpenAIServerModel(
#     model_id="gemini-2.0-flash",  # or whichever Gemini model you want
#     api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
#     api_key=GEMINI_API_KEY,
# )

model=OpenAIServerModel("gpt-4o", max_tokens=8096)

web_agent = CodeAgent(
    model=model,
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    additional_authorized_imports= ["requests"],
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10,
)





def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True


manager_agent = CodeAgent(
    # model=OpenAIServerModel(
    #     model_id="gemini-2.0-flash",  # or whichever Gemini model you want
    #     api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    #     api_key=GEMINI_API_KEY,
    # ),
    model=OpenAIServerModel("gpt-4o", max_tokens=8096),
    tools=[calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
        "requests",
        "plotly.graph_objects",
        "plotly.graph_objs.scattermapbox",
        "plotly.graph_objs.layout",
        "shapely.geometry",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)

manager_agent.visualize()

manager_agent.run("""
Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W).
Also give me some supercar factories with the same cargo plane transfer time. You need at least 6 points in total.
Represent this as spatial map of the world, with the locations represented as scatter points with a color that depends on the travel time, and save it to saved_map.png!

Here's an example of how to plot and return a map:
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio

# Sample dataset
df = pd.DataFrame({
    "centroid_lat": [40.75, 40.78, 40.73, 40.70],
    "centroid_lon": [-73.99, -73.95, -74.00, -73.97],
    "name": ["A", "B", "C", "D"],
    "peak_hour": [5, 10, 7, 12]
})

# Create Scattergeo figure
fig = go.Figure(go.Scattermapbox(
    lat=df["centroid_lat"],
    lon=df["centroid_lon"],
    mode="markers+text",
    marker=go.scattermapbox.Marker(
        size=15,
        color=df["peak_hour"],
        colorscale="Magma",
        colorbar=dict(title="Peak Hour")
    ),
    text=df["name"]
))

# Set the map layout
fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=go.layout.mapbox.Center(
            lat=df["centroid_lat"].mean(),
            lon=df["centroid_lon"].mean()
        ),
        zoom=10
    ),
    margin=dict(l=0, r=0, t=0, b=0)
)

# Show figure
fig.show()

# Save as PNG (requires kaleido)
fig.write_image("saved_image.png")

# Return to smolagents
final_answer(fig)



Never try to process strings using code: when you have a string to read, just print it and you'll see it.
""")