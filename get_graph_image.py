from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

image_bytes = app.get_graph().draw_mermaid_png()

filename = "app_graph.png"

with open(filename, "wb") as f:
    f.write(image_bytes)

print(f"Graph successfully saved as '{filename}'")