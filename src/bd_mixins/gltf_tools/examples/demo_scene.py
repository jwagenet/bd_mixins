from build123d import *
from bd_mixins.gltf_tools import appearance_collection, export_wrapper


scene = import_step("demo_scene.step")

properties = [
    {
        "label": "sphere",
        "appearance": appearance_collection["copper - polished"]
    },
    {
        "label": "gear",
        "appearance": appearance_collection["brass"]
    },
    {
        "label": "bottle",
        "color": Color(0x3C4F16),
        "appearance": appearance_collection["plastic - shiny"]
    },
    {
        "label": "screw",
        "appearance": appearance_collection["steel - dull"]
    },
    {
        "label": "bracket",
        "appearance": appearance_collection["aluminum"]
    },
    {
        "label": "sleeve",
        "color": Color(0.99, 1, 0.7),
        "appearance": appearance_collection["plastic - dull"]
    },
    {
        "label": "light_cap",
        "color": Color(0x8B5742),
        "appearance": appearance_collection["glass"]
    },
]

scene.color = None
for i, child in enumerate(scene.children):
    prop = next((prop for prop in properties if prop["label"] == child.label), None)
    if "color" in prop:
        child.color = prop["color"]
    child.appearance = prop["appearance"]

export_wrapper(scene, "demo_scene.gltf")