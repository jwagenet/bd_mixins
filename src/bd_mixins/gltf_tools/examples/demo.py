import copy
from build123d import *
from bd_mixins.gltf_tools import appearance_collection, export_wrapper


properties = [{
    "color": None,
    "appearance": "copper - polished"
},
{
    "color": None,
    "appearance": "brass"
},
{
    "color": Color(0x3C4F16),
    "appearance": "plastic - shiny"
},
{
    "color": None,
    "appearance": "steel - dull"
},
{
    "color": None,
    "appearance": "aluminum"
},
{
    "color": Color(.99, 1, .7),
    "appearance": "plastic - dull"
},
{
    "color": Color(0x8B5742),
    "appearance": "glass"
},
{
    "color": None,
    "appearance": "wood - pine"
}]

with BuildPart() as part:
    Box(1, 1, 1)
    with Locations((0, 0, .9)):
        Sphere(.5)

part = part.part

cols = 4
objects = []
for i in range(len(properties)):
    x = i % cols * 1.5
    y = i // cols * 1.5
    object = copy.copy(part).translate((x, y, 0))
    object.color = properties[i]["color"]
    object.appearance = appearance_collection[properties[i]["appearance"]]
    object.label = object.appearance.name
    objects.append(object)

group = Compound(children=objects)
export_wrapper(group, "demo.gltf")