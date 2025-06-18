from build123d import *
from ocp_vscode import *

from bd_mixins.gltf_tools.appearance import appearance_collection
from bd_mixins.gltf_tools.export import export_wrapper



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
    with Locations((0, 0, 1)):
        Sphere(.5)

part = part.part

cols = 4
objects = []
for i in range(len(properties)):
    x = i % cols * 1.5
    y = i // cols * 1.5
    object = part.moved(Location((x, y, 0)))
    object.color = properties[i]["color"]
    object.appearance = appearance_collection[properties[i]["appearance"]]
    objects.append(object)

group = Compound(children=objects)
show(group)
print(group)
export_wrapper(group, "demo_scene.gltf")

exit()
spur.label = "gear"
spur.appearance = appearance_collection["brass"]

bottle.label = "bottle"
bottle.color = Color(0x3C4F16)
bottle.appearance = appearance_collection["plastic - shiny"]

screw.label = "screw"
screw.appearance = appearance_collection["steel - dull"]

bracket.label = "bracket"
bracket.appearance = appearance_collection["aluminum"]

sleeve.label = "sleeve"
sleeve.color = Color(.99, 1, .7)
sleeve.appearance = appearance_collection["plastic - dull"]

light_cap.label = "light_cap"
light_cap.color = Color(0x8B5742)
light_cap.appearance = appearance_collection["glass"]

show(sphere, spur, screw, bracket, bottle, light_cap, sleeve)

group = [sphere, spur, screw, bracket, bottle, light_cap, sleeve]
com = Compound(children=group)

print("bottle", bottle.color)

export_step(bottle, "demo_bottle.step")
export_step(com, "demo_scene.step")

b2 = import_step("demo_bottle.step")
print(b2.color, b2.label)

for each in com.children:
    print(each.label)


from bd_mixins import gltf_tools

gltf_tools.export.export_wrapper(com, "cccc.gltf")
bottle.appearance = appearance_collection["chrome"]
gltf_tools.export.export_wrapper(bottle, "bottle.gltf")

show(bottle, scene)