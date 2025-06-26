from build123d import *
from bd_mixins.gltf_tools import appearance_collection, export_wrapper


bottle = import_step("demo_bottle.step")
bottle.appearance = appearance_collection["chrome"]
export_wrapper(bottle, "demo_bottle.gltf")