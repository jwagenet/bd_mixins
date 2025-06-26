# GLTF Tools

Extension of build123s `export_gltf` to apply material apperances to obejcts and add materials to GLTF export. Currently rewrites material properties of exported GLTF rather than using OCCT GLTF export builder. Does not yet support textures.

## Setup

Install bd_mixins from github

````
pip install git+https://github.com/jwagenet/bd_mixins
````

## Usage

Import at minimum `export_wrapper` and `MaterialAppearance` from gltf_tools:

````py
from build123d import *
from bd_mixins.gltf_tools import appearance_collection, export_wrapper, MaterialAppearance
````

Add `appearance` property and optionally `color` to object(s) to export either as MaterialAppearance or from appearance_collection. To reduce headaches relocating parts, its a good idea to convert objects to `Solid` before applying transformations.

````py
box = Box(1, 1, 1).solid()
box.appearance = MaterialAppearance(
        name="box color",
        base_color=(.5, .5, .5, 1),
        roughness=.23,
        use_part_color=True
    ),
box.color = Color(0x13ff95)

sphere = Pos((1.5, 0, 0)) * Sphere(.5).solid()
sphere.appearance = appearance_collection["aluminum"]
````

Export a single object or multiple objects as children of a Compound:

````py
export_wrapper(Compound(children=[box, sphere]), "box_and_sphere.gltf")
````

## Documentation

* `MaterialAppearance` object to represent appearance shader properties. Supports base GLTF material properties and official material extensions
* `apperance_collection` library of predefined material appearances accessible as a dict
* `export_wrapper` replaces `export_gltf` for exporting objects with appearance property

## Notes

* `export_wrapper` rebuilds the GLTF node tree to remove empty parent nodes and rename nodes to match object labels. This *shouldn't* cause issues.