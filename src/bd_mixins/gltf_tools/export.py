import copy
from dataclasses import asdict

from pygltflib import GLTF2, Material

from build123d import Compound, Color, export_gltf

from appearance import MaterialAppearance, TextureData


class ConvertMaterial():
    def __init__(self, gltf: GLTF2):
        self.texture_sources = gltf.images

    def convert_group(self, props: list[str], names: list[str]) -> dict:
        group = {}
        for name in names:
            target = name.replace("Factor", "")
            target = "".join("_" + p.lower() if p.isupper() else p for p in target)
            if props[target] is not None:
                if isinstance(props[target], tuple):
                    group[name] = list(props[target])
                elif "texture" in target:
                    group[name] = self.add_texture(target, props[target])
                else:
                    group[name] = props[target]

        return group

    def add_texture(self, property: str, texture: str | TextureData) -> dict:
        def coerce_scale(property: str, scale: float) -> dict:
            if scale is None:
                return {}
            elif "normal" in property:
                return {"scale": scale}
            elif "occlusion" in property:
                return {"strength": scale}
            else:
                return {}

        info = {}
        if isinstance(texture, TextureData):
            info.update(coerce_scale(property, texture.scale))
            if texture["tex_coord"] is not None:
                info.update({"texCoord": texture.tex_coord})

            source = texture.source
        else:
            source = texture

        if source not in self.texture_sources:
            self.texture_sources.append(source)

        info.update({"index": self.texture_sources.index(source)})

        return info

    def add_material(self, appearance: MaterialAppearance):
        properties = asdict(appearance)
        material = {}

        material_names = ["name", "alphaMode", "alphaCutoff", "doubleSided", "normalTexture", "occlusionTexture", "emissiveFactor", "emissiveTexture"]
        material = self.convert_group(properties, material_names)

        pbr_names = ["baseColorFactor", "baseColorTexture", "metallicFactor", "roughnessFactor", "metallicRoughnessTexture"]
        if pbr := self.convert_group(properties, pbr_names):
            material["pbrMetallicRoughness"] = pbr

        extensions = {}

        transmission_names = ["transmissionFactor", "transmissionTexture"]
        if transmission := self.convert_group(properties, transmission_names):
            extensions["KHR_materials_transmission"] = transmission

        clearcoat_names = [
            "clearcoatFactor", "clearcoatTexture",
            "clearcoatRoughnessFactor", "clearcoatRoughnessTexture",
            "clearcoatNormalTexture"
        ]
        if clearcoat := self.convert_group(properties, clearcoat_names):
            extensions["KHR_materials_clearcoat"] = clearcoat

        volume_names = [
            "thicknessFactor", "thicknessTexture",
            "attenuationDistance", "attenuationColor"
        ]
        if volume := self.convert_group(properties, volume_names):
            extensions["KHR_materials_volume"] = volume

        sheen_names = [
            "sheenColorFactor", "sheenColorTexture",
            "sheenRoughnessFactor", "sheenRoughnessTexture"
        ]
        if sheen := self.convert_group(properties, sheen_names):
            extensions["KHR_materials_sheen"] = sheen

        specular_names = [
            "specularFactor", "specularTexture",
            "specularColorFactor", "specularColorTexture"
        ]
        if specular := self.convert_group(properties, specular_names):
            extensions["KHR_materials_specular"] = specular

        anisotropy_names = [
            "anisotropyStrength", "anisotropyRotation",
            "anisotropyTexture"
        ]
        if anisotropy := self.convert_group(properties, anisotropy_names):
            extensions["KHR_materials_anisotropy"] = anisotropy

        if ior := self.convert_group(properties, ["ior"]):
            extensions["KHR_materials_ior"] = ior

        if emissive_strength := self.convert_group(properties, ["emissiveStrength"]):
            extensions["KHR_materials_emissive_strength"] = emissive_strength

        if extensions:
            material["extensions"] = extensions

        return Material().from_dict(material)


def export_wrapper(objects, filename):
    if type(objects) is not Compound:
        objects = Compound(children=objects)

    appearances = []
    for object in objects.children:
        # Check if object has MaterialAppearance and apply default if not
        if "appearance" not in object.__dict__.keys() or not isinstance(object.appearance, MaterialAppearance):
            object.appearance = MaterialAppearance(
                name="default",
                base_color=(1, 1, 1.0, 1.0),
                metallic=0,
                roughness=1,
                use_part_color=True
            )

        # Replace appearance color with object color if allowed
        if object.appearance.use_part_color and object.color is not None:
            object.appearance = copy.deepcopy(object.appearance)
            object.appearance.base_color = tuple(object.color)

        if object.appearance not in appearances:
            appearances.append(object.appearance)

        # Give object a color unique to MaterialAppearance if it doesn't have one
        # we are abusing export_gltf assigning a material for each object color in the
        # initial export to make material substitution easier
        if object.color is None:
            value = appearances.index(object.appearance) / 1e6
            object.color = Color(value, value, value)

    export_gltf(objects, filename)
    exit()
    gltf = GLTF2.load(filename)

    # Replace initial gltf material properties from color with AppearanceMaterial
    for object in objects.children:
        idx = appearances.index(object.appearance)

        gltf_mats = [mat.name for mat in gltf.materials]
        if object.appearance.name not in gltf_mats:
            gltf.materials[idx] = ConvertMaterial(gltf).add_material(object.appearance)

    extensions_used = set()
    for mat in gltf.materials:
        if mat.extensions:
            extensions_used.update(mat.extensions.keys())

    gltf.extensionsUsed = sorted(extensions_used)
    gltf.save(filename)