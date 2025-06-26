from dataclasses import dataclass, asdict

@dataclass
class TextureData:
    source: str
    tex_coord: int | None = None
    scale: float | None = None


@dataclass
class MaterialAppearance:
    name: str | None = None

    # Color
    base_color: tuple[float, float, float, float] | None = None
    base_color_texture: str | TextureData | None = None

    # Metallic Roughness PBR
    metallic: float | None = 0.0
    roughness: float | None = 0.0
    metallic_roughness_texture: str | TextureData | None = None

    # Base Textures
    normal_texture: str | TextureData | None = None
    occlusion_texture: str | TextureData | None = None

    # Alpha
    alpha_mode: str | None = None
    alpha_cutoff: float | None = None

    # Emissive
    emissive: tuple[float, float, float] | None = None
    emissive_texture: str | TextureData | None = None
    emissive_strength: float | None = None

    # Transmission
    transmission: float | None = None
    transmission_texture: str | TextureData | None = None

    # Clearcoat
    clearcoat: float | None = None
    clearcoat_texture: str | TextureData | None = None
    clearcoat_roughness: float | None = None
    clearcoat_roughness_texture: str | TextureData | None = None
    clearcoat_normal_texture: str | TextureData | None = None

    # Volume
    thickness: float | None = None
    thickness_texture: str | TextureData | None = None
    attenuation_distance: float | None = None
    attenuation_color: tuple[float, float, float] | None = None

    # Index of Refraction
    ior: float | None = None

    # Specular
    specular: float | None = None
    specular_color: tuple[float, float, float] | None = None
    specular_texture: str | TextureData | None = None
    specular_color_texture: str | TextureData | None = None

    # Sheen
    sheen_color: tuple[float, float, float] | None = None
    sheen_color_texture: str | TextureData | None = None
    sheen_roughness: float | None = None
    sheen_roughness_texture: str | TextureData | None = None

    # Anisotropy
    anisotropy_strength: float | None = None
    anisotropy_rotation: float | None = None
    anisotropy_texture: str | TextureData | None = None

    # Misc
    unlit: bool = False
    double_sided: bool = False
    use_part_color: bool = False

    def __repr__(self):
        props = []
        for key, value in asdict(self).items():
            if key == "name":
                name = value
            elif value not in [None, False]:
                props.append(f"{key}={value}")
        prop_string = ", ".join(props)
        return f"MaterialAppearance [{name}] ({prop_string})"


class AppearanceCollection:
    def __init__(self, appearances):
        self._items = appearances
        self._lookup = {app.name: app for app in appearances}

    def __getitem__(self, name):
        return self._lookup[name]

    def add(self, appearance):
        self._items.append(appearance)
        self._lookup[appearance.name] = appearance

    def all(self):
        return list(self._items)


appearance_collection = AppearanceCollection([
    MaterialAppearance(
        name="default",
        base_color=(1, 1, 1.0, 1.0),
        metallic=0,
        roughness=1,
        use_part_color=True
    ),
    MaterialAppearance(
        name="aluminum",
        base_color=(0.8, 0.81, 0.82, 1),
        metallic=1.0,
        roughness=.27
    ),
    MaterialAppearance(
        name="aluminum - stamped",
        base_color=(0.94, 0.94, 0.94, 1.0),
        metallic=0.9,
        roughness=0.55
    ),
    MaterialAppearance(
        name="aluminum - extruded",
        base_color=(0.55, 0.55, 0.55, 1.0),
        metallic=1.0,
        roughness=0.15
    ),
    MaterialAppearance(
        name="brass",
        base_color=(0.85, 0.62, 0.49, 1.0),
        metallic=1,
        roughness=0.282,
        anisotropy_strength=1,
        anisotropy_rotation=.11,
        clearcoat=.25,
    ),
    MaterialAppearance(
        name="chrome",
        base_color=(1, 1, 1, 1.0),
        metallic=1.0,
        roughness=0.05,
        specular=1,
        specular_color=(1, 1, 1)
    ),
    MaterialAppearance(
        name="copper - polished",
        base_color=(0.73, 0.44, 0.33, 1.0),
        metallic=1.0,
        roughness=0.14
    ),
    MaterialAppearance(
        name="concrete",
        base_color=(0.33, 0.33, 0.33, 1.0),
        metallic=0.0,
        roughness=0.45
    ),
    MaterialAppearance(
        name="glass",
        base_color=(1, 1, 1, 1.0),
        metallic=0.0,
        roughness=0.05,
        transmission=1,
        attenuation_color=(1, 1, 1),
        attenuation_distance=0.1,
        thickness=1,
    ),
    MaterialAppearance(
        name="paint",
        base_color=(1, 0.0, 0.0, 1.0),
        metallic=0.0,
        roughness=0.4,
        anisotropy_strength=.5,
        clearcoat=.5,
        clearcoat_roughness=.25,
    ),
    MaterialAppearance(
        name="plastic - shiny",
        base_color=(0.1, 0.1, 1.0, 1.0),
        metallic=0.0,
        roughness=0.1,
        use_part_color=True
    ),
    MaterialAppearance(
        name="plastic - dull",
        base_color=(1.0, 0.1, 0.1, 1.0),
        metallic=0.0,
        roughness=0.8,
        use_part_color=True,
    ),
    MaterialAppearance(
        name="steel - dull",
        base_color=(0.31, 0.31, 0.31, 1),
        metallic=1.0,
        roughness=0.61
    ),
    MaterialAppearance(
        name="rubber",
        base_color=(0.05, 0.05, 0.05, 1.0),
        metallic=0.0,
        roughness=0.53
    ),
    MaterialAppearance(
        name="wood - pine",
        base_color=(0.99, 0.79, 0.62, 1),
        metallic=0.0,
        roughness=.6
    ),
])