from panda_gym.pybullet import PyBullet

from experiments.warehouse.scenes.basic import BasicSceneConstructor
from experiments.warehouse.scenes.fancy import FancySceneConstructor


def construct_scene(sim: PyBullet, scene_type: str) -> None:
    """Construct a scene."""
    if scene_type == "basic":
        return BasicSceneConstructor(sim).construct()
    elif scene_type == "fancy":
        return FancySceneConstructor(sim).construct()
    else:
        raise ValueError(f"Unknown scene type: {scene_type}")
