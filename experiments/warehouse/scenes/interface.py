from abc import ABC, abstractmethod

import numpy as np


class SceneManager(ABC):
    """Abstract class for scene managers."""

    @property
    @abstractmethod
    def frame_delay(self) -> float:
        """Get the frame delay for animations in seconds."""

    @abstractmethod
    def construct(self) -> None:
        """Construct the scene."""

    @abstractmethod
    def update_ee_identifier(self, ee_pos: np.ndarray) -> None:
        """Update the end-effector identifier."""

    @abstractmethod
    def animate_red_block(self) -> None:
        """Animations to run after a red block has been packed."""

    @abstractmethod
    def animate_green_block(self) -> None:
        """Animations to run after a green block has been packed."""

    @abstractmethod
    def animate_blue_block(self) -> None:
        """Animations to run after a blue block has been packed."""

    @abstractmethod
    def translate_tray(self, destination: str) -> None:
        """Translate the tray to the destination."""
