from enum import Enum
import time


class Light(Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


class TrafficLightScorer:
    def __init__(self, target_idx: int, hold_seconds: float = 1.0,
                 amber_min: float = 0.50, green_min: float = 0.80):
        self._target = target_idx
        self._hold = hold_seconds
        self._amber_min = amber_min
        self._green_min = green_min
        self._green_since: float | None = None

    def evaluate(self, pred_idx: int, conf: float) -> tuple[Light, bool]:
        """Returns (light, completed)."""
        now = time.monotonic()
        if pred_idx != self._target or conf < self._amber_min:
            self._green_since = None
            return Light.RED, False
        if conf < self._green_min:
            self._green_since = None
            return Light.AMBER, False
        # Green territory
        if self._green_since is None:
            self._green_since = now
        completed = (now - self._green_since) >= self._hold
        return Light.GREEN, completed

    def render_light(self, light: Light) -> str:
        colours = {
            Light.RED: "#E74C3C",
            Light.AMBER: "#F39C12",
            Light.GREEN: "#27AE60",
        }
        colour = colours[light]
        return (f'<div style="width:60px;height:60px;'
                f'border-radius:50%;background:{colour};'
                f'margin:auto;box-shadow:0 0 12px {colour};"></div>')
