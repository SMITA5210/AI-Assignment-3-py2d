# Your agent goes here. The Agent1 class must hold your final submission.

from pyrusgeom.geom_2d import Sector2D, Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


class Agent1(StarterAgent):
    def __init__(self):
        super().__init__()

class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None

# STARTER CODE ENDS HERE #
