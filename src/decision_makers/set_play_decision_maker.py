from typing import TYPE_CHECKING
from src.interfaces.IDecisionMaker import IDecisionMaker
from service_pb2 import *
from src.behaviors.bhv_setplay import BhvSetPlay
from src.behaviors.starter.bhv_starter_setplay import BhvStarterSetPlay
import os

if TYPE_CHECKING:
    from src.sample_player_agent import SamplePlayerAgent

class SetPlayDecisionMaker(IDecisionMaker):
    def __init__(self, agent: "SamplePlayerAgent"):
        self.mode = os.environ.get("SETPLAY", "helios")
        self.bhv_setplay = None
        if self.mode == "own":
            self.bhv_setplay = (BhvStarterSetPlay() if agent.use_starter_code
                        else BhvSetPlay())

    def make_decision(self, agent: "SamplePlayerAgent"):
        if self.mode == "helios":
            agent.add_action(PlayerAction(helios_set_play=HeliosSetPlay()))
        elif self.bhv_setplay is not None:
            self.bhv_setplay.execute(agent)