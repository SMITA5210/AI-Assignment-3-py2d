from typing import TYPE_CHECKING
from src.interfaces.IDecisionMaker import IDecisionMaker
from .play_on_decision_maker import PlayOnDecisionMaker
from .set_play_decision_maker import SetPlayDecisionMaker
from .penalty_decision_maker import PenaltyDecisionMaker
from .goalie_decision_maker import GoalieDecisionMaker
from service_pb2 import *

if TYPE_CHECKING:
    from src.sample_player_agent import SamplePlayerAgent


class DecisionMaker(IDecisionMaker):
    """Routes each cycle to the handler for the current game mode.

    The goalie branch falls through when its handler returns False.
    GoalieDecisionMaker claims PlayOn and our
    own goalie-catch restart and declines the rest, so the keeper reaches
    the set-play handler and takes goal kicks as an ordinary player.
    """

    def __init__(self, agent: "SamplePlayerAgent"):
        self.play_on_decision_maker = PlayOnDecisionMaker(agent)
        self.set_play_decision_maker = SetPlayDecisionMaker(agent)
        self.penalty_decision_maker = PenaltyDecisionMaker(agent)
        self.goalie_decision_maker = GoalieDecisionMaker(agent)

    def make_decision(self, agent: "SamplePlayerAgent"):
        if agent.wm.self.is_goalie or agent.wm.self.uniform_number == 1:
            if self.goalie_decision_maker.make_decision(agent):
                return
        if agent.wm.game_mode_type == GameModeType.PlayOn:
            self.play_on_decision_maker.make_decision(agent)
        elif agent.wm.is_penalty_kick_mode:
            self.penalty_decision_maker.make_decision(agent)
        else:
            self.set_play_decision_maker.make_decision(agent)