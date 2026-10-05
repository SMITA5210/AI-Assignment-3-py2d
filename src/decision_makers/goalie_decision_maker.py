import sys
from typing import TYPE_CHECKING

from src.interfaces.IDecisionMaker import IDecisionMaker
from service_pb2 import *

if TYPE_CHECKING:
    from src.sample_player_agent import SamplePlayerAgent

# DIFF vs py2d: returns a bool so the keeper falls through to set play.
# helios_set_play has no goalie branch
# and no outfielder may touch a caught ball, so every post-catch restart
# timed out at 49 cycles. is_kickable is false while he holds it, so the
# claim keys off the ball sitting in our own area shortly after a catch.

BOX_X = 36.0
BOX_Y = 20.0
SINCE_CATCH = 60


class GoalieDecisionMaker(IDecisionMaker):
    def __init__(self, agent: "SamplePlayerAgent"):
        self._seen = -1

    def make_decision(self, agent: "SamplePlayerAgent") -> bool:
        wm = agent.wm
        if wm.game_mode_type == GameModeType.PlayOn:
            agent.add_action(PlayerAction(helios_goalie=HeliosGoalie()))
            return True
        if not self._my_restart(wm):
            return False

        a = PlayerAction()
        if wm.self.is_kickable:
            tx, ty = self._clear_to(wm)
            a.body_smart_kick.target_point.x = tx
            a.body_smart_kick.target_point.y = ty
            a.body_smart_kick.first_speed = 2.7
            a.body_smart_kick.first_speed_threshold = 1.5
            a.body_smart_kick.max_steps = 2
        else:
            b = wm.ball.position
            a.body_go_to_point.target_point.x = b.x
            a.body_go_to_point.target_point.y = b.y
            a.body_go_to_point.distance_threshold = 0.5
            a.body_go_to_point.max_dash_power = 100.0
        agent.add_action(a)

        if wm.self.catch_time != self._seen:          # delete once working
            self._seen = wm.self.catch_time
            print("GK restart mode=%s kickable=%s ball=(%.1f,%.1f) "
                  "catch=%s cyc=%s" % (wm.game_mode_type, wm.self.is_kickable,
                                       wm.ball.position.x, wm.ball.position.y,
                                       wm.self.catch_time, wm.cycle),
                  file=sys.stderr, flush=True)
        return True

    @staticmethod
    def _clear_to(wm):
        # upfield and wide, away from our own goal
        y = 25.0 if wm.ball.position.y <= 0.0 else -25.0
        return -10.0, y

    @staticmethod
    def _my_restart(wm) -> bool:
        if wm.game_mode_type not in (GameModeType.GoalieCatch_,
                                     GameModeType.FreeKick_):
            return False
        if (wm.self.catch_time <= 0
                or wm.cycle - wm.self.catch_time > SINCE_CATCH):
            return False
        b = wm.ball.position
        return b.x < -BOX_X and abs(b.y) < BOX_Y