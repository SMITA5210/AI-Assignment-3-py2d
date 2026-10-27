import math
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

CORNER = (-37.0, 18.5)
HOLD = 35
PASS_BY = 8
OPEN = 10
SAFE_X = -10.0
DECAY = 0.94


class GoalieDecisionMaker(IDecisionMaker):
    def __init__(self, agent: "SamplePlayerAgent"):
        pass

    def make_decision(self, agent: "SamplePlayerAgent") -> bool:
        wm = agent.wm
        if wm.game_mode_type == GameModeType.PlayOn:
            agent.add_action(PlayerAction(helios_goalie=HeliosGoalie()))
            return True
        if not self._my_restart(wm):
            return False

        a = PlayerAction()
        held = wm.cycle - wm.last_set_play_start_time
        corner = self._corner(wm)
        if held <= 3 and math.hypot(wm.self.position.x - corner[0],
                                    wm.self.position.y - corner[1]) > 2.0:
            a.move.x, a.move.y = corner
        elif wm.self.is_kickable:
            kick = self._choose(wm, held)
            if kick is None:
                a.body_turn_to_point.target_point.x = 0.0
                a.body_turn_to_point.target_point.y = 0.0
                a.body_turn_to_point.cycle = 1
            else:
                a.body_smart_kick.target_point.x = kick[0]
                a.body_smart_kick.target_point.y = kick[1]
                a.body_smart_kick.first_speed = kick[2]
                a.body_smart_kick.first_speed_threshold = 1.5
                a.body_smart_kick.max_steps = 2
        else:
            b = wm.ball.position
            a.body_go_to_point.target_point.x = b.x
            a.body_go_to_point.target_point.y = b.y
            a.body_go_to_point.distance_threshold = 0.5
            a.body_go_to_point.max_dash_power = 100.0
        agent.add_action(a)
        return True

    def _choose(self, wm, held):
        b = wm.ball.position
        me = wm.self.uniform_number
        mates = [p.position for p in wm.teammates
                 if p.uniform_number > 0 and p.uniform_number != me]
        opps = [p.position for p in wm.opponents if p.uniform_number > 0]
        passes, longs = [], []
        for deg in range(-84, 85, 6):
            for speed in (1.4, 1.8, 2.2, 2.6, 3.0):
                path = self._roll(b.x, b.y, deg, speed)
                if not path:
                    continue
                ours = self._first(mates, path, 1.0)
                theirs = self._first(opps, path, 1.6)
                if ours is not None and ours <= PASS_BY and (theirs is None or theirs - ours >= OPEN):
                    passes.append((path[ours - 1][0], deg, speed))
                if speed == 3.0:
                    longs.append((path[(theirs or len(path)) - 1][0], deg, speed))
        best = max(passes or longs, key=lambda k: k[0], default=None)
        if best is None or not (passes or best[0] >= SAFE_X or held >= HOLD):
            return None
        _, deg, speed = best
        return b.x + 10.0 * math.cos(math.radians(deg)), b.y + 10.0 * math.sin(math.radians(deg)), speed

    @staticmethod
    def _roll(x, y, deg, speed):
        dx, dy = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        path = []
        while speed > 0.1 and len(path) < 60:
            x, y, speed = x + dx * speed, y + dy * speed, speed * DECAY
            if abs(x) > 50.5 or abs(y) > 32.0:
                return []
            path.append((x, y))
        return path if len(path) >= 10 else []

    @staticmethod
    def _first(players, path, reach):
        for t, (x, y) in enumerate(path, 1):
            if any(math.hypot(p.x - x, p.y - y) - reach <= t - 1 for p in players):
                return t
        return None

    @staticmethod
    def _corner(wm):
        ys = [p.position.y for p in wm.opponents if p.uniform_number > 0]
        lean = sum(y > 0 for y in ys) - sum(y < 0 for y in ys) or (-1 if wm.ball.position.y >= 0 else 1)
        return CORNER[0], CORNER[1] * (-1.0 if lean > 0 else 1.0)

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
