"""Offline checks for src/student/agent.py with a fake world.

Run from the repo root:  python3 tools/test_agent.py
"""
import math, random, sys, types
sys.path.insert(0, ".")
import numpy as np
from pyrusgeom.geom_2d import Vector2D
from pyrusgeom.angle_deg import AngleDeg
import service_pb2 as pb
from src.utils.tools import Tools as RawTools
import src.student.agent as A

SP = types.SimpleNamespace(player_size=0.3, ball_size=0.085, stopped_ball_vel=0.01, goal_post_radius=0.06, kick_rand=0.1,
    max_moment=180.0, inertia_moment=5.0, tackle_cycles=10, max_tackle_power=100.0, max_back_tackle_power=0.0, tackle_power_rate=0.027, 
    ball_decay=0.94, ball_speed_max=3.0, max_power=100.0, ball_accel_max=2.7,
    real_speed_max=1.0, player_speed_max=1.05, kickable_area=1.085,
    catchable_area=1.3, tackle_dist=2.0, pitch_half_length=52.5,
    pitch_half_width=34.0, goal_width=14.02, their_penalty_area_line_x=36.0,
    our_penalty_area_line_x=-36.0, penalty_area_half_width=20.16,
    simulator_step=100, half_time=75, nr_normal_halfs=2, stamina_max=8000.0)
m = A._Model(SP)
fails = 0

def check(name, ok, detail=""):
    global fails
    fails += 0 if ok else 1
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))

# 1. one_step_speed == Tools.calc_max_velocity
rng = random.Random(1)
worst = 0.0
for _ in range(3000):
    bv = np.array([rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)])
    if np.hypot(*bv) > 3: continue
    kr = rng.uniform(0.005, 0.027)
    ang = rng.uniform(-180, 180)
    ref = RawTools.calc_max_velocity(AngleDeg(ang), kr, Vector2D(*bv), SP).r()
    u = np.array([[math.cos(math.radians(ang)), math.sin(math.radians(ang))]])
    got = float(m.one_step_speed(u, bv, kr)[0])
    worst = max(worst, abs(ref - got))
check("one_step_speed matches calc_max_velocity", worst < 1e-6, f"max err {worst:.2e}")

# 2. steps_for == Tools.ball_move_step ; speed_for == interface kick formula
bad = 0
for _ in range(3000):
    s, d = rng.uniform(0.5, 3.0), rng.uniform(0.5, 49.0)
    ref = RawTools.ball_move_step(s, d, 0.94)
    got = m.steps_for(np.array(d), np.array(s))
    got = -1 if not np.isfinite(got) else int(got)
    if ref < 0: ref = -1
    bad += (ref != got) and not (ref == 0 and got == 1)
check("steps_for matches ball_move_step", bad == 0, f"{bad} mismatches")
n = 7; d = 12.0
check("speed_for covers dist in n", abs(float((m.speed_for(d, n) * m.travel[n-1]) - d)) < 1e-9)

# 3. fake world for whole-decision tests
def V(x, y): return Vector2D(x, y)
def player(unum, x, y, body=0.0, goalie=False, vx=0.0, vy=0.0):
    return types.SimpleNamespace(uniform_number=unum, position=V(x, y), velocity=V(vx, vy),
                                 body_direction=body, is_goalie=goalie, dist_from_self=0.0)

import time as _time
class FakeBudget:
    """Counts real time like the framework: match cpu left, and wall ms
    left of a 1s decision cap that starts when the agent is built."""
    def __init__(self, match_ms=70000.0):
        self._c0, self._w0, self._match = _time.process_time(), _time.perf_counter(), match_ms
    @property
    def remaining_ms(self): return self._match - (_time.process_time() - self._c0) * 1000.0
    @property
    def decision_remaining_ms(self): return 1000.0 - (_time.perf_counter() - self._w0) * 1000.0

class FakeStrategy:
    def get_position(self, unum, a): return V(-20 + 5 * unum, 0)

class FakeAgent:
    def __init__(self, me, mates, opps, ball, bvel=(0, 0), kickable=False,
                 tm_kick=False, op_kick=False, line=10.0):
        self.actions = []
        self.server_params = SP
        self.budget = FakeBudget()
        self.strategy = FakeStrategy()
        selfp = types.SimpleNamespace(uniform_number=me.uniform_number, position=me.position,
            velocity=me.velocity, body_direction=me.body_direction, is_kickable=kickable,
            kick_rate=0.025 if kickable else 0.0001, stamina=8000.0,
            tackle_probability=0.0, foul_probability=0.0, is_goalie=False)
        b = types.SimpleNamespace(position=V(*ball), velocity=V(*bvel),
                                  dist_from_self=me.position.dist(V(*ball)))
        self.wm = types.SimpleNamespace(self=selfp, ball=b, teammates=mates, opponents=opps,
            offside_line_x=line, kickable_teammate_existance=tm_kick,
            kickable_opponent_existance=op_kick, cycle=500, intercept_table=None)
    def add_action(self, a): self.actions.append(a)

def kinds(agent): return [a.WhichOneof("action") for a in agent.actions]

GK = player(1, -50, 0, goalie=True)
def world(rng):
    ours = [player(u, rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(-180, 180)) for u in (2, 3, 4, 5)]
    theirs = [player(1, 50, 0, goalie=True)] + [player(u, rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(-180, 180)) for u in (2, 3, 4, 5)]
    ball = (rng.uniform(-45, 45), rng.uniform(-30, 30))
    bvel = (rng.uniform(-2, 2), rng.uniform(-2, 2))
    return ours, theirs, ball, bvel

# 3a. every player derives the same role allocation
agree = 0; trials = 300; one_chaser = 0; defend_dupes = 0
import src.student.interface as I
for _ in range(trials):
    ours, theirs, ball, bvel = world(rng)
    op_kick = rng.random() < 0.3
    roles = {}
    for me in ours:
        mates = [p for p in ours if p is not me] + [GK]
        ag = FakeAgent(me, mates, theirs, ball, bvel, op_kick=op_kick)
        got = []
        A.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
        A.Agent1().off_ball(ag)
        roles[me.uniform_number] = got[0]
    labels = [r[0] for r in roles.values()]
    one_chaser += labels.count("chase") + labels.count("tackle") == 1
    marks = [tuple(r[1]["at"]) for r in roles.values() if r[0] in ("mark", "cover")]
    defend_dupes += len(marks) != len(set(marks))
    agree += "error" not in labels
check("off_ball never raises", agree == trials, f"{agree}/{trials}")
check("exactly one chaser per state", one_chaser == trials, f"{one_chaser}/{trials}")
check("no two defenders on one job", defend_dupes == 0, f"{defend_dupes} dupes")

# 3b. on_ball never raises, always emits a body action, finishes in budget
errs = 0; empty = 0; worst_ms = 0.0
import time
for _ in range(150):
    ours, theirs, ball, bvel = world(rng)
    me = ours[0]
    me.position = V(ball[0] - 0.5, ball[1])
    mates = ours[1:] + [GK]
    ag = FakeAgent(me, mates, theirs, ball, (0.2, 0.0), kickable=True)
    got = []
    A.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
    t = time.process_time(); A.Agent1().on_ball(ag); worst_ms = max(worst_ms, (time.process_time() - t) * 1000)
    errs += any(g[0] == "error" for g in got)
    empty += not ag.actions
    if any(g[0] == "error" for g in got): print("   ", got)
check("on_ball never raises", errs == 0, f"{errs} errors")
check("on_ball always emits", empty == 0)
check("on_ball stops near its share (cap 60ms)", worst_ms < A.THINK_CAP_MS + 15, f"worst {worst_ms:.1f} ms")

# 3c. obvious cases
def decide(me, mates, theirs, ball):
    ag = FakeAgent(me, mates + [GK], theirs, ball, kickable=True, line=40)
    got = []
    A.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
    A.Agent1().on_ball(ag)
    return got[0]
empty_goal = [player(1, 52, 6.5, goalie=True)]
r = decide(player(2, 39.5, 0), [player(3, 0, 0)], empty_goal, (40, 0))
check("shoots at an open goal from 12m", r[0] == "shoot", str(r))
r = decide(player(2, -20.5, 0), [player(3, 0, 0)], [player(1, 50, 0, goalie=True), player(2, -19, 0.5)], (-20, 0))
check("pressed in midfield -> releases the ball (not hold/dribble)", r[0] in ("pass", "clear", "shoot"), str(r))
r = decide(player(2, -10.5, 0), [player(3, 10, 5)], [player(1, 50, 0, goalie=True), player(2, 0, 2.5)], (-10, 0))
check("blocked lane is not a high-p pass", r[0] != "pass" or r[1]["p"] < 0.5, str(r))
r = decide(player(2, 47.5, -22), [player(3, 20, 10)], [player(1, 50, 0, goalie=True), player(2, 10, 20)], (48, -22))
check("unpressed near the corner flag -> moves the ball inwards", r[0] in ("pass", "shoot") or abs(r[1].get("at", [0, 99])[1]) < 22, str(r))
r = decide(player(2, -30.5, 0), [player(3, -10, 0)], [player(1, 50, 0, goalie=True), player(2, 30, 20)], (-30, 0))
check("free in own half -> progresses (pass/dribble)", r[0] in ("pass", "dribble"), str(r))

# 3d. rule filters
cand = m.pass_targets(np.array([0.0, 0.0]), np.array([[30.0, 0.0], [10.0, 0.0]]), line_x=20.0)
check("offside receiver filtered", cand is not None and set(cand[1].tolist()) == {1})
cand = m.pass_targets(np.array([-30.0, 0.0]), np.array([[-45.0, 0.0]]), line_x=0.0)
check("no targets inside our penalty area", cand is None or not ((cand[0][:, 0] < -36) & (abs(cand[0][:, 1]) < 20.16)).any())
cand = m.pass_targets(np.array([-10.0, 0.0]), np.array([[-5.0, 0.0]]), line_x=0.0)
check("own-half receiver never offside", cand is not None)

# 3e. tackle charge geometry (rcssserver Referee::checkFoul)
def tackle_case(opp_xy):
    me = player(2, 0, 0)
    ag = FakeAgent(me, [GK], [player(1, 50, 0, goalie=True), player(5, *opp_xy)], (1.0, 0.0), op_kick=True)
    ag.wm.self.tackle_probability = 0.95
    got = []
    A.note = lambda agent, what, **f: (what != "model" and got.append(what))
    a = A.Agent1(); a._m = A._Model(SP)
    return a._tackle(ag, a._m, np.array([1.0, 0.0]), np.array([[10.0, 5.0]]), np.array([[50.0, 0.0], list(opp_xy)])), got
ok_between, g1 = tackle_case((0.6, 0.1))     # opponent between me and the ball
ok_beyond, g2 = tackle_case((1.8, 0.0))      # opponent on the far side of the ball
ok_wide, g3 = tackle_case((0.5, 1.5))        # opponent well off the line
check("no tackle through an opponent (charge risk)", not ok_between and "no_tackle" in g1, str(g1))
check("tackle allowed with opponent beyond the ball", ok_beyond, str(g2))
check("tackle allowed with opponent off the line", ok_wide, str(g3))

print("\n%d failure(s)" % fails)
sys.exit(1 if fails else 0)
