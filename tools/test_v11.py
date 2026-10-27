"""Offline tests for v11: tackle outcomes judged by who reaches the ball
along its path (_Model.touch_outcome) instead of a race to a clipped landing
point.

usage (repo root):  python3 tools/test_v11.py [path/to/agent.py]   (default tools/versions/v11/agent.py)

Also runs tools/test_v10.py (all v3 base tests + the v10 checks) on the same file.
"""
import importlib.util, os, subprocess, sys
sys.path.insert(0, os.getcwd())

path = sys.argv[1] if len(sys.argv) > 1 else "tools/versions/v11/agent.py"
base = subprocess.run([sys.executable, "tools/test_v10.py", path], capture_output=True, text=True)
print(base.stdout.strip().splitlines()[-1])
base_ok = base.returncode == 0

spec = importlib.util.spec_from_file_location("agent_v11", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import numpy as np
import types

SP = types.SimpleNamespace(player_size=0.3, ball_size=0.085, stopped_ball_vel=0.01, goal_post_radius=0.06,
    kick_rand=0.1, max_moment=180.0, inertia_moment=5.0, tackle_cycles=10, max_tackle_power=100.0,
    max_back_tackle_power=0.0, tackle_power_rate=0.027, ball_decay=0.94, ball_speed_max=3.0, max_power=100.0,
    ball_accel_max=2.7, real_speed_max=1.0, player_speed_max=1.05, kickable_area=1.085, catchable_area=1.3,
    tackle_dist=2.0, pitch_half_length=52.5, pitch_half_width=34.0, goal_width=14.02,
    their_penalty_area_line_x=36.0, our_penalty_area_line_x=-36.0, penalty_area_half_width=20.16,
    simulator_step=100, half_time=75, nr_normal_halfs=2, stamina_max=8000.0)
m = mod._Model(SP)
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), name, detail)


def team(*pl):
    """(x, y, body) tuples -> (positions, bodies, speeds), standing still."""
    if not pl:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    a = np.array(pl, float)
    return a[:, :2], a[:, 2], np.zeros(len(a))


def out(ball, vel, ours, theirs):
    return m.touch_outcome(np.array(ball, float), np.array(vel, float), ours, theirs)


# 1. our player reaches it before all opponents
r = out((0, 0), (1.5, 0), team((8, 0.5, 180)), team((-10, 20, 0), (5, 25, 0)))
check("our player first", r["how"] == "ours" and r["p"] > 0.8 and r["t_us"] < (r["t_them"] or 99), str(r))

# 2. an opponent reaches it first
r = out((0, 0), (1.5, 0), team((-10, 20, 0)), team((8, 0.5, 180)))
check("opponent first", r["how"] == "theirs" and r["p"] < 0.2, str(r))

# 3. the carrier, next to the ball, recovers it at once after a weak touch
r = out((0, 0), (0.3, 0), team((-6, 0, 0)), team((0.7, 0.3, 180)))
check("carrier recovers immediately", r["t_them"] == 1.0 and r["p"] < 0.2, str(r))

# 4. ball sent backward / sideways: the path really goes that way
r = out((-20, 0), (-0.9, 0), team((-32, 0, 0)), team((-19.3, 0, 180)))
check("backward touch follows a backward path", r["at"][0] < -20.0, str(r))
r = out((0, 0), (0, 0.9), team((0, 9, -90)), team((0, -9, 90)))
check("sideways touch goes to whoever is on that side", r["how"] == "ours" and r["at"][1] > 0, str(r))

# 5. out of bounds: not clipped back as a normal chance
r = out((0, 33), (0, 2.0), team((-20, 0, 0)), team((20, 0, 0)))
check("over the touchline -> their restart, p=0", r["how"] == "out" and r["p"] == 0.0
      and abs(r["at"][1] - 34.0) < 1e-6 and r["value"] < 0, str(r))
r = out((-50, 0), (-2.0, 0), team((0, 30, 0)), team((0, -30, 0)))
check("into our own goal -> -1", r["how"] == "own_goal" and r["value"] == -1.0, str(r))
r = out((-51, 20), (-2.0, 0), team((0, 30, 0)), team((0, -30, 0)))
check("over our goal line wide of the posts -> their restart", r["how"] == "out", str(r))

# 6. several teammates contest: the earliest one counts
r1 = out((0, 0), (1.5, 0), team((15, 3, 180)), team((20, -3, 180)))
r2 = out((0, 0), (1.5, 0), team((15, 3, 180), (6, 1, 180)), team((20, -3, 180)))
check("a nearer second teammate gets there sooner", r2["t_us"] < r1["t_us"] and r2["p"] >= r1["p"],
      f"{r1['t_us']} -> {r2['t_us']}")

# 7. nobody can reach the path within the horizon: race to where it stops
#    (with base.conf the horizon runs until the ball stops, 93 cycles, so
#    on this pitch someone always gets there; a larger stopped_ball_vel gives
#    a short horizon to exercise this branch)
m_short = mod._Model(types.SimpleNamespace(**{**vars(SP), "stopped_ball_vel": 1.0}))
r = m_short.touch_outcome(np.array([0.0, 0.0]), np.array([0.5, 0.0]),
                          team((-40, 30, 0)), team((45, -30, 0)))
check("nobody reaches it in the horizon -> race to the resting point",
      r["how"] == "none" and 0 <= r["p"] <= 1 and m_short.horizon < 40,
      f"horizon {m_short.horizon}: {r}")

# 8. decision level, with the fake world of tools/test_agent.py
spec_t = importlib.util.spec_from_file_location("src.student.agent", path)
mod_t = importlib.util.module_from_spec(spec_t)
sys.modules["src.student.agent"] = mod_t
spec_t.loader.exec_module(mod_t)
g = {"__name__": "t", "__file__": "tools/test_agent.py"}
exec(compile(open("tools/test_agent.py").read().split("# 1. one_step_speed")[0], "test_agent", "exec"), g)
exec(compile(open("tools/test_agent.py").read().split("# 3. fake world for whole-decision tests")[1]
             .split("# 3a.")[0], "test_agent", "exec"), g)
FakeAgent, player, SPt = g["FakeAgent"], g["player"], g["SP"]
SPt.foul_detect_probability, SPt.red_card_probability = 0.5, 0.0


def decide(me_xy, me_body, opp_xy, opp_body, ball, mates):
    me = player(3, *me_xy, body=me_body)
    a = FakeAgent(me, [player(u, x, y) for u, (x, y) in mates],
                  [player(1, 50, 0, goalie=True), player(9, *opp_xy, body=opp_body)], ball, op_kick=True)
    a.wm.self.tackle_probability = 0.9
    got = []
    mod_t.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
    mod_t.Agent1().off_ball(a)
    return [x for x in got if x[0] in ("tackle", "no_tackle")]


# carrier on the ball, facing us (no charge), no teammate near: a poke he can
# recover at once is not worth being frozen for -> declined; same-cycle risk logged
timing = getattr(mod_t, "TACKLE_TIMING", False)
if timing:
    print("skip v11 controlled-carrier decisions: v12 never tackles a controlled ball (tools/test_v12.py)")
n = [] if timing else decide((-20.0, 0.0), 0.0, (-18.6, 0.0), 180.0, (-19.0, 0.0), [(2, (-45, 25))])
timing or check("carrier on the ball: same-cycle kick risk is priced", bool(n) and n[0][1].get("same_cycle") == mod_t.SAME_CYCLE_KICK, str(n))
timing or check("weak poke next to the carrier, no support -> declined", bool(n) and n[0][0] == "no_tackle", str(n))
# replacement for the v3 test "tackle allowed with opponent beyond the ball":
# with the opponent beyond the ball there is no charge risk; a refusal must be
# on expected value, never "charge_risk"
n = [] if timing else decide((0.0, 0.0), 0.0, (1.8, 0.0), 0.0, (1.0, 0.0), [(2, (10, 5))])
timing or check("opponent beyond the ball: no charge risk", bool(n) and n[0][1].get("charge") == 0.0
      and n[0][1].get("why") != "charge_risk", str(n))
# loose ball near nobody of theirs (not kickable for him): no same-cycle risk
n = decide((-20.0, 0.0), 0.0, (-15.5, 0.0), 180.0, (-19.0, 0.0), [(2, (-10, 2))])
check("opponent not on the ball: no same-cycle term", (not n) or n[0][1].get("same_cycle", 0.0) == 0.0, str(n))
print(f"\nbase+v10 tests: {'ok' if base_ok else 'FAILED'}; v11 tests: {fails} failure(s)")
sys.exit(1 if (fails or not base_ok) else 0)
