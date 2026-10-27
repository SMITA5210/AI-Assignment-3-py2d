"""Offline tests for the v10 tactical candidate (goal-side defence, charge rule).

usage (repo root):  python3 tools/test_v10.py [path/to/agent.py]   (default tools/versions/v10/agent.py)

Runs every test in tools/test_agent.py against that file, then the v10 checks.
"""
import importlib.util, math, os, sys, types
sys.path.insert(0, os.getcwd())

path = sys.argv[1] if len(sys.argv) > 1 else "tools/versions/v10/agent.py"
spec = importlib.util.spec_from_file_location("src.student.agent", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["src.student.agent"] = mod
spec.loader.exec_module(mod)
g = {"__name__": "t", "__file__": "tools/test_agent.py"}
# v3 tests whose assumption the goal-side defence changes on purpose; they are
# replaced by the checks in section 4 below
CHANGED = ("exactly one chaser per state", "no two defenders on one job")
import contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        exec(compile(open("tools/test_agent.py").read(), "test_agent.py", "exec"), g)
    except SystemExit:
        pass
base_out = buf.getvalue()
print(base_out, end="")
base_fails = [l for l in base_out.splitlines() if l.startswith("FAIL")]
replaced = getattr(sys.modules["src.student.agent"], "FIX_GOALSIDE", False)
# v11+ judges a tackle by who gets the ball afterwards, so a tackle the v3
# test expected (opponent on the ball beyond it, nobody of ours near) is now
# declined on expected value; tools/test_v11.py checks it is not refused for
# charge risk, which is what that test was about
CHANGED_V11 = ("tackle allowed with opponent beyond the ball",)
v11 = hasattr(sys.modules["src.student.agent"], "SAME_CYCLE_KICK")
# v15 stage 2 fits the pass curve to data: passes with margin < -2 succeeded 69 % of the
# time on held-out matches, so the hand-set "blocked lane -> p < 0.5" no longer holds;
# tools/test_v15.py checks the relative version (blocked lane clearly below an open one)
CHANGED_V15 = ("blocked lane is not a high-p pass",)
v15s2 = getattr(sys.modules["src.student.agent"], "STAGE_EVAL", False)
base_fail = len([l for l in base_fails
                 if not ((replaced and any(c in l for c in CHANGED))
                         or (v11 and any(c in l for c in CHANGED_V11))
                         or (v15s2 and any(c in l for c in CHANGED_V15)))])
A, FakeAgent, player, SP, V = g["A"], g["FakeAgent"], g["player"], g["SP"], g["V"]
assert A is mod
SP.foul_detect_probability = 0.5      # values of base.conf (replay header)
SP.red_card_probability = 0.0
import numpy as np

fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), name, detail)


m = A._Model(SP)
ag = A.Agent1()

# 1. goal-side points: in front of the attacker, reachable before him
if hasattr(ag, "_goal_side"):
    att = np.array([-20.0, 0.0])
    pts, t, ok = ag._goal_side(m, att, mod.DRIBBLE_SPEED, np.array([[-15.0, 3.0]]), m.kickable)
    check("goal-side point lies between attacker and goal", pts[0][0] < att[0] and ok[0], f"{pts[0]} t={t[0]:.1f}")
    reach_him = (np.linalg.norm(pts[0] - att)) / mod.DRIBBLE_SPEED
    check("defender gets there first", t[0] <= reach_him - mod.GOAL_SIDE_LEAD + 1e-9, f"{t[0]:.1f} vs {reach_him:.1f}")
    pts, t, ok = ag._goal_side(m, np.array([-40.0, 0.0]), m.speed, np.array([[10.0, 0.0]]), m.mark_dist)
    check("hopeless chase: flagged late, on his line, between him and goal",
          (not ok[0]) and -m.hl - 1e-6 <= pts[0][0] < -40.0 and abs(pts[0][1]) < 1e-6, f"{pts[0]}")


def defend_roles(ours, theirs, ball, op_kick=True):
    roles = {}
    for me in ours:
        mates = [p for p in ours if p is not me]
        a = FakeAgent(me, mates, theirs, ball, (0, 0), op_kick=op_kick)
        got = []
        A.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
        A.Agent1().off_ball(a)
        roles[me.uniform_number] = got[0] if got else ("none", {})
    return roles


# 2. snapshot of a goal conceded by v3 (V3F1, 12 cycles before the goal at 119):
#    ball (-35,-3) at the carrier; v3's four were all behind the ball
GK = player(1, -50, 0, goalie=True)
ours = [player(2, -29, -11), player(3, -31, -15), player(4, -32, 1), player(5, -25, -8)]
theirs = [player(1, 50, 0, goalie=True), player(9, -34.5, -2.5, body=180.0), player(10, -29, -14),
          player(11, -27, -8), player(7, 2, 3)]
r = defend_roles(ours, theirs, (-35.0, -3.0))
kinds = [k for k, _ in r.values()]
if getattr(mod, "FIX_GOALSIDE", False):
    check("no chaser while they have the ball", "chase" not in kinds, str(kinds))
    check("exactly one player on the carrier", kinds.count("cover") == 1, str(kinds))
    tgt = [tuple(f["at"]) for k, f in r.values() if k in ("cover", "mark")]
    check("no two defenders share a job", len(tgt) == len(set(tgt)), str(tgt))
    cov = [f["at"] for k, f in r.values() if k == "cover"]
    check("carrier's cover is goal-side of the ball", bool(cov) and cov[0][0] < -35.0, str(cov))

# 3. charge rule (FIX_TACKLE)
def tackle_note(opp_body, red=0.0):
    SP.red_card_probability = red
    me = player(3, -20.0, 0.0, body=0.0)
    opp = player(9, -19.0, 0.0, body=opp_body)       # between us and the ball, on the ball
    a = FakeAgent(me, [player(2, -30, 5)], [player(1, 50, 0, goalie=True), opp], (-18.3, 0.0),
                  op_kick=True)
    a.wm.self.tackle_probability = 0.9
    got = []
    A.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
    A.Agent1().off_ball(a)
    SP.red_card_probability = 0.0
    return [x for x in got if x[0] in ("tackle", "no_tackle")]

if getattr(mod, "TACKLE_TIMING", False):
    print("skip charge-rule decisions: v12 never tackles a controlled ball (tools/test_v12.py)")
elif getattr(mod, "FIX_TACKLE", False):
    n = tackle_note(0.0)       # he faces away from us: a charge from behind
    check("from behind: charge risk priced, not a veto", bool(n) and n[0][1].get("charge") == 0.5, str(n))
    n = tackle_note(180.0)     # he faces us: no charge
    check("from the front: no charge risk", bool(n) and n[0][1].get("charge") == 0.0, str(n))
    n = tackle_note(0.0, red=0.1)
    check("cards possible -> no tackle from behind", bool(n) and n[0][1].get("why") == "card_risk", str(n))

# 4. replacements for the changed v3 tests (random states, as test_agent 3a)
if replaced:
    import random
    rng = random.Random(11)
    chasers_ok = dup_bad = 0
    T = 300
    for _ in range(T):
        ours, theirs, ball, bvel = g["world"](rng)
        op_kick = rng.random() < 0.3
        roles = {}
        for me in ours:
            mates = [p for p in ours if p is not me] + [g["GK"]]
            a = FakeAgent(me, mates, theirs, ball, bvel, op_kick=op_kick)
            got = []
            A.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
            A.Agent1().off_ball(a)
            roles[me.uniform_number] = got[0]
        labels = [r[0] for r in roles.values()]
        n_chase = labels.count("chase") + labels.count("tackle")
        chasers_ok += (n_chase == 0) if op_kick else (n_chase == 1)
        tgt = [tuple(f["at"]) for k, f in roles.values() if k in ("mark", "cover")]
        goal_mouth = (round(-m.hl + 1, 1), 0.0)
        dups = [t for t in set(tgt) if tgt.count(t) > 1]
        dup_bad += any(t != goal_mouth for t in dups)
    check("one chaser for a loose ball, none while they have it", chasers_ok == T, f"{chasers_ok}/{T}")
    check("no two defenders share a job, except the goal mouth where lines meet",
          dup_bad == 0, f"{dup_bad} states")

print(f"\nbase tests: {base_fail} unexpected failure(s)"
      f"{' (' + str(len(base_fails) - base_fail) + ' replaced by design)' if len(base_fails) != base_fail else ''};"
      f" v10 tests: {fails} failure(s)")
sys.exit(1 if (fails or base_fail) else 0)
