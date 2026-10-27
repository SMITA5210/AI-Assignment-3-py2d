"""Offline tests for v15 (bounded tactical search, built in stages on v14).

usage (repo root):  python3 tools/test_v15.py [path/to/agent.py]   (default tools/versions/v15/agent.py)

Stage 1: the whole v14 test chain on v15; with every STAGE_* switch off, v15 must
decide exactly like v14 (same notes, same actions) on random on-ball and
off-ball states; decision-time profile on the fake world.
Later stages add their own sections below.
"""
import importlib.util, json, os, random, subprocess, sys, time
sys.path.insert(0, os.getcwd())

path = sys.argv[1] if len(sys.argv) > 1 else "tools/versions/v15/agent.py"
prev = subprocess.run([sys.executable, "tools/test_v14.py", path], capture_output=True, text=True)
print(prev.stdout.strip().splitlines()[-1])
prev_ok = prev.returncode == 0


def load(p):
    spec = importlib.util.spec_from_file_location("src.student.agent", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.student.agent"] = mod
    spec.loader.exec_module(mod)
    return mod


v15 = load(path)
g = {"__name__": "t", "__file__": "tools/test_agent.py"}
src = open("tools/test_agent.py").read()
exec(compile(src.split("# 1. one_step_speed")[0], "test_agent", "exec"), g)
exec(compile(src.split("# 3. fake world for whole-decision tests")[1].split("# 3a.")[0], "test_agent", "exec"), g)
FakeAgent, player, SP, world, GK = g["FakeAgent"], g["player"], g["SP"], g["world"], g["GK"]
SP.foul_detect_probability, SP.red_card_probability = 0.5, 0.0
v14 = load("tools/versions/v14/agent.py")
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), name, detail)


STAGES = ("STAGE_EVAL", "STAGE_LOOKAHEAD", "STAGE_RESPONSE", "STAGE_CSP", "STAGE_OFFBALL")
on = [s for s in STAGES if getattr(v15, s, None)]
check("all stage switches exist", all(hasattr(v15, s) for s in STAGES), str([s for s in STAGES if not hasattr(v15, s)]))


def run(mod, state, kickable, op_kick, cycle):
    ours, theirs, ball, bvel = state
    me = ours[0]
    mates = [p for p in ours if p is not me] + [GK]
    a = FakeAgent(me, mates, theirs, ball, bvel, kickable=kickable, op_kick=op_kick)
    a.wm.cycle = cycle
    got = []
    mod.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
    t = time.perf_counter()
    (mod.Agent1().on_ball if kickable else mod.Agent1().off_ball)(a)
    ms = (time.perf_counter() - t) * 1000.0
    acts = [str(x) for x in a.actions]
    return got, acts, ms


# stage 1: equivalence with v14 with every switch forced off (they are read at run time);
# both search to full depth, so wall-clock timing cannot change the result
saved = {k: getattr(v15, k) for k in STAGES}
for k in STAGES: setattr(v15, k, False)
clock_ok = (v14._Clock.ok, v15._Clock.ok)
v14._Clock.ok = v15._Clock.ok = lambda self: True
if True:
    rng = random.Random(21)
    same = total = depth_diff = 0
    diffs = []
    for i in range(300):
        st = world(rng)
        kickable = i % 2 == 0
        if kickable:                      # put the ball at our first player's feet
            ours, theirs, ball, bvel = st
            ball = (ours[0].position.x() + 0.5, ours[0].position.y()); st = (ours, theirs, ball, (0.0, 0.0))
        op_kick = (not kickable) and rng.random() < 0.3
        a = run(v14, st, kickable, op_kick, 100 + i)
        b = run(v15, st, kickable, op_kick, 100 + i)
        total += 1
        # what the agent does: body actions, and each note's kind and target;
        # the reached search depth (and so ev) depends on wall-clock time, not code
        key = lambda r: (r[1], [(k, f.get("at"), f.get("to")) for k, f in r[0]])
        if key(a) == key(b):
            same += 1
            depth_diff += any(x[1].get("depth") != y[1].get("depth") for x, y in zip(a[0], b[0]))
        elif len(diffs) < 3:
            diffs.append((a[0][:1], b[0][:1]))
    check("switches off: v15 decides exactly like v14", same == total, f"{same}/{total} {diffs}")
    print(f"     (same decision but a different search depth reached, from timing: {depth_diff})")

for k, v in saved.items(): setattr(v15, k, v)
v14._Clock.ok, v15._Clock.ok = clock_ok

# stage 2: pass model (STAGE_EVAL)
import numpy as np
mdl = v15._Model(SP)
SP.player_decay = 0.4
def cand_to(origin, target, n):
    vec = np.array(target, float) - origin; dist = float(np.linalg.norm(vec))
    return (np.array([target], float), np.array([0]), np.array([float(n)]),
            np.array([mdl.speed_for(dist, n)]), (vec / dist)[None], np.array([dist]))
O = np.array([0.0, 0.0]); Z2 = np.zeros((1, 2))
def p2(opp, ovel=(0, 0), obody=None, mate=(15.0, 0.0), mvel=(0, 0), mbody=None, target=(15.0, 0.0), n=8, delay=0.0):
    c = cand_to(O, target, n)
    opps = np.array([opp], float).reshape(-1, 2)
    return float(mdl.pass_p2(O, c, delay, np.array([mate], float), np.array([mvel], float),
                             None if mbody is None else np.array([mbody], float),
                             opps, np.array([ovel], float).reshape(-1, 2),
                             None if obody is None else np.array([obody], float),
                             np.zeros(len(opps), bool))[0][0])
rng = random.Random(9)
vals = [p2((rng.uniform(-20, 30), rng.uniform(-20, 20)), (rng.uniform(-1, 1), rng.uniform(-1, 1)),
           rng.uniform(-180, 180), (rng.uniform(5, 30), rng.uniform(-15, 15)), (rng.uniform(-1, 1), rng.uniform(-1, 1)),
           rng.uniform(-180, 180), (rng.uniform(5, 30), rng.uniform(-15, 15)), rng.randint(3, 25), rng.choice((0.0, 1.0)))
        for _ in range(500)]
check("stage 2: probabilities valid", all(0.0 <= v <= 1.0 and v == v for v in vals), f"min {min(vals):.3f} max {max(vals):.3f}")
check("stage 2: an opponent nearer the lane lowers p", p2((8, 1.0)) < p2((8, 6.0)) < p2((8, 15.0)),
      f"{p2((8, 1.0)):.3f} {p2((8, 6.0)):.3f} {p2((8, 15.0)):.3f}")
check("stage 2: an opponent running into the lane lowers p", p2((8, 4.0), ovel=(0, -0.8)) < p2((8, 4.0), ovel=(0, 0.8)),
      f"{p2((8, 4.0), ovel=(0, -0.8)):.3f} vs {p2((8, 4.0), ovel=(0, 0.8)):.3f}")
check("stage 2: an opponent facing away pays a turn", p2((8, 4.0), obody=90.0) > p2((8, 4.0), obody=-90.0),
      f"{p2((8, 4.0), obody=90.0):.3f} vs {p2((8, 4.0), obody=-90.0):.3f}")
check("stage 2: pressure on the passer lowers p", p2((1.5, 0.5)) < p2((4.0, 8.0)),
      f"{p2((1.5, 0.5)):.3f} vs {p2((4.0, 8.0)):.3f}")
check("stage 2: a receiver moving to the ball raises p (target ahead of him)",
      p2((30, 20), mate=(12, 0), mvel=(0.8, 0), target=(18, 0), n=6) > p2((30, 20), mate=(12, 0), mvel=(-0.8, 0), target=(18, 0), n=6))
c = cand_to(O, (15.0, 0.0), 8)
pn = mdl.pass_p2(O, c, 0.0, np.array([[15.0, 0.0]]), Z2, None, np.zeros((0, 2)), np.zeros((0, 2)), None, np.zeros(0, bool))
check("stage 2: no opponents -> p equals the receiver/noise factor", abs(pn[0][0] - pn[2][0]) < 1e-9 and pn[3][0] == 0.0, str(pn))
check("stage 2: receiver already on the target", 0.0 <= p2((30, 20), mate=(15, 0), n=8) <= 1.0)
check("stage 2: candidate generation unchanged (same targets as v14)",
      np.allclose(mdl.pass_targets(O, np.array([[10.0, 5.0], [20.0, -8.0]]), 30.0)[0],
                  v14._Model(SP).pass_targets(O, np.array([[10.0, 5.0], [20.0, -8.0]]), 30.0)[0]))
# replacement for the v3 test "blocked lane is not a high-p pass": the same scene, with
# the defender in the lane and with him far away; the blocked version must score clearly lower
def best_to_mate(opp_xy):
    ag = v15.Agent1(); a = FakeAgent(player(2, -10.5, 0), [player(3, 10, 5), GK],
                                     [player(1, 50, 0, goalie=True), player(2, *opp_xy)], (-10, 0), kickable=True, line=40)
    got = []; v15.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
    ag.on_ball(a)
    pts, ev, p, kinds = ag._last_expand
    feet = [i for i, k in enumerate(kinds) if k == "pass" and np.linalg.norm(pts[i] - np.array([10.0, 5.0])) < 0.5]
    return max(float(p[i]) for i in feet)
pb, po = best_to_mate((0, 2.5)), best_to_mate((0, 20))
check("stage 2: a pass through a blocked lane scores clearly below the open lane", pb < po - 0.1,
      f"blocked {pb:.3f} open {po:.3f}")
check("stage 2: fitted constants in place", (v15.PASS2_OFFSET, v15.PASS2_SCALE, v15.PASS2_PRESS) == (-7.533, 1.730, 2.613))

# profile (fake world, single process; real matches add the framework's own cost)
rng = random.Random(3)
ms_on, ms_off = [], []
for i in range(120):
    st = world(rng)
    ours, theirs, ball, bvel = st
    if i % 2 == 0:
        st = (ours, theirs, (ours[0].position.x() + 0.5, ours[0].position.y()), (0.0, 0.0))
        ms_on.append(run(v15, st, True, False, 100 + i)[2])
    else:
        ms_off.append(run(v15, st, False, rng.random() < 0.3, 100 + i)[2])
import numpy as np
print(f"     profile: on-ball mean {np.mean(ms_on):.1f} ms, p95 {np.percentile(ms_on, 95):.1f}, max {max(ms_on):.1f};"
      f" off-ball mean {np.mean(ms_off):.1f} ms, max {max(ms_off):.1f}")
check("on-ball decisions stay near the 60 ms think cap", max(ms_on) < v15.THINK_CAP_MS * 1.5, f"max {max(ms_on):.1f} ms")
check("off-ball decisions are cheap", max(ms_off) < 50.0, f"max {max(ms_off):.1f} ms")

print(f"\nv14/.../base tests: {'ok' if prev_ok else 'FAILED'}; v15 tests: {fails} failure(s); stages on: {on or 'none'}")
sys.exit(1 if (fails or not prev_ok) else 0)
