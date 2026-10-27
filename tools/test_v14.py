"""Offline tests for v14: fitted shot curve (tools/fit_shots.py) and shot_p logging.

usage (repo root):  python3 tools/test_v14.py [path/to/agent.py]   (default tools/versions/v14/agent.py)

Also runs tools/test_v13.py (and through it every earlier test chain).
"""
import importlib.util, json, os, random, subprocess, sys
sys.path.insert(0, os.getcwd())

path = sys.argv[1] if len(sys.argv) > 1 else "tools/versions/v14/agent.py"
prev = subprocess.run([sys.executable, "tools/test_v13.py", path], capture_output=True, text=True)
print(prev.stdout.strip().splitlines()[-1])
prev_ok = prev.returncode == 0


def load(p, name):
    spec = importlib.util.spec_from_file_location("src.student.agent", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["src.student.agent"] = mod
    spec.loader.exec_module(mod)
    return mod


new = load(path, "v14")
g = {"__name__": "t", "__file__": "tools/test_agent.py"}
src = open("tools/test_agent.py").read()
exec(compile(src.split("# 1. one_step_speed")[0], "test_agent", "exec"), g)
exec(compile(src.split("# 3. fake world for whole-decision tests")[1].split("# 3a.")[0], "test_agent", "exec"), g)
FakeAgent, player, SP = g["FakeAgent"], g["player"], g["SP"]
SP.foul_detect_probability, SP.red_card_probability = 0.5, 0.0
old = load("tools/versions/v13/agent.py", "v13")
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), name, detail)


check("fitted shot constants in place", (new.SHOT_OFFSET, new.SHOT_SCALE) == (-1.85, 1.372),
      f"{new.SHOT_OFFSET}, {new.SHOT_SCALE}")


def decide(mod, rng):
    """Our player on the ball 12-25 m from goal, a goalie, two defenders and two mates nearby."""
    d = rng.uniform(12, 25); ang = rng.uniform(-0.6, 0.6)
    bx, by = 52.5 - d * __import__("math").cos(ang), d * __import__("math").sin(ang)
    me = player(9, bx - 0.4, by, body=0.0)
    mates = [player(10, bx - rng.uniform(5, 15), by + rng.uniform(-12, 12)),
             player(11, bx - rng.uniform(5, 15), by + rng.uniform(-12, 12))]
    opps = [player(1, 51.0, rng.uniform(-2, 2), goalie=True),
            player(2, bx + rng.uniform(2, 8), by + rng.uniform(-5, 5)),
            player(3, bx + rng.uniform(3, 10), by + rng.uniform(-8, 8))]
    a = FakeAgent(me, mates, opps, (bx, by), kickable=True, line=45.0)
    got = []
    mod.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
    mod.Agent1().on_ball(a)
    return got


rng_seed = 5
shots = {}
notes_ok = True
for name, mod in (("v13", old), ("v14", new)):
    rng = random.Random(rng_seed)
    n = 0
    for _ in range(150):
        got = decide(mod, rng)
        main = [x for x in got if x[0] in ("shoot", "pass", "dribble", "hold", "clear")]
        n += bool(main) and main[0][0] == "shoot"
        if name == "v14":
            for w, f in got:
                try: json.dumps(f, allow_nan=False)
                except (TypeError, ValueError): notes_ok = False
            if main and "shot_p" not in main[0][1]: notes_ok = False
    shots[name] = n
check("v14 notes carry shot_p and stay JSON-safe", notes_ok)
check("same 150 states 12-25 m out: v14 shoots more often than v13", shots["v14"] > shots["v13"], str(shots))

print(f"\nv13/v12/v11/v10/base tests: {'ok' if prev_ok else 'FAILED'}; v14 tests: {fails} failure(s)")
sys.exit(1 if (fails or not prev_ok) else 0)
