"""Offline tests for v13: a tackle that sends the ball to our own goalie inside
our penalty area is a back-pass foul, not a possession win.

usage (repo root):  python3 tools/test_v13.py [path/to/agent.py]   (default tools/versions/v13/agent.py)

Also runs tools/test_v12.py (and through it test_v11, test_v10 and the v3 base tests).
"""
import importlib.util, os, subprocess, sys, types
sys.path.insert(0, os.getcwd())

path = sys.argv[1] if len(sys.argv) > 1 else "tools/versions/v13/agent.py"
prev = subprocess.run([sys.executable, "tools/test_v12.py", path], capture_output=True, text=True)
print(prev.stdout.strip().splitlines()[-1])
prev_ok = prev.returncode == 0

spec = importlib.util.spec_from_file_location("src.student.agent", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["src.student.agent"] = mod
spec.loader.exec_module(mod)
g = {"__name__": "t", "__file__": "tools/test_agent.py"}
src = open("tools/test_agent.py").read()
exec(compile(src.split("# 1. one_step_speed")[0], "test_agent", "exec"), g)
exec(compile(src.split("# 3. fake world for whole-decision tests")[1].split("# 3a.")[0], "test_agent", "exec"), g)
FakeAgent, player, SP = g["FakeAgent"], g["player"], g["SP"]
SP.foul_detect_probability, SP.red_card_probability = 0.5, 0.0
import numpy as np
m = mod._Model(SP)
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), name, detail)


def team(*pl):
    if not pl:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    a = np.array(pl, float)
    return a[:, :2], a[:, 2], np.zeros(len(a))


# 1. ball rolling to our goalie inside our penalty area: back-pass foul
r = m.touch_outcome(np.array([-38.0, 0.0]), np.array([-1.2, 0.0]), team((-20, 20, 0)),
                    team((-25, -20, 0)), keeper=team((-49, 0, 0)))
check("goalie first inside our box -> back-pass foul", r["how"] == "back_pass" and r["p"] == 0.0
      and abs(r["value"] + mod.BACK_PASS_COST) < 1e-9, str(r))

# 2. same, without the keeper argument (v11/v12 behaviour) the goalie is ignored
r2 = m.touch_outcome(np.array([-38.0, 0.0]), np.array([-1.2, 0.0]), team((-20, 20, 0)), team((-25, -20, 0)))
check("without keeper info it is not a back pass", r2["how"] != "back_pass", str(r2))

# 3. goalie first but outside the penalty area: an ordinary teammate (he cannot catch there)
r = m.touch_outcome(np.array([-20.0, 0.0]), np.array([-1.0, 0.0]), team((10, 20, 0)),
                    team((10, -20, 0)), keeper=team((-30, 0, 0)))
check("goalie first outside our box -> ours", r["how"] == "ours" and r["p"] > 0.5 and r["at"][0] > -m.pen_x, str(r))

# 4. an outfield teammate gets there before the goalie: ours, no foul
r = m.touch_outcome(np.array([-38.0, 0.0]), np.array([-1.2, 0.0]), team((-41, 1, 0)),
                    team((-25, -20, 0)), keeper=team((-50, 0, 0)))
check("outfield teammate first -> ours, no foul", r["how"] == "ours" and r["p"] > 0.5, str(r))

# 5. decision level: a loose ball near our box, an opponent closing in, our goalie
#    behind us: whatever v13 does, its chosen tackle is never a back-pass route
me = player(3, -40.0, 3.0, body=180.0)
a = FakeAgent(me, [player(1, -50, 0, goalie=True), player(2, -30, 20)],
              [player(1, 50, 0, goalie=True), player(9, -36.0, 0.0, body=180.0)], (-39.0, 2.0), (0.5, -0.2),
              op_kick=False)
a.wm.teammates[0].is_goalie = True
a.wm.self.tackle_probability = 0.9
got = []
mod.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
mod.Agent1().off_ball(a)
t = [f for k, f in got if k == "tackle"]
check("near our box: no tackle is chosen on a back-pass route", all(f.get("how") != "back_pass" for f in t), str(got))

print(f"\nv12/v11/v10/base tests: {'ok' if prev_ok else 'FAILED'}; v13 tests: {fails} failure(s)")
sys.exit(1 if (fails or not prev_ok) else 0)
