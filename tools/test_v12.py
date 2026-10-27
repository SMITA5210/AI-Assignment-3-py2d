"""Offline tests for v12: tackle timing (never a controlled ball; a loose ball
an opponent would reach first may be tackled).

usage (repo root):  python3 tools/test_v12.py [path/to/agent.py]   (default tools/versions/v12/agent.py)

Also runs tools/test_v11.py (which runs test_v10.py and the v3 base tests).
"""
import importlib.util, os, subprocess, sys
sys.path.insert(0, os.getcwd())

path = sys.argv[1] if len(sys.argv) > 1 else "tools/versions/v12/agent.py"
prev = subprocess.run([sys.executable, "tools/test_v11.py", path], capture_output=True, text=True)
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
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), name, detail)


def run(me_xy, opp_xy, ball, bvel, mates, op_kick, tprob=0.9, opp_body=180.0):
    me = player(3, *me_xy, body=0.0)
    a = FakeAgent(me, [player(u, x, y) for u, (x, y) in mates],
                  [player(1, 50, 0, goalie=True), player(9, *opp_xy, body=opp_body)], ball, bvel,
                  op_kick=op_kick)
    a.wm.self.tackle_probability = tprob
    got = []
    mod.note = lambda agent, what, **f: (what != "model" and got.append((what, f)))
    mod.Agent1().off_ball(a)
    acts = [type(x).__name__ if not hasattr(x, "WhichOneof") else x.WhichOneof("action") for x in a.actions]
    return got, a.actions


def kinds(got):
    return [k for k, _ in got]


# 1. a controlled ball (carrier within kickable range) is never tackled
got, acts = run((-20.0, 0.0), (-18.6, 0.0), (-19.0, 0.0), (0.0, 0.0), [(2, (-45, 25))], op_kick=True)
check("controlled ball: no tackle, reason logged",
      "tackle" not in kinds(got) and any(k == "no_tackle" and f.get("why") == "timing" for k, f in got), str(got))

# 2. a loose ball rolling to an opponent who gets there first, we are in
#    tackle range: the tackle is considered (taken or refused on EV, not timing)
got, acts = run((-21.2, 0.0), (-16.0, 0.0), (-19.0, 0.0), (0.6, 0.0), [(2, (-15, 4))], op_kick=False)
t = [(k, f) for k, f in got if k in ("tackle", "no_tackle")]
check("loose ball they reach first: tackle considered on its merits",
      bool(t) and (t[0][0] == "tackle" or t[0][1].get("why") == "ev"), str(got))

# 3. a loose ball we reach first: no tackle, we go and get it
got, acts = run((-20.0, 0.0), (-8.0, 0.0), (-19.0, 0.0), (-0.3, 0.0), [(2, (-30, 4))], op_kick=False)
check("loose ball we reach first: no tackle, chase", "tackle" not in kinds(got) and "chase" in kinds(got), str(got))

# 4. no tackle chance at all (tackle_probability 0): nothing tackle-related happens
got, acts = run((-20.0, 0.0), (-18.6, 0.0), (-19.0, 0.0), (0.0, 0.0), [(2, (-45, 25))], op_kick=True, tprob=0.0)
check("out of tackle range: no tackle notes", not any(k in ("tackle", "no_tackle") for k in kinds(got)), str(got))

print(f"\nv11/v10/base tests: {'ok' if prev_ok else 'FAILED'}; v12 tests: {fails} failure(s)")
sys.exit(1 if (fails or not prev_ok) else 0)
