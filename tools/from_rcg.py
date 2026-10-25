"""Load one replay cycle into the fake agent and print Agent1's options.

usage (repo root): python3 tools/from_rcg.py logs/<match>.rcg CYCLE TEAM UNUM
Team-relative frame: left team as-is, right team rotated 180 degrees.
"""
import re, sys, math
HERE = __file__.rsplit("/", 1)[0]
src = open(HERE + "/test_agent.py").read()
exec(src.split("# 1. one_step_speed")[0])
exec(src.split("# 3. fake world for whole-decision tests")[1].split("# 3a.")[0])
import numpy as np

def load(rcg, cycle, team):
    names = {}
    for line in open(rcg):
        if line.startswith("(team"):
            parts = line.split(); names = {parts[2]: "l", parts[3]: "r"}
        if line.startswith(f"(show {cycle} "):
            side = names[team]
            flip = -1.0 if side == "r" else 1.0
            b = re.search(r"\(\(b\) (\S+) (\S+) (\S+) (\S+)\)", line).groups()
            ball = (flip * float(b[0]), flip * float(b[1])); bvel = (flip * float(b[2]), flip * float(b[3]))
            ps = re.findall(r"\(\((l|r) (\d+)\) \d+ (0x[0-9a-f]+) (\S+) (\S+) (\S+) (\S+) (\S+)", line)
            ours, theirs = [], []
            for s, u, st, x, y, vx, vy, bd in ps:
                if int(st, 16) == 0: continue
                bodyd = float(bd) + (180.0 if side == "r" else 0.0)
                bodyd = (bodyd + 180) % 360 - 180
                p = player(int(u), flip * float(x), flip * float(y), bodyd, goalie=(int(st, 16) & 0x8) != 0,
                           vx=flip * float(vx), vy=flip * float(vy))
                (ours if s == side else theirs).append(p)
            return ours, theirs, ball, bvel
    raise SystemExit("cycle not found")

if __name__ == "__main__":
    rcg, cycle, team, unum = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    ours, theirs, ball, bvel = load(rcg, cycle, team)
    me = [p for p in ours if p.uniform_number == unum][0]
    mates = [p for p in ours if p is not me]
    line = max(0.0, sorted([p.position.x() for p in theirs])[-2], ball[0])
    kick = me.position.dist(V(*ball)) < 1.1
    ag = FakeAgent(me, mates, theirs, ball, bvel, kickable=kick, line=line)
    print("me", unum, me.position, "body", round(me.body_direction), "ball", ball, "kickable", kick)
    print("mates", [(p.uniform_number, round(p.position.x(), 1), round(p.position.y(), 1)) for p in mates])
    print("opps ", [(p.uniform_number, round(p.position.x(), 1), round(p.position.y(), 1)) for p in theirs])
    A.note = lambda agent, what, **f: print("CHOSEN", what, f)
    a = A.Agent1()
    (a.on_ball if kick else a.off_ball)(ag)
    for o in sorted(getattr(a, "_last_options", []), key=lambda o: -o[0]):
        print("  ", round(o[0], 3), o[1], np.round(o[2], 1))
    if hasattr(a, "_last_expand"):
        pts, ev, p, kinds = a._last_expand
        for k in np.argsort(-ev)[:12]:
            print("   exp", kinds[k], np.round(pts[k], 1), "ev", round(float(ev[k]), 3), "p", round(float(p[k]), 2),
                  "value", round(float(a._m.value(*pts[k])), 3))
