"""How each goal against a team came about, from replay + trace.

usage (repo root):  python3 tools/goals.py --team ME logs/<match>.rcg [...]

For every goal conceded by --team it prints:
  - the scorer's last touch (shot) position and distance to goal
  - how the opponent got the ball: the turnover (our last touch before
    their unbroken possession), where, and what our player had decided
    there (from the trace note at that cycle)
  - whether a restart (set play, handled by the framework) intervened
  - cycles from turnover to goal
Touches: player-state flags KICK 0x2, CATCH 0x10, TACKLE 0x1000.
"""
import argparse, glob, json, os, re
from collections import Counter

TOUCH = 0x2 | 0x10 | 0x1000
HALF = 52.5


def parse(path):
    names, touches, modes, last_mode = {}, [], [], None
    for line in open(path, errors="replace"):
        if line.startswith("(team"):
            p = line.split(); names = {"l": p[2], "r": p[3]}
        elif line.startswith("(playmode"):
            m = re.match(r"\(playmode (\d+) (\S+?)\)", line)
            if m and m.group(2) != last_mode:
                modes.append((int(m.group(1)), m.group(2))); last_mode = m.group(2)
        elif line.startswith("(show"):
            c = int(line.split()[1])
            b = re.search(r"\(\(b\) (\S+) (\S+)", line)
            bx, by = float(b.group(1)), float(b.group(2))
            for side, unum, st in re.findall(r"\(\((l|r) (\d+)\) \d+ (0x[0-9a-f]+)", line):
                if int(st, 16) & TOUCH:
                    touches.append((c, side, int(unum), bx, by))
    return names, touches, modes


def trace_note(path, team, cycle, unum):
    tag = os.path.basename(path)[:-4]
    f = os.path.join("runlogs", "trace", tag, team, f"{unum}.jsonl")
    if not os.path.exists(f): return None
    for line in open(f):
        r = json.loads(line)
        if r["c"] == cycle:
            n = [x for x in r.get("notes", []) if x["note"] != "model"]
            return n[0] if n else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--team", default="ME")
    a = ap.parse_args()
    summary = Counter()
    for path in a.logs:
        names, touches, modes = parse(path)
        side = {v: k for k, v in names.items()}.get(a.team)
        if side is None: continue
        opp = "r" if side == "l" else "l"
        goals = [c for c, m in modes if m == f"goal_{opp}"]
        print(os.path.basename(path), f"conceded {len(goals)}")
        for g in goals:
            before = [t for t in touches if t[0] <= g]
            if not before:
                continue
            shot = before[-1]
            gx = -HALF if opp == "l" else HALF          # the goal they attack, in field coords
            gx = HALF if opp == "l" else -HALF
            dist = ((gx - shot[3]) ** 2 + shot[4] ** 2) ** 0.5
            ours = [t for t in before if t[1] == side]
            restart = None
            if ours:
                to = ours[-1]
                restart = next((m for c, m in modes if to[0] < c < g and not m.startswith(("play_on", "goal"))), None)
                # a touch shown at cycle c is the command decided at c - 1
                note = trace_note(path, a.team, to[0] - 1, to[2]) or trace_note(path, a.team, to[0], to[2])
                kind = note["note"] if note else "?"
                x = to[3] if side == "l" else -to[3]     # our attacking direction is +x
                where = "own third" if x < -17.5 else ("middle" if x < 17.5 else "final third")
                summary[(kind, where)] += 1
                gap = g - to[0]
                summary[("cycles turnover->goal", "<=25" if gap <= 25 else ("26-75" if gap <= 75 else ">75"))] += 1
                if restart: summary[("restart_before_goal", restart)] += 1
                print(f"   goal@{g}: shot from {dist:.1f} m by #{shot[2]};"
                      f" turnover @{to[0]} by our #{to[2]} at x={x:.1f} ({where}) after '{kind}'"
                      + (f" {json.dumps({k: note[k] for k in ('p', 'm', 'ev') if k in note})}" if note else "")
                      + f"; {g - to[0]} cycles to goal" + (f"; restart: {restart}" if restart else ""))
            else:
                print(f"   goal@{g}: shot from {dist:.1f} m; we never touched it before")
    print("\nturnovers leading to goals (our last decision, area):")
    for k, v in summary.most_common():
        print(f"   {v:3d}  {k}")


if __name__ == "__main__":
    main()
