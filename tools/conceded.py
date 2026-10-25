"""For each goal conceded by team ME: its last on-ball decision before it.

Run from the repo root after matches named ME vs TAG:  python3 tools/conceded.py TAG ...
"""
import glob, json, sys
for tag in sys.argv[1:]:
    rcg = glob.glob(f"logs/*_ME_vs_{tag}.rcg")
    rcg = sorted(rcg)[-1]
    names, cyc, goals = {}, 0, []
    for line in open(rcg):
        if line.startswith("(team"): p = line.split(); names = {"l": p[2], "r": p[3]}
        elif line.startswith("(show"): cyc = int(line.split()[1])
        elif line.startswith("(playmode") and "goal_" in line:
            side = line.split()[2].rstrip(")")[-1]
            if names[side] != "ME": goals.append(cyc)
    d = sorted(glob.glob(f"runlogs/trace/*_ME_vs_{tag}"))[-1]
    rows = []
    for p in glob.glob(d + "/ME/*.jsonl"): rows += [json.loads(l) for l in open(p)]
    on = sorted([r for r in rows if r["kickable"]], key=lambda r: r["c"])
    print(tag, "conceded at", goals)
    for g in goals:
        before = [r for r in on if g - 150 < r["c"] < g]
        if not before: print("   ", g, "no possession in 150 cycles before"); continue
        r = before[-1]; n = r["notes"][0]
        print("   ", g, "last ours at", r["c"], "unum", r["unum"], "ball", r["ball"], n["note"], {k: n[k] for k in ("at", "p", "ev", "to") if k in n}, "opp", r["nearest_opp"])
