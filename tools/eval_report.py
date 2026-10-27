"""Summarise an interleaved evaluation (tools/eval_run.sh output) per version.

usage (repo root):  python3 tools/eval_report.py runlogs/eval/stageP.tsv [--boot BASE CAND]

Per version: W/D/L, points/match, GF, GA, our shots and big chances (<20 m),
opponent shots and big chances, turnovers in our half / our defensive third,
fouls, planning time; and per-match lines. With --boot, a match-bootstrap 90%
interval (4000 replicates, seed 7) of CAND - BASE for points/match, goal
difference and goals against. Metrics come from tools/metrics.py (replay only).
"""
import math, os, re, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import metrics
import keep_data as K

TOUCH = 0x2 | 0x10 | 0x1000


def def_third_turnovers(path):
    """Our touch followed by an opponent touch with the ball in our defensive third."""
    g = K.parse_rcg(path)
    me = {v: k for k, v in g["names"].items()}["ME"]
    sg = 1.0 if me == "l" else -1.0
    hl = g["sp"].get("pitch_half_length", 52.5)
    seq = [(c, s) for c in sorted(g["touch"]) for s, _ in g["touch"][c]]
    n = 0
    for (pc, ps), (c, s) in zip(seq, seq[1:]):
        if ps == me and s != me and sg * g["ball"][pc][0] < -hl / 3:
            n += 1
    return n


def load(tsv):
    rows = []
    for line in open(tsv):
        f = line.rstrip("\n").split("\t")
        if len(f) < 4: continue
        version = f[0].split("/")[-2]
        rep = os.path.join("logs", f[2])
        m = metrics.analyse(rep)
        me, opp = m["ME"], [v for k, v in m.items() if k != "ME"][0]
        gf, ga = me["goals"], opp["goals"]
        rows.append({"version": version, "name": f[1], "gf": gf, "ga": ga,
                     "pts": 3 if gf > ga else (1 if gf == ga else 0),
                     "shots": me["shots"], "big": me["big_chances"],
                     "oshots": opp["shots"], "obig": opp["big_chances"],
                     "to_own_half": me["turnovers_own_half"], "to_def_third": def_third_turnovers(rep),
                     "fouls": me["fouls"], "plan": float(f[4].rstrip("%")) if len(f) > 4 and f[4] else float("nan"),
                     "result": f[3]})
    return rows


def main():
    tsvs = [a for a in sys.argv[1:] if a.endswith(".tsv")]
    rows = [r for t in tsvs for r in load(t)]
    by = defaultdict(list)
    for r in rows: by[r["version"]].append(r)
    cols = ("pts", "gf", "ga", "shots", "big", "oshots", "obig", "to_own_half", "to_def_third", "fouls", "plan")
    print(f"{'version':8s} {'n':>3s} {'W-D-L':>8s} " + " ".join(f"{c:>9s}" for c in cols) + "   (per-match means)")
    for v, rs in sorted(by.items()):
        w = sum(r["pts"] == 3 for r in rs); d = sum(r["pts"] == 1 for r in rs); l = len(rs) - w - d
        print(f"{v:8s} {len(rs):3d} {f'{w}-{d}-{l}':>8s} " + " ".join(f"{np.nanmean([r[c] for r in rs]):9.2f}" for c in cols))
    print("\nper match (version name GF-GA shots/opp shots plan%):")
    for v, rs in sorted(by.items()):
        print(f"  {v:6s} " + "  ".join(f"{r['name']}:{r['gf']}-{r['ga']}" for r in rs))
    if "--boot" in sys.argv:
        i = sys.argv.index("--boot"); base, cand = sys.argv[i + 1], sys.argv[i + 2]
        rng = np.random.default_rng(7)
        b, c = by[base], by[cand]
        out = defaultdict(list)
        for _ in range(4000):
            bb = [b[j] for j in rng.integers(len(b), size=len(b))]
            cc = [c[j] for j in rng.integers(len(c), size=len(c))]
            for k, f in (("points", lambda r: r["pts"]), ("goal diff", lambda r: r["gf"] - r["ga"]),
                         ("goals against", lambda r: r["ga"]), ("goals for", lambda r: r["gf"])):
                out[k].append(np.mean([f(r) for r in cc]) - np.mean([f(r) for r in bb]))
        print(f"\nbootstrap {cand} - {base} (4000 replicates, seed 7, matches resampled within each version):")
        for k, v in out.items():
            lo, hi = np.percentile(v, [5, 95])
            print(f"  {k:14s} {np.mean(v):+.3f}  90% [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
