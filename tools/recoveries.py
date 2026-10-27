"""What happens after each team recovers the ball (from the replays; decisions from our traces).

usage (repo root):  python3 tools/recoveries.py 'OPPNAME_GLOB' [...]      e.g. 'B13_*'

A recovery is a team's first touch after the other team's touch, in play_on.
For each one: where it happened, how long the team then kept the ball (until
the other team touches it or play stops), how far the ball got toward the
opponent goal, whether a shot came of it (tools/metrics.py shot rule), and how
it ended. For ME also the first on-ball decisions (trace notes, kickable rows).
Touches count a TACKLE flag only in its first cycle (tools/keep_data.py).
"""
import glob, json, math, os, sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import keep_data as K
from metrics import heads_for_goal, HALF


def recoveries(g, side):
    """[(cycle, x in side's frame, cycles kept, progress m, shot?, end)] for `side`."""
    sg = 1.0 if side == "l" else -1.0
    seq = [(c, s) for c in sorted(g["touch"]) for s, _ in g["touch"][c]]
    stops = sorted(c for c, m in g["modes"].items() if m != "play_on")
    out = []
    for i in range(1, len(seq)):
        c, s = seq[i]
        if s != side or seq[i - 1][1] == side or g["modes"].get(c, "play_on") not in ("play_on",) and c in g["modes"]:
            continue
        if any(seq[i - 1][0] < t <= c for t in stops):
            continue                                  # a restart came between: not a recovery in play
        x0 = sg * g["ball"][c][0]
        last, maxx, shot, end = c, x0, False, "?"
        nxt_stop = next((t for t in stops if t > c), 10 ** 9)
        for c2, s2 in seq[i:]:
            if c2 >= nxt_stop:
                end = "stop:" + g["modes"][nxt_stop].rsplit("_", 1)[0]; break
            if s2 != side:
                end = "lost"; break
            last = c2
            bx, by = g["ball"][c2][:2]
            maxx = max(maxx, sg * bx)
            nb = g["ball"].get(c2 + 1)
            if nb and math.hypot((HALF if side == "l" else -HALF) - bx, by) <= 35 and heads_for_goal(*nb, side):
                shot = True
        else:
            end = "end"
        out.append((c, x0, last - c, maxx - x0, shot, end))
    return out


def main():
    pats = sys.argv[1:]
    stats = defaultdict(list); ends = defaultdict(Counter); first = Counter(); first3 = Counter()
    for pat in pats:
        for d in sorted(glob.glob(f"runlogs/trace/*_ME_vs_{pat}")):
            tag = os.path.basename(d)
            g = K.parse_rcg(f"logs/{tag}.rcg")
            me = {v: k for k, v in g["names"].items()}["ME"]
            op = "r" if me == "l" else "l"
            dec = defaultdict(list)          # cycle -> our on-ball decisions
            for f in glob.glob(f"{d}/ME/*.jsonl"):
                for line in open(f):
                    r = json.loads(line)
                    if r.get("kickable") and r.get("mode") == 2:
                        n = [x for x in r.get("notes", []) if x["note"] not in ("model", "keep_ctx")]
                        if n: dec[r["c"]].append(n[0]["note"])
            dcyc = sorted(dec)
            for who, side in (("ME", me), ("FULL", op)):
                for rec in recoveries(g, side):
                    stats[who].append(rec[:5]); ends[who][rec[5]] += 1
                    if who == "ME":
                        c, kept = rec[0], rec[2]
                        ds = [dec[t][0] for t in dcyc if c - 1 <= t <= c + max(kept, 1)][:3]
                        if ds:
                            first[ds[0]] += 1
                            first3[" > ".join(ds)] += 1
    n_m = len({d for p in pats for d in glob.glob(f"runlogs/trace/*_ME_vs_{p}")}) or 1
    for who in ("ME", "FULL"):
        a = np.array(stats[who], float)
        if not len(a): continue
        x = a[:, 1]
        print(f"\n{who}: {len(a) / n_m:.1f} recoveries/match; where (own frame x): def third {np.mean(x < -17.5):.0%},"
              f" middle {np.mean((x >= -17.5) & (x < 17.5)):.0%}, att third {np.mean(x >= 17.5):.0%}")
        print(f"   kept: median {np.median(a[:, 2]):.0f} cycles; <=5 cycles {np.mean(a[:, 2] <= 5):.0%}; >=50 cycles {np.mean(a[:, 2] >= 50):.0%}")
        print(f"   progress toward goal: median {np.median(a[:, 3]):.1f} m, mean {a[:, 3].mean():.1f} m; reached the final third"
              f" {np.mean(a[:, 1] + a[:, 3] >= 17.5):.0%}")
        print(f"   led to a shot: {a[:, 4].mean():.1%} ({a[:, 4].sum() / n_m:.2f}/match)")
        for name, s in (("def third", x < -17.5), ("middle", (x >= -17.5) & (x < 17.5)), ("att third", x >= 17.5)):
            if s.any():
                print(f"      from {name:9s} n={s.sum():4d} kept median {np.median(a[s, 2]):.0f}  progress mean {a[s, 3].mean():5.1f} m"
                      f"  shot {a[s, 4].mean():.1%}")
        print("   ended by:", dict(ends[who].most_common(6)))
    print("\nME first on-ball decision after a recovery:", dict(first.most_common(6)))
    print("ME first three decisions:", dict(first3.most_common(8)))


if __name__ == "__main__":
    main()
