"""Our tackles in traced matches: attempts, contacts, possession wins, immediate
opponent recoveries, fouls and cards, and how well the logged prediction
(p_ours, v11+) matched what happened.

usage (repo root):  python3 tools/tackle_report.py 'OPPNAME_GLOB' [...]   e.g. 'S11*'

From the replay (decision at cycle c shows at c+1):
  contact     the tackler's state at c+1 has TACKLE 0x1000 and not TACKLE_FAULT 0x2000
  possession  event label (tools/keep_data.label_event): the next touch after the
              tackle, or a restart, goes to us -> win; to them -> loss
  immediate   the opponent's touch comes within 2 cycles of the tackle
  fouls       play mode foul_charge_<our side>; cards: our players' state flags
              0x40000 / 0x80000 (only HELIOS players ever showed them so far)
"""
import glob, json, os, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
import keep_data as K

TACKLE, FAULT, CARDS = 0x1000, 0x2000, 0x40000 | 0x80000


def main():
    tot = Counter(); per = []; calib = []
    for pat in sys.argv[1:]:
        for d in sorted(glob.glob(f"runlogs/trace/*_ME_vs_{pat}")):
            tag = os.path.basename(d)
            g = K.parse_rcg(f"logs/{tag}.rcg")
            side = {v: k for k, v in g["names"].items()}["ME"]
            c_ = Counter()
            c_["fouls"] = sum(1 for m in g["modes"].values() if m == f"foul_charge_{side}")
            c_["cards"] = len({(s, u) for c, pl in g["players"].items() for s, u, f, *_ in pl
                               if s == side and f & CARDS})
            for f in glob.glob(f"{d}/ME/*.jsonl"):
                for line in open(f):
                    r = json.loads(line)
                    for n in r.get("notes", []):
                        if n["note"] == "no_tackle":
                            c_["declined"] += 1
                        if n["note"] != "tackle":
                            continue
                        c, u = r["c"], r["unum"]
                        c_["attempts"] += 1
                        st = next((fl for s, uu, fl, *_ in g["players"].get(c + 1, []) if s == side and uu == u), 0)
                        contact = bool(st & TACKLE) and not (st & FAULT)
                        c_["contacts"] += contact
                        y, why, dt = K.label_event(g, c, side, u)
                        if y == 1: c_["poss_win"] += 1
                        elif y == 0:
                            c_["poss_loss"] += 1
                            if dt <= 2: c_["immediate_opp"] += 1
                        else: c_["unknown"] += 1
                        if "p_ours" in n and contact and y is not None:
                            calib.append((n["p_ours"], y, n.get("how")))
            per.append((tag.split("_vs_")[1], c_))
            tot.update(c_)
    keys = ("attempts", "contacts", "poss_win", "poss_loss", "immediate_opp", "unknown", "declined", "fouls", "cards")
    print(f"{'match':8s} " + " ".join(f"{k:>13s}" for k in keys))
    for name, c_ in per:
        print(f"{name:8s} " + " ".join(f"{c_[k]:13d}" for k in keys))
    n = max(len(per), 1)
    print(f"{'per match':8s} " + " ".join(f"{tot[k] / n:13.2f}" for k in keys))
    if calib:
        import numpy as np
        a = np.array([(p, y) for p, y, _ in calib], float)
        print(f"\nprediction vs outcome on contacts with a known label (n={len(a)}):"
              f" mean predicted P(we keep it) {a[:, 0].mean():.2f}, observed {a[:, 1].mean():.2f}")
        for lo, hi in ((0, .3), (.3, .7), (.7, 1.01)):
            s = (a[:, 0] >= lo) & (a[:, 0] < hi)
            if s.any(): print(f"   predicted [{lo:.1f},{min(hi, 1):.1f}): n={s.sum():3d} observed win rate {a[s, 1].mean():.2f}")
        print("   by predicted route:", dict(Counter(h for _, _, h in calib)))


if __name__ == "__main__":
    main()
