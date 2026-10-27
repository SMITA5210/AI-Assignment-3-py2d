"""Fit Agent1's success-probability curves to match data (L12: MLE / MAP).

usage (repo root):  python3 tools/fit_probs.py [--last N] [--team ME]

The agent models   P(success) = sigmoid((m - OFFSET) / SCALE) * f
where m is the opponent margin in cycles and f the other, physics-derived
factors; both are logged in every pass/dribble/shoot/hold note. This script

  1. builds a dataset (kind, m, f, outcome) from the traces, with outcomes
     read from the replay: who touches the ball next (KICK 0x2, TACKLE
     0x1000, CATCH 0x10, ignoring the decider's own touches) and restarts
     awarded to either side; a shot succeeds when we score;
  2. for each kind separately (with complete data the likelihood
     decomposes into one problem per parameter, L12 p28-29) maximises
         log-likelihood + log-prior,
     the prior being k pseudo-observations drawn from the current
     hand-set curve (Laplace smoothing as pseudo-counts, L12 p27; MAP p15);
  3. picks k on held-out matches (overfitting, L12 p26) and reports the
     held-out log-likelihood of the hand-set and the fitted curves;
  4. refits on all matches and prints the constants to put in agent.py.

The grid search over (OFFSET, SCALE) is our engineering choice: a sigmoid
has no closed-form maximum like the coin's relative counts.
"""
import argparse, glob, json, math, os, re, sys
from collections import defaultdict

import numpy as np

TOUCH = 0x2 | 0x1000 | 0x10
HAND = {"pass": (0.5, 1.0), "shoot": (0.5, 1.0), "dribble": (0.0, 0.7), "hold": (1.0, 1.0)}
WINDOW = {"dribble": 8, "hold": 3, "shoot": 30}
PRIOR_GRID = np.arange(-3.0, 8.5, 1.0)        # margins the pseudo-counts sit at
K_CHOICES = (2.0, 8.0, 32.0, 128.0)


def replay_events(rcg, team):
    names, touches, restarts, goals, cyc = {}, defaultdict(list), {}, {}, 0
    for line in open(rcg, errors="replace"):
        if line.startswith("(team"):
            p = line.split(); names = {"l": p[2], "r": p[3]}
        elif line.startswith("(playmode"):
            m = re.match(r"\(playmode (\d+) (\S+?)_(l|r)\)", line)
            if not m or not names: continue        # kick-off precedes the team line
            c, mode, side = int(m.group(1)), m.group(2), names[m.group(3)] == team
            if mode in ("kick_in", "free_kick", "corner_kick", "goal_kick",
                        "indirect_free_kick", "goalie_catch_ball"):
                restarts.setdefault(c, side)
            elif mode == "goal":
                goals[c] = side
        elif line.startswith("(show"):
            cyc = int(line.split()[1])
            for side, unum, st in re.findall(r"\(\((l|r) (\d+)\) \d+ (0x[0-9a-f]+)", line):
                if int(st, 16) & TOUCH:
                    touches[cyc].append((names[side] == team, int(unum)))
    return touches, restarts, goals


def outcome(kind, c, me, n, touches, restarts, goals):
    """1 success, 0 failure, None unknown."""
    if kind == "shoot":
        for c2 in range(c + 1, c + WINDOW["shoot"] + 1):
            if c2 in goals: return 1 if goals[c2] else 0
            if c2 in restarts: return 0
            if any(not ours for ours, _ in touches.get(c2, [])): return 0
        return 0
    win = int(n) + 5 if kind == "pass" else WINDOW[kind]
    for c2 in range(c + 1, c + win + 1):
        if c2 in restarts: return 1 if restarts[c2] else 0
        others = [t for t in touches.get(c2, []) if t != (True, me)]
        if any(not ours for ours, _ in others): return 0
        if others: return 1
    return None if kind == "pass" else 1


def dataset(dirs, team):
    rows = defaultdict(list)          # kind -> [(match, m, f, y)]
    for d in dirs:
        tag = os.path.basename(d)
        rcg = os.path.join("logs", tag + ".rcg")
        if not os.path.exists(rcg): continue
        touches, restarts, goals = replay_events(rcg, team)
        for p in glob.glob(os.path.join(d, team, "*.jsonl")):
            for line in open(p):
                r = json.loads(line)
                if not r["kickable"]: continue
                for n in r.get("notes", []):
                    kind = n["note"]
                    if kind not in HAND or "m" not in n: continue
                    y = outcome(kind, r["c"], r["unum"], n.get("n", 0), touches, restarts, goals)
                    if y is None: continue
                    rows[kind].append((tag, float(n["m"]), float(n.get("f", 1.0)), y))
    return rows


def prob(m, f, off, scale):
    return 1.0 / (1.0 + np.exp(-np.clip((m - off) / scale, -30, 30))) * f


def loglik(m, f, y, w, off, scale):
    p = np.clip(prob(m, f, off, scale), 1e-6, 1 - 1e-6)
    return float(np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def fit(m, f, y, kind, k):
    """MAP by grid search: data plus k pseudo-observations from the hand curve."""
    off0, s0 = HAND[kind]
    pm = PRIOR_GRID
    py = prob(pm, 1.0, off0, s0)                    # fractional pseudo-successes
    pw = np.full(len(pm), k / len(pm))
    M = np.concatenate([m, pm, pm]); F = np.concatenate([f, np.ones(2 * len(pm))])
    Y = np.concatenate([y, np.ones(len(pm)), np.zeros(len(pm))])
    W = np.concatenate([np.ones(len(m)), pw * py, pw * (1 - py)])
    best = (-np.inf, off0, s0)
    for off in np.linspace(-4.0, 6.0, 101):
        for scale in np.exp(np.linspace(math.log(0.15), math.log(5.0), 61)):
            ll = loglik(M, F, Y, W, off, scale)
            if ll > best[0]: best = (ll, off, scale)
    return best[1], best[2]


def calib_table(m, f, y, off, scale):
    p = prob(m, f, off, scale)
    out = []
    for lo in np.arange(0, 1, 0.2):
        sel = (p >= lo) & (p < lo + 0.2 + (1e-9 if lo >= 0.8 else 0))
        if sel.any(): out.append(f"{lo:.1f}-{lo + 0.2:.1f}: pred {p[sel].mean():.2f} obs {y[sel].mean():.2f} n={sel.sum()}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=40, help="use the N most recent traced matches")
    ap.add_argument("--team", default="ME")
    a = ap.parse_args()
    dirs = sorted(glob.glob(f"runlogs/trace/*_{a.team}_vs_*"), key=os.path.getmtime)[-a.last:]
    rows = dataset(dirs, a.team)
    matches = sorted({r[0] for k in rows for r in rows[k]})
    test = set(matches[1::2])                       # alternate matches held out
    print(f"{len(matches)} matches with logged margins ({len(test)} held out)\n")
    for kind in ("pass", "dribble", "hold", "shoot"):
        data = rows.get(kind, [])
        if not data:
            print(f"{kind}: no data\n"); continue
        arr = lambda sel: tuple(np.array([r[i] for r in data if sel(r)]) for i in (1, 2, 3))
        mtr, ftr, ytr = arr(lambda r: r[0] not in test)
        mte, fte, yte = arr(lambda r: r[0] in test)
        off0, s0 = HAND[kind]
        print(f"{kind}: n={len(data)}  successes={sum(r[3] for r in data)}")
        if len(mte) and len(mtr):
            res = []
            for k in K_CHOICES:
                off, s = fit(mtr, ftr, ytr, kind, k)
                res.append((loglik(mte, fte, yte, 1.0, off, s), k, off, s))
            ll_fit, k_best, _, _ = max(res)
            ll_hand = loglik(mte, fte, yte, 1.0, off0, s0)
            print(f"   held-out log-likelihood per sample: hand {ll_hand / len(mte):.4f}"
                  f"   fitted {ll_fit / len(mte):.4f}   (prior strength k={k_best:g})")
        else:
            k_best = K_CHOICES[-1]
        mall, fall, yall = arr(lambda r: True)
        off, s = fit(mall, fall, yall, kind, k_best)
        print(f"   hand   OFFSET={off0:.2f} SCALE={s0:.2f}")
        print(f"   fitted OFFSET={off:.2f} SCALE={s:.2f}   (all {len(mall)} samples, k={k_best:g})")
        for line in calib_table(mall, fall, yall, off0, s0): print("      hand   " + line)
        for line in calib_table(mall, fall, yall, off, s): print("      fitted " + line)
        print()


if __name__ == "__main__":
    main()
