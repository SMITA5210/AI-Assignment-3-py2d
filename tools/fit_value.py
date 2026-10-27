"""Fit Agent1's evaluation functions attack() and danger() to match outcomes
(L10 p43-45: the ideal evaluation is the true value of the position; L12:
MLE / MAP).

usage (repo root):  python3 tools/fit_value.py [--team ME] [--opp-pattern 'F[0-9]']

Samples, from the replays: every touch by a team, at the ball's position.
  attack: our touch -> 1 if this possession of ours ends in our goal
  danger: their touch -> 1 if that possession of theirs ends in their goal
Both are Bernoulli. The agent's feature forms are kept:
  attack(x, y) = A0 * (1 - d / (2 hl)) + A1 * exp(-d / LA) * min(1, angle)
                 d = distance to their goal, angle = goal mouth seen from (x, y)
  danger(x, y) = D0 * (hl - x) / (2 hl) + D1 * exp(-d' / LD)
                 d' = distance to our goal
and their weights fitted by maximising log-likelihood + log-prior, the prior
being k pseudo-observations drawn from the current hand-set curve
(pseudo-counts, L12 p27; MAP p15). k is chosen on held-out matches.
Samples are binned on a 1 m grid first (same likelihood, far fewer terms).
"""
import argparse, glob, math, re
from collections import defaultdict

import numpy as np

TOUCH = 0x2 | 0x10 | 0x1000
HL, HW, GOAL_HALF = 52.5, 34.0, 7.01
HAND = {"attack": (0.25, 0.6, 15.0), "danger": (0.3, 0.6, 14.0)}
K_CHOICES = (0.0, 200.0, 800.0, 3200.0, 12800.0)
W0 = np.linspace(0.0, 0.5, 51)
W1 = np.linspace(0.0, 2.0, 81)
LS = np.arange(4.0, 51.0, 1.0)


def features(kind, x, y, L):
    if kind == "attack":
        dx = HL - x
        d = np.hypot(dx, y)
        ang = np.abs(np.arctan2(GOAL_HALF - y, dx) - np.arctan2(-GOAL_HALF - y, dx))
        return 1.0 - d / (2 * HL), np.exp(-d / L) * np.minimum(1.0, ang)
    d = np.hypot(x + HL, y)
    return (HL - x) / (2 * HL), np.exp(-d / L)


def samples(paths, team):
    """{kind: {match: [(x, y, outcome)]}} in the team's frame (attack +x)."""
    out = {"attack": defaultdict(list), "danger": defaultdict(list)}
    for path in paths:
        names, touches, goals, ball = {}, [], {}, {}
        for line in open(path, errors="replace"):
            if line.startswith("(team"):
                p = line.split(); names = {"l": p[2], "r": p[3]}
            elif line.startswith("(playmode"):
                m = re.match(r"\(playmode (\d+) goal_(l|r)\)", line)
                if m: goals[int(m.group(1))] = m.group(2)
            elif line.startswith("(show"):
                c = int(line.split()[1])
                b = re.search(r"\(\(b\) (\S+) (\S+)", line)
                ball[c] = (float(b.group(1)), float(b.group(2)))
                for side, _, st in re.findall(r"\(\((l|r) (\d+)\) \d+ (0x[0-9a-f]+)", line):
                    if int(st, 16) & TOUCH: touches.append((c, side))
        me = {v: k for k, v in names.items()}.get(team)
        if me is None: continue
        sg = 1.0 if me == "l" else -1.0
        runs, cur = [], None
        for c, s in touches:
            if cur and cur[0] == s: cur[2] = c
            else:
                if cur: runs.append(cur + [c])
                cur = [s, c, c]
        if cur: runs.append(cur + [10 ** 9])
        for s, a, b, nxt in runs:
            y = int(any(goals[g] == s for g in goals if a <= g <= nxt))
            kind = "attack" if s == me else "danger"
            for c, s2 in touches:
                if s2 == s and a <= c <= b:
                    out[kind][path].append((sg * ball[c][0], sg * ball[c][1], y))
    return out


def binned(rows):
    cells = defaultdict(lambda: [0, 0])
    for x, y, o in rows:
        e = cells[(round(x), round(y))]; e[0] += 1; e[1] += o
    xy = np.array(list(cells.keys()), float).reshape(-1, 2)
    nk = np.array(list(cells.values()), float).reshape(-1, 2)
    return xy[:, 0], xy[:, 1], nk[:, 0], nk[:, 1]


def loglik_grid(kind, x, y, n, k, L):
    """(len(W0), len(W1)) log-likelihoods for one length scale."""
    f0, f1 = features(kind, x, y, L)
    p = W0[:, None, None] * f0 + W1[None, :, None] * f1
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return (k * np.log(p) + (n - k) * np.log(1 - p)).sum(axis=-1)


def prior_points(kind, strength):
    """Pseudo-observations from the hand curve on a coarse pitch grid."""
    gx, gy = np.meshgrid(np.arange(-50, 51, 5.0), np.arange(-30, 31, 5.0))
    x, y = gx.ravel(), gy.ravel()
    a0, a1, L = HAND[kind]
    f0, f1 = features(kind, x, y, L)
    p = np.clip(a0 * f0 + a1 * f1, 1e-4, 1 - 1e-4)
    n = np.full(len(x), strength / len(x))
    return x, y, n, n * p


def fit(kind, data, strength):
    px, py, pn, pk = prior_points(kind, strength)
    x, y, n, k = (np.concatenate([a, b]) for a, b in zip(data, (px, py, pn, pk)))
    best = (-np.inf, None)
    for L in LS:
        ll = loglik_grid(kind, x, y, n, k, L)
        i, j = np.unravel_index(int(np.argmax(ll)), ll.shape)
        if ll[i, j] > best[0]: best = (ll[i, j], (W0[i], W1[j], L))
    return best[1]


def heldout_ll(kind, data, w):
    x, y, n, k = data
    f0, f1 = features(kind, x, y, w[2])
    p = np.clip(w[0] * f0 + w[1] * f1, 1e-4, 1 - 1e-4)
    return float((k * np.log(p) + (n - k) * np.log(1 - p)).sum() / max(n.sum(), 1))


def table(kind, rows, w):
    """Observed vs predicted by distance band (to their goal for attack, ours for danger)."""
    x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); o = np.array([r[2] for r in rows])
    d = np.hypot(HL - x, y) if kind == "attack" else np.hypot(x + HL, y)
    hand, fitted = (np.clip(a * f0 + b * f1, 0, 1) for a, b, (f0, f1) in
                    ((HAND[kind][0], HAND[kind][1], features(kind, x, y, HAND[kind][2])),
                     (w[0], w[1], features(kind, x, y, w[2]))))
    for lo in range(0, 65, 10):
        s = (d >= lo) & (d < lo + 10)
        if s.any():
            print(f"      {lo:2d}-{lo + 10:<3d} n={s.sum():6d} observed {o[s].mean():.3f}"
                  f"  hand {hand[s].mean():.3f}  fitted {fitted[s].mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="ME")
    ap.add_argument("--opp-pattern", default="F[0-9]", help="glob for opponent names, e.g. 'F[0-9]' = runs vs full")
    a = ap.parse_args()
    paths = sorted(glob.glob(f"logs/*_{a.team}_vs_*{a.opp_pattern}*.rcg"))
    data = samples(paths, a.team)
    print(f"{len(paths)} matches")
    for kind in ("attack", "danger"):
        by = data[kind]
        ms = sorted(by)
        test = set(ms[1::2])
        tr = binned([r for m in ms if m not in test for r in by[m]])
        te = binned([r for m in test for r in by[m]])
        res = [(heldout_ll(kind, te, fit(kind, tr, s)), s) for s in K_CHOICES]
        ll_best, s_best = max(res)
        print(f"\n{kind}: {sum(len(by[m]) for m in ms)} touches")
        print(f"   held-out log-lik per sample: hand {heldout_ll(kind, te, HAND[kind]):.4f}"
              f"  fitted {ll_best:.4f} (prior strength {s_best:g})")
        w = fit(kind, binned([r for m in ms for r in by[m]]), s_best)
        print(f"   hand   {HAND[kind]}")
        print(f"   fitted ({w[0]:.3f}, {w[1]:.3f}, {w[2]:.1f})")
        table(kind, [r for m in ms for r in by[m]], w)


if __name__ == "__main__":
    main()
