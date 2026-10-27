"""Refit Agent1's shot-success curve (L12: MLE / MAP) on traced matches vs full.

usage (repo root):  python3 tools/fit_shots.py

Data: every "shoot" decision in PlayOn (trace note, with the logged margin m
and other-factor f) in the traced matches against full; outcome 1 if we score
within 40 cycles with no opponent touch before it (replay, touches counted as
in tools/keep_data.py). Model, as in the agent:
    P(goal) = sigmoid((m - SHOT_OFFSET) / SHOT_SCALE) * f
Baseline: the hand-set prior (0.5, 1.0), never fitted before. Fit: MAP by
grid search with k pseudo-observations from the baseline curve on a grid of m
(pseudo-counts, L12 p27); k chosen by leave-one-match-group-out validation
inside the training matches.

Split, fixed before looking: matches sorted by name, every second one held out.
Predeclared gates on the held-out matches (fixed before the first run):
  G1 log loss improves by >= 5 % relative to the baseline curve
  G2 Brier score does not get worse
  G3 the 12-25 m bin's |observed - predicted| shrinks
  G4 the result keeps its sign when any one held-out match is left out
Decision: ACCEPT only if all hold; otherwise keep the baseline curve.
Caveat: only shots the agent chose are observed (selection bias); the
evaluation in matches, not this fit, decides whether shooting more helps.
"""
import glob, json, math, os, re, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import keep_data as K

BASE = (0.5, 1.0)
FULL_NAMES = re.compile(r"_ME_vs_(V\dB?F\d|TKF|E3P|E3V_|EAP|EBP|ECP|E11_|C11_|C3_|T11_|T12_|B11_|B13_"
                        r"|KC\d|SMV|SMX|SMK|S11|S12_|S13_|RUN|RST|PRM|P3F)")
OFFS = np.arange(-4.0, 6.001, 0.05)
SCS = np.exp(np.linspace(math.log(0.15), math.log(6.0), 61))
PRIOR_M = np.arange(-4.0, 8.01, 1.0)
K_CHOICES = (2.0, 8.0, 32.0, 128.0)
EPS = 1e-6


def data():
    rows = []
    for d in sorted(glob.glob("runlogs/trace/*_ME_vs_*")):
        if not FULL_NAMES.search(d): continue
        tag = os.path.basename(d); rcg = f"logs/{tag}.rcg"
        if not os.path.exists(rcg): continue
        g = K.parse_rcg(rcg)
        side = {v: k for k, v in g["names"].items()}.get("ME")
        if side is None: continue
        goals = sorted(c for c, m in g["modes"].items() if m == f"goal_{side}")
        sg = 1.0 if side == "l" else -1.0
        for f in glob.glob(f"{d}/ME/*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                if not r.get("kickable") or r.get("mode") != 2: continue
                for n in r.get("notes", []):
                    if n["note"] != "shoot" or "m" not in n: continue
                    c = r["c"]
                    gl = next((t for t in goals if c < t <= c + 40), None)
                    opp = any(s != side for t in range(c + 1, (gl or c + 40) + 1) for s, _ in g["touch"].get(t, []))
                    rows.append({"match": tag, "m": float(n["m"]), "f": float(n.get("f", 1.0)),
                                 "y": 1 if (gl and not opp) else 0,
                                 "dist": math.hypot(52.5 - r["ball"][0], r["ball"][1])})
    return rows


def prob(m, f, off, sc):
    return 1.0 / (1.0 + np.exp(-np.clip((m - off) / sc, -30, 30))) * f


def ll(rows, off, sc):
    m = np.array([r["m"] for r in rows]); f = np.array([r["f"] for r in rows]); y = np.array([r["y"] for r in rows])
    p = np.clip(prob(m, f, off, sc), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(rows, off, sc):
    m = np.array([r["m"] for r in rows]); f = np.array([r["f"] for r in rows]); y = np.array([r["y"] for r in rows])
    return float(np.mean((prob(m, f, off, sc) - y) ** 2))


def fit(rows, k):
    m = np.array([r["m"] for r in rows]); f = np.array([r["f"] for r in rows]); y = np.array([r["y"] for r in rows], float)
    pp = prob(PRIOR_M, 1.0, *BASE); w = k / len(PRIOR_M)
    M = np.concatenate([m, PRIOR_M, PRIOR_M]); F = np.concatenate([f, np.ones(2 * len(PRIOR_M))])
    Y = np.concatenate([y, np.ones(len(PRIOR_M)), np.zeros(len(PRIOR_M))])
    W = np.concatenate([np.ones(len(m)), w * pp, w * (1 - pp)])
    p = np.clip(prob(M[None, None, :], F[None, None, :], OFFS[:, None, None], SCS[None, :, None]), EPS, 1 - EPS)
    L = (W * (Y * np.log(p) + (1 - Y) * np.log(1 - p))).sum(axis=-1)
    i, j = np.unravel_index(int(np.argmax(L)), L.shape)
    return float(OFFS[i]), float(SCS[j])


def table(rows, params, label):
    for lo, hi in ((0, 12), (12, 25), (25, 40)):
        s = [r for r in rows if lo <= r["dist"] < hi]
        if not s: continue
        obs = np.mean([r["y"] for r in s])
        pr = np.mean([prob(r["m"], r["f"], *params) for r in s])
        print(f"      {label:9s} {lo:2d}-{hi:<2d} m  n={len(s):3d} observed {obs:.2f} predicted {pr:.2f}  |err| {abs(obs - pr):.2f}")


def main():
    rows = data()
    matches = sorted({r["match"] for r in rows})
    test = set(matches[1::2])
    tr = [r for r in rows if r["match"] not in test]; te = [r for r in rows if r["match"] in test]
    print(f"{len(rows)} shoot decisions in {len(matches)} matches with shots; train {len(tr)}, held out {len(te)}"
          f" (goals {sum(r['y'] for r in tr)} / {sum(r['y'] for r in te)})")
    # choose k inside training: two halves of the training matches
    trm = sorted({r["match"] for r in tr}); fold = {mm: i % 2 for i, mm in enumerate(trm)}
    cv = {}
    for k in K_CHOICES:
        s = 0.0
        for fo in (0, 1):
            a = [r for r in tr if fold[r["match"]] != fo]; b = [r for r in tr if fold[r["match"]] == fo]
            s += ll(b, *fit(a, k)) * len(b)
        cv[k] = s / len(tr)
    k = min(cv, key=cv.get)
    params = fit(tr, k)
    print(f"prior strength by inner validation: {cv} -> k={k}")
    print(f"fitted on training matches: SHOT_OFFSET={params[0]:.2f} SHOT_SCALE={params[1]:.3f}")
    llb, llf = ll(te, *BASE), ll(te, *params)
    brb, brf = brier(te, *BASE), brier(te, *params)
    print(f"held out: log loss {llb:.4f} -> {llf:.4f} ({100 * (llf - llb) / llb:+.1f}%), Brier {brb:.4f} -> {brf:.4f}")
    table(te, BASE, "baseline"); table(te, params, "fitted")
    loo = []
    for mm in sorted(test):
        rest = [r for r in te if r["match"] != mm]
        loo.append(ll(rest, *params) - ll(rest, *BASE))
    e = lambda ps, lo, hi: abs(np.mean([r["y"] for r in te if lo <= r["dist"] < hi])
                               - np.mean([prob(r["m"], r["f"], *ps) for r in te if lo <= r["dist"] < hi]))
    gates = {"G1 log loss -5%": (llf - llb) / llb <= -0.05, "G2 Brier not worse": brf <= brb,
             "G3 12-25 m calibration": e(params, 12, 25) < e(BASE, 12, 25),
             "G4 leave-one-match-out sign": all(x < 0 for x in loo)}
    for g_, v in gates.items(): print(f"  {g_:28s} {'PASS' if v else 'FAIL'}")
    print("DECISION:", "ACCEPT" if all(gates.values()) else "KEEP BASELINE")
    allp = fit(rows, k)
    print(f"refit on all {len(rows)} (k={k}): SHOT_OFFSET={allp[0]:.2f} SHOT_SCALE={allp[1]:.3f}")


if __name__ == "__main__":
    main()
