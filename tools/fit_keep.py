"""Refit Agent1's hold-success curve on the collector data (L12: MLE / MAP).

usage (repo root):  python3 tools/fit_keep.py [runlogs/keep/kc_rows.jsonl]

Data: rows from tools/keep_data.py (collector matches KC1-KC20 vs full), the
executed action "hold", event-based labels (1 kept, 0 lost; unknown dropped).

Baseline (v3/v8b):  P(keep) = sig((m_cap - OFF) / SC),  OFF=-1.1, SC=0.46
    m_cap = max(0, u) - 1,  u = (d1 - tackle_dist) / speed   (logged: umargin)
    d1 = distance of the nearest opponent to the ball. The cap makes every
    opponent inside tackle range look the same.
Candidate 1 (same two parameters, uncapped feature):
    P(keep) = sig((u - 1 - OFF) / SC)
Candidate 2 (one more parameter, only if inner validation supports it):
    P(keep) = sig((u - 1 - OFF) / SC - B * (n_press - 1)),  n_press = opponents
    within 2 * tackle_dist of the ball.
Fit: MAP by grid search, the prior being k pseudo-observations from the
baseline curve on a grid of u (pseudo-counts, L12 p27; MAP p15). The feature
set and k are chosen by leave-one-match-out cross-validation inside the
training matches only. The validation matches are scored once, at the end.

Split (fixed before looking at results): odd-numbered KC matches train, even
ones validate. Metrics are per decision; probabilities are clipped to
[1e-6, 1 - 1e-6] for the log loss. Uncertainty: match-cluster bootstrap over
validation matches, B_REPS replicates, seed BOOT_SEED.

Predeclared acceptance gates (fixed before the first run):
  G1 validation log loss improves by >= 2 % relative to the baseline
  G2 validation Brier score worsens by no more than 1 % relative
  G3 close-pressure (d1 <= tackle_dist) calibration error |obs - pred| shrinks,
     and in the 2-4 m and > 4 m bins it grows by no more than MATERIAL (abs.)
  G4 no single match decides it: the log-loss improvement keeps its sign when
     any one validation match is left out
  G5 the 90 % bootstrap interval of the log-loss difference excludes zero
Decision: PASS if G1-G5 all hold; FAIL if any of G1-G4 fails on the point
estimates; otherwise INCONCLUSIVE.
"""
import json, math, re, sys
from collections import defaultdict

import numpy as np

BASE_OFF, BASE_SC = -1.1, 0.46
TACKLE, PRESS = 2.0, 4.0            # tackle_dist and 2 * tackle_dist (server params of these matches)
K_CHOICES = (2.0, 8.0, 32.0, 128.0)
PRIOR_U = np.arange(-2.0, 6.01, 0.5)
OFFS = np.arange(-4.0, 4.001, 0.05)
SCS = np.exp(np.linspace(math.log(0.1), math.log(5.0), 61))
BS = np.arange(0.0, 3.01, 0.1)
B_REPS, BOOT_SEED = 2000, 12345
MATERIAL = 0.02
EPS = 1e-6


def load(path):
    rows = [json.loads(l) for l in open(path)]
    out = []
    for r in rows:
        if r["kind"] != "hold": continue
        num = int(re.search(r"_vs_KC(\d+)$", r["match"]).group(1))
        out.append({"match": r["match"], "num": num, "u": r["umargin"], "d1": r["d1"],
                    "n_press": r["n_press"], "y": r["label"], "m_cap": r["model_m"],
                    "p_logged": r["model_p"]})
    return out


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def p_base(d):
    return sig((d["m_cap"] - BASE_OFF) / BASE_SC)


def p_cand(d, w):
    off, sc, b = w
    return sig((d["u"] - 1.0 - off) / sc - b * (d["n_press"] - 1.0))


def arr(rows):
    return {k: np.array([r[k] for r in rows], float) for k in ("u", "d1", "n_press", "y", "m_cap")}


def fit(d, k, with_b):
    """MAP grid search: data + k pseudo-observations from the baseline curve
    (pseudo rows have n_press = 1, so B is anchored by the data only)."""
    pu = PRIOR_U
    pp = sig((np.maximum(0.0, pu) - 1.0 - BASE_OFF) / BASE_SC)
    w = k / len(pu)
    U = np.concatenate([d["u"], pu, pu]); N = np.concatenate([d["n_press"], np.ones(2 * len(pu))])
    Y = np.concatenate([d["y"], np.ones(len(pu)), np.zeros(len(pu))])
    W = np.concatenate([np.ones(len(d["u"])), w * pp, w * (1 - pp)])
    best = (-np.inf, None)
    for b in (BS if with_b else (0.0,)):
        z = (U[None, None, :] - 1.0 - OFFS[:, None, None]) / SCS[None, :, None] - b * (N - 1.0)
        p = np.clip(sig(z), EPS, 1 - EPS)
        ll = (W * (Y * np.log(p) + (1 - Y) * np.log(1 - p))).sum(axis=-1)
        i, j = np.unravel_index(int(np.argmax(ll)), ll.shape)
        if ll[i, j] > best[0]: best = (ll[i, j], (float(OFFS[i]), float(SCS[j]), float(b)))
    return best[1]


def logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def bins(d):
    return {"<=2m": d["d1"] <= TACKLE, "2-4m": (d["d1"] > TACKLE) & (d["d1"] <= PRESS),
            ">4m": d["d1"] > PRESS}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runlogs/keep/kc_rows.jsonl"
    allrows = load(path)
    tr_rows = [r for r in allrows if r["num"] % 2 == 1]
    va_rows = [r for r in allrows if r["num"] % 2 == 0]

    # ---- Table A
    print("TABLE A - dataset and split (executed holds; labels event-based)")
    print(f"  {'partition':10s} {'matches':>7s} {'holds':>6s} {'kept':>5s} {'lost':>5s} {'unknown':>7s} {'<=2m':>5s} {'2-4m':>5s} {'>4m':>5s}")
    for name, rs in (("train", tr_rows), ("validation", va_rows), ("all", allrows)):
        lab = [r for r in rs if r["y"] is not None]
        print(f"  {name:10s} {len({r['match'] for r in rs}):7d} {len(rs):6d} {sum(r['y'] == 1 for r in lab):5d}"
              f" {sum(r['y'] == 0 for r in lab):5d} {len(rs) - len(lab):7d}"
              f" {sum(r['d1'] <= TACKLE for r in lab):5d} {sum(TACKLE < r['d1'] <= PRESS for r in lab):5d}"
              f" {sum(r['d1'] > PRESS for r in lab):5d}")
    for name, rs in (("train", tr_rows), ("validation", va_rows)):
        print(f"  {name} matches: " + " ".join(sorted({r['match'].split('_vs_')[1] for r in rs},
                                                      key=lambda s: int(s[2:]))))
        lab = [r for r in rs if r["y"] is not None]
        close = [r for r in lab if r["d1"] <= TACKLE]
        print(f"     <=2m: {len(close)} holds, {sum(r['y'] == 0 for r in close)} lost, from "
              f"{len({r['match'] for r in close})} matches; u range "
              f"{min(r['u'] for r in close):.2f}..{max(r['u'] for r in close):.2f};"
              f" n_press distribution {dict(sorted(__import__('collections').Counter(r['n_press'] for r in close).items()))}")

    tr = [r for r in tr_rows if r["y"] is not None]
    va = [r for r in va_rows if r["y"] is not None]

    # ---- model selection inside training matches (leave one match out)
    tmatches = sorted({r["match"] for r in tr})
    cv = {}
    for with_b in (False, True):
        for k in K_CHOICES:
            ll = []
            for m in tmatches:
                fit_d = arr([r for r in tr if r["match"] != m])
                hold_d = arr([r for r in tr if r["match"] == m])
                w = fit(fit_d, k, with_b)
                p = np.clip(p_cand(hold_d, w), EPS, 1 - EPS)
                ll.append(-(hold_d["y"] * np.log(p) + (1 - hold_d["y"]) * np.log(1 - p)).sum())
            cv[(with_b, k)] = sum(ll) / len(tr)
    trd = arr(tr)
    base_cv = logloss(trd["y"], p_base(trd))
    print("\nModel selection (leave-one-match-out over the 10 training matches; mean log loss per decision)")
    print(f"  baseline (no fitting) {base_cv:.4f}")
    for (with_b, k), v in sorted(cv.items()):
        print(f"  {'cand2 (+n_press)' if with_b else 'cand1 (uncapped)':18s} k={k:<5g} {v:.4f}")
    best1 = min((v, k) for (b, k), v in cv.items() if not b)
    best2 = min((v, k) for (b, k), v in cv.items() if b)
    # the extra parameter must earn its place: >= 2 % better inner-CV log loss
    use_b = best2[0] <= 0.98 * best1[0]
    k_sel = best2[1] if use_b else best1[1]
    w = fit(trd, k_sel, use_b)
    print(f"  selected: {'cand2' if use_b else 'cand1'} with k={k_sel:g}"
          f" -> OFF={w[0]:.2f} SC={w[1]:.3f} B={w[2]:.2f}  (fitted on the 10 training matches)")

    # ---- Table B (validation, scored once)
    vd = arr(va)
    pb, pc = p_base(vd), p_cand(vd, w)
    print("\nTABLE B - hold model on the 10 validation matches (per decision)")
    print(f"  {'':10s} {'logloss':>8s} {'Brier':>7s} {'obs loss':>8s} {'pred loss':>9s}")
    for name, p in (("baseline", pb), ("candidate", pc)):
        print(f"  {name:10s} {logloss(vd['y'], p):8.4f} {brier(vd['y'], p):7.4f}"
              f" {1 - vd['y'].mean():8.3f} {1 - p.mean():9.3f}")
    print("  calibration by pressure bin (loss rates):")
    print(f"  {'bin':6s} {'n':>5s} {'lost':>5s} {'obs':>6s} {'base':>6s} {'cand':>6s} {'|err| base':>10s} {'|err| cand':>10s}")
    calib = {}
    for name, s in bins(vd).items():
        obs = 1 - vd["y"][s].mean() if s.any() else float("nan")
        b_, c_ = 1 - pb[s].mean(), 1 - pc[s].mean()
        calib[name] = (abs(obs - b_), abs(obs - c_))
        print(f"  {name:6s} {s.sum():5d} {int((vd['y'][s] == 0).sum()):5d} {obs:6.3f} {b_:6.3f} {c_:6.3f}"
              f" {calib[name][0]:10.3f} {calib[name][1]:10.3f}")
    match_w = {}
    for name, p in (("baseline", pb), ("candidate", pc)):
        per = [logloss(vd["y"][np.array([r["match"] == m for r in va])],
                       p[np.array([r["match"] == m for r in va])]) for m in sorted({r["match"] for r in va})]
        match_w[name] = float(np.mean(per))
    print(f"  match-weighted log loss (secondary): baseline {match_w['baseline']:.4f} candidate {match_w['candidate']:.4f}")

    # ---- Table C (match-cluster bootstrap)
    vmatches = sorted({r["match"] for r in va})
    idx = {m: np.array([i for i, r in enumerate(va) if r["match"] == m]) for m in vmatches}
    rng = np.random.default_rng(BOOT_SEED)
    rep = defaultdict(list)
    for _ in range(B_REPS):
        pick = np.concatenate([idx[m] for m in rng.choice(vmatches, len(vmatches))])
        y = vd["y"][pick]
        lb, lc = logloss(y, pb[pick]), logloss(y, pc[pick])
        bb, bc = brier(y, pb[pick]), brier(y, pc[pick])
        rep["ll_base"].append(lb); rep["ll_cand"].append(lc); rep["ll_diff"].append(lc - lb)
        rep["ll_rel"].append((lc - lb) / lb)
        rep["br_base"].append(bb); rep["br_cand"].append(bc); rep["br_diff"].append(bc - bb)
        cs = vd["d1"][pick] <= TACKLE
        if cs.any():
            o = 1 - y[cs].mean()
            rep["close_err_base"].append(abs(o - (1 - pb[pick][cs].mean())))
            rep["close_err_cand"].append(abs(o - (1 - pc[pick][cs].mean())))
    print(f"\nTABLE C - match-cluster bootstrap over the {len(vmatches)} validation matches,"
          f" {B_REPS} replicates, seed {BOOT_SEED}: 90% intervals")
    for k, v in rep.items():
        lo, hi = np.percentile(v, [5, 95])
        print(f"  {k:16s} {np.mean(v):8.4f}   [{lo:8.4f}, {hi:8.4f}]")

    # ---- per-match and leave-one-out
    print("\n  per validation match: n, lost, log loss base -> cand")
    loo = []
    for m in vmatches:
        s = np.array([r["match"] == m for r in va])
        print(f"   {m.split('_vs_')[1]:5s} n={s.sum():3d} lost={int((vd['y'][s] == 0).sum()):2d}"
              f"  {logloss(vd['y'][s], pb[s]):.4f} -> {logloss(vd['y'][s], pc[s]):.4f}")
        rest = ~s
        loo.append(logloss(vd["y"][rest], pc[rest]) - logloss(vd["y"][rest], pb[rest]))

    # ---- gates
    llb, llc = logloss(vd["y"], pb), logloss(vd["y"], pc)
    brb, brc = brier(vd["y"], pb), brier(vd["y"], pc)
    g = {
        "G1 log loss -2% rel": (llc - llb) / llb <= -0.02,
        "G2 Brier <= +1% rel": (brc - brb) / brb <= 0.01,
        "G3 calibration": (calib["<=2m"][1] < calib["<=2m"][0]
                           and calib["2-4m"][1] - calib["2-4m"][0] <= MATERIAL
                           and calib[">4m"][1] - calib[">4m"][0] <= MATERIAL),
        "G4 leave-one-match-out sign": all(x < 0 for x in loo),
        "G5 bootstrap CI excludes 0": np.percentile(rep["ll_diff"], 95) < 0,
    }
    print(f"\nGATES  (log loss {llb:.4f} -> {llc:.4f}, {100 * (llc - llb) / llb:+.1f}%;"
          f" Brier {brb:.4f} -> {brc:.4f}, {100 * (brc - brb) / brb:+.1f}%;"
          f" leave-one-out diffs {min(loo):+.4f}..{max(loo):+.4f})")
    for k, v in g.items():
        print(f"  {k:30s} {'PASS' if v else 'FAIL'}")
    hard = [g[k] for k in list(g)[:4]]
    decision = "PASS" if all(g.values()) else ("FAIL" if not all(hard) else "INCONCLUSIVE")
    print(f"DECISION: {decision}")
    print(f"MODEL: OFF={w[0]:.2f} SC={w[1]:.3f} B={w[2]:.2f} use_n_press={use_b} k={k_sel:g}")


if __name__ == "__main__":
    main()
