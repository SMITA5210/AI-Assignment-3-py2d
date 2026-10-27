"""Fit the v15 stage 2 pass model (L12: MLE / MAP) on traced matches vs full.

usage (repo root):  python3 tools/fit_passes.py [path/to/agent.py]   (default tools/versions/v15/agent.py)

Data: every root "pass" decision in PlayOn (trace note: receiver, target, flight
cycles n, logged p / m / f) in the listed recent matches against full. The
state at the decision cycle comes from the replay (positions, velocities,
bodies, ball), rotated into the team frame. Features come from the agent's own
_Model.pass_features, so they are what the agent would compute.
Label: the next touch by anyone other than the passer (touches as in
tools/keep_data.py, restarts by who is awarded them): a teammate -> 1, an
opponent -> 0, nothing within n + delay + 10 cycles -> dropped.

Pipeline check: the baseline features recomputed from the replay must match the
logged m and f (otherwise the fit stops: offline features would not be what the
agent saw).

Models compared (P = sig(z) * f):
  BASE  the current curve on the current features (what v14/v11 use)
  A     refit (OFF, SCALE) on the current features
  B     refit (OFF, SCALE) on the motion-aware features margin_m, f_m
  C     refit (OFF, SCALE, PRESS) on margin_m, f_m and passer pressure x
Fit: MAP, the prior being k pseudo-observations from the current curve
(L12 p27, p15); maximised by Newton's method on (1/SCALE, -OFF/SCALE, -PRESS).
Model and k chosen by 2-fold cross-validation by match inside the training
matches; a model with more features is chosen only if its inner-CV log loss is
at least 1 % better than the simpler one's.

Split, fixed before looking: matches sorted by name, every second one held out.
Predeclared gates on the held-out matches (fixed before the first run):
  G1 log loss improves by >= 2 % relative to BASE
  G2 Brier score not worse than BASE
  G3 calibration |observed - predicted| shrinks for passes made under pressure
     (an opponent within 3 m of the passer) and gets worse by no more than 0.02
     for forward passes (target > 5 m ahead of the ball)
  G4 the log-loss improvement keeps its sign when any one held-out match is left out
  G5 the 90 % match-bootstrap interval (2000 replicates, seed 11) of the log-loss
     difference excludes zero
All pass -> ACCEPT (print the constants); otherwise KEEP the current pass model.
Probabilities are clipped to [1e-6, 1 - 1e-6]; metrics are per decision.
"""
import glob, importlib.util, json, math, os, re, sys, types
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import keep_data as K

AGENT = sys.argv[1] if len(sys.argv) > 1 else "tools/versions/v15/agent.py"
SETS = ("B11_", "A11_", "Q11_", "T11_", "E11_", "C11_", "T12_", "B13_", "A14_", "Q14_", "P14S1_", "P15S1_")
EPS = 1e-6
PRIOR_M = np.arange(-6.0, 6.01, 1.0)
K_CHOICES = (8.0, 64.0, 512.0)
RE_PLV = re.compile(r"\(\((l|r) (\d+)\) (\d+) (0x[0-9a-f]+) (\S+) (\S+) (\S+) (\S+) (\S+)")
RE_SHOW = re.compile(r"^\(show (\d+) ")

spec = importlib.util.spec_from_file_location("src.student.agent", AGENT)
A = importlib.util.module_from_spec(spec)
sys.modules["src.student.agent"] = A
spec.loader.exec_module(A)
# the simulator constants of base.conf / the proxy (as tools/test_agent.py, verified earlier)
SP = types.SimpleNamespace(player_size=0.3, ball_size=0.085, stopped_ball_vel=0.01, goal_post_radius=0.06,
    kick_rand=0.1, max_moment=180.0, inertia_moment=5.0, tackle_cycles=10, max_tackle_power=100.0,
    max_back_tackle_power=0.0, tackle_power_rate=0.027, ball_decay=0.94, ball_speed_max=3.0, max_power=100.0,
    ball_accel_max=2.7, real_speed_max=1.0, player_speed_max=1.05, kickable_area=1.085, catchable_area=1.3,
    tackle_dist=2.0, pitch_half_length=52.5, pitch_half_width=34.0, goal_width=14.02,
    their_penalty_area_line_x=36.0, our_penalty_area_line_x=-36.0, penalty_area_half_width=20.16,
    simulator_step=100, half_time=75, nr_normal_halfs=2, stamina_max=8000.0, player_decay=0.4,
    kick_power_rate=0.027, kickable_margin=0.7)
M = A._Model(SP)
BASE = (A.PASS_OFFSET, A.PASS_SCALE)


def states(rcg):
    """cycle -> {(side, unum): (x, y, vx, vy, body, goalie)}, ball (x, y, vx, vy)."""
    st = {}
    for line in open(rcg, errors="replace"):
        if not line.startswith("(show"): continue
        c = int(RE_SHOW.match(line).group(1))
        pl = {}
        for side, unum, _, f, x, y, vx, vy, body in RE_PLV.findall(line):
            fl = int(f, 16)
            if fl == 0: continue
            pl[(side, int(unum))] = (float(x), float(y), float(vx), float(vy), float(body), bool(fl & 0x8))
        st[c] = pl
    return st


def label_pass(g, c, side, passer, n):
    for t in range(c + 1, c + int(n) + 12):
        mode = g["modes"].get(t)
        if mode and mode != "play_on":
            a = K.mode_award(mode)
            if isinstance(a, tuple): return 1 if a[1] == side else 0
            if a is not None: return 1 if a == side else 0
            return None
        ts = [x for x in g["touch"].get(t, []) if x != (side, passer)]
        sides = {s for s, _ in ts}
        if len(sides) == 2: return None
        if sides: return 1 if side in sides else 0
    return None


def krate(me, ball):
    """rcssserver effective kick power rate for the ball relative to the kicker."""
    rel = ball[:2] - me[:2]
    d = np.linalg.norm(rel)
    ang = abs(A._norm_deg(math.degrees(math.atan2(rel[1], rel[0])) - me[4]))
    rate = SP.kick_power_rate * (1.0 - 0.25 * ang / 180.0
                                 - 0.25 * max(0.0, d - SP.player_size - SP.ball_size) / SP.kickable_margin)
    return max(rate, 0.0)


def rows_for(d):
    tag = os.path.basename(d); rcg = f"logs/{tag}.rcg"
    if not os.path.exists(rcg): return []
    g = K.parse_rcg(rcg)
    side = {v: k for k, v in g["names"].items()}.get("ME")
    if side is None: return []
    sg = 1.0 if side == "l" else -1.0
    st = states(rcg)
    out = []
    for f in glob.glob(f"{d}/ME/*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if not r.get("kickable") or r.get("mode") != 2: continue
            n_ = next((x for x in r.get("notes", []) if x["note"] == "pass"), None)
            if n_ is None or "m" not in n_: continue
            c = r["c"]; u = r["unum"]
            if c not in st or c not in g["ball"]: continue
            pl = st[c]
            def tf(v):                     # replay -> team frame
                x, y, vx, vy, body, gk = v
                return np.array([sg * x, sg * y, sg * vx, sg * vy, A._norm_deg(body + (0.0 if sg > 0 else 180.0)), gk], float)
            if (side, u) not in pl: continue
            me = tf(pl[(side, u)])
            mates = {k[1]: tf(v) for k, v in pl.items() if k[0] == side and k[1] != u and not v[5]}
            opps = [tf(v) for k, v in pl.items() if k[0] != side]
            if n_["to"] not in mates or not opps: continue
            bx, by, bvx, bvy = g["ball"][c]
            ball = np.array([sg * bx, sg * by, sg * bvx, sg * bvy])
            order = sorted(mates)
            mp = np.array([mates[k][:2] for k in order]); mv = np.array([mates[k][2:4] for k in order])
            mb = np.array([mates[k][4] for k in order])
            op = np.array([o[:2] for o in opps]); ov = np.array([o[2:4] for o in opps])
            ob = np.array([o[4] for o in opps]); gk = np.array([bool(o[5]) for o in opps])
            ri = order.index(n_["to"])
            pt = np.array(n_["at"], float); n = float(n_["n"])
            vec = pt - ball[:2]; dist = float(np.linalg.norm(vec))
            if dist < 1e-6: continue
            dirs = (vec / dist)[None]; speed = np.array([M.speed_for(dist, n)])
            two = speed[0] > M.one_step_speed(dirs, ball[2:4], krate(me, ball))[0] + 1e-6
            delay = 1.0 if two else 0.0
            cand = (pt[None], np.array([ri]), np.array([n]), speed, dirs, np.array([dist]))
            p0, m0, f0 = M.pass_p(ball[:2], cand, delay, mp, op, gk)
            mm, late, x, fm = M.pass_features(ball[:2], cand, delay, mp, mv, mb, op, ov, ob, gk)
            y = label_pass(g, c, side, u, n + delay)
            if y is None: continue
            press_d = float(np.min(np.linalg.norm(op - me[:2], axis=1)))
            out.append({"match": tag, "y": y, "m0": float(m0[0]), "f0": float(f0[0]), "mm": float(mm[0]),
                        "fm": float(fm[0]), "x": float(x[0]), "log_m": float(n_["m"]), "log_f": float(n_.get("f", 1.0)),
                        "log_p": float(n_["p"]), "press_d": press_d, "fwd": float(pt[0] - ball[0])})
    return out


def prob(th, m, f, x):
    a, b, c = th
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(a * m + b - c * x, -30, 30))) * f, EPS, 1 - EPS)


def ll(th, m, f, x, y, w=None):
    p = prob(th, m, f, x)
    v = y * np.log(p) + (1 - y) * np.log(1 - p)
    return float(np.sum(v if w is None else w * v))


def fit(rows, feats, k, with_x):
    m = np.array([r[feats[0]] for r in rows]); f = np.array([r[feats[1]] for r in rows])
    x = np.array([r["x"] for r in rows]) if with_x else np.zeros(len(rows))
    y = np.array([r["y"] for r in rows], float)
    a0, b0 = 1.0 / BASE[1], -BASE[0] / BASE[1]
    pp = 1.0 / (1.0 + np.exp(-(a0 * PRIOR_M + b0))); wk = k / len(PRIOR_M)
    M_ = np.concatenate([m, PRIOR_M, PRIOR_M]); F_ = np.concatenate([f, np.ones(2 * len(PRIOR_M))])
    X_ = np.concatenate([x, np.zeros(2 * len(PRIOR_M))])
    Y_ = np.concatenate([y, np.ones(len(PRIOR_M)), np.zeros(len(PRIOR_M))])
    W_ = np.concatenate([np.ones(len(m)), wk * pp, wk * (1 - pp)])
    th = np.array([a0, b0, 0.0])
    free = [0, 1, 2] if with_x else [0, 1]
    obj = lambda t: ll(t, M_, F_, X_, Y_, W_)
    # Fisher scoring with the exact gradient: z = a*m + b - c*x, p = sig(z) * f.
    # (A first version used numeric second differences of the summed log-likelihood;
    # they were too noisy and the 3-parameter fit never left its start.)
    U = np.stack([M_, np.ones_like(M_), -X_], axis=1)[:, free]
    for _ in range(100):
        z = np.clip(U @ th[free], -30, 30)
        s = 1.0 / (1.0 + np.exp(-z)); p = np.clip(s * F_, EPS, 1 - EPS)
        dp = F_ * s * (1 - s)                                  # dp/dz
        g_ = U.T @ (W_ * (Y_ / p - (1 - Y_) / (1 - p)) * dp)
        info = (U * (W_ * dp * dp / (p * (1 - p)))[:, None]).T @ U   # expected information
        try:
            step = np.linalg.solve(info + 1e-9 * np.eye(len(free)), g_)
        except np.linalg.LinAlgError:
            break
        base = obj(th); t = 1.0
        while t > 1e-6:
            cand = th.copy(); cand[free] = th[free] + t * step
            if obj(cand) >= base: break
            t /= 2
        if t <= 1e-6: break
        th = cand
        if np.max(np.abs(t * step)) < 1e-7: break
    return th


def metrics(th, rows, feats, with_x):
    m = np.array([r[feats[0]] for r in rows]); f = np.array([r[feats[1]] for r in rows])
    x = np.array([r["x"] for r in rows]) if with_x else np.zeros(len(rows))
    y = np.array([r["y"] for r in rows], float)
    p = prob(th, m, f, x)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)), np.mean((p - y) ** 2), p, y


MODELS = {"A": (("m0", "f0"), False), "B": (("mm", "fm"), False), "C": (("mm", "fm"), True)}


def main():
    dirs = sorted(d for s in SETS for d in glob.glob(f"runlogs/trace/*_ME_vs_{s}*"))
    rows = [r for d in dirs for r in rows_for(d)]
    matches = sorted({r["match"] for r in rows})
    print(f"{len(rows)} labelled root passes in {len(matches)} matches; success rate {np.mean([r['y'] for r in rows]):.3f}")
    # pipeline check: recomputed baseline features vs logged values
    dm = np.array([abs(r["m0"] - r["log_m"]) for r in rows]); df = np.array([abs(r["f0"] - r["log_f"]) for r in rows])
    ok_m, ok_f = np.mean(dm < 0.05), np.mean(df < 0.02)
    print(f"pipeline check: recomputed m within 0.05 of logged for {ok_m:.1%}, f within 0.02 for {ok_f:.1%}")
    if ok_m < 0.95 or ok_f < 0.95:
        print("STOP: offline features do not reproduce what the agent logged"); return
    test = set(matches[1::2])
    tr = [r for r in rows if r["match"] not in test]; te = [r for r in rows if r["match"] in test]
    a0, b0 = 1.0 / BASE[1], -BASE[0] / BASE[1]
    base_th = np.array([a0, b0, 0.0])
    # model and k by 2-fold CV by match inside training
    trm = sorted({r["match"] for r in tr}); fold = {mm: i % 2 for i, mm in enumerate(trm)}
    cv = {}
    for name, (feats, wx) in MODELS.items():
        for k in K_CHOICES:
            s = 0.0
            for fo in (0, 1):
                a = [r for r in tr if fold[r["match"]] != fo]; b = [r for r in tr if fold[r["match"]] == fo]
                th = fit(a, feats, k, wx)
                s += metrics(th, b, feats, wx)[0] * len(b)
            cv[(name, k)] = s / len(tr)
    for key, v in sorted(cv.items()): print(f"  inner CV {key[0]} k={key[1]:<5g} log loss {v:.5f}")
    best = {nm: min((v, k) for (n2, k), v in cv.items() if n2 == nm) for nm in MODELS}
    chosen = "A"
    for nm in ("B", "C"):
        if best[nm][0] <= 0.99 * best[chosen][0]: chosen = nm
    feats, wx = MODELS[chosen]; k = best[chosen][1]
    th = fit(tr, feats, k, wx)
    print(f"chosen model {chosen} (k={k:g}): OFF={-th[1] / th[0]:.3f} SCALE={1 / th[0]:.3f} PRESS={th[2]:.3f}")
    llb, brb, pb, y = metrics(base_th, te, ("m0", "f0"), False)
    llc, brc, pc, _ = metrics(th, te, feats, wx)
    print(f"held out ({len(te)} passes, {len(test)} matches): log loss {llb:.5f} -> {llc:.5f} ({100 * (llc - llb) / llb:+.1f}%),"
          f" Brier {brb:.5f} -> {brc:.5f}")
    press = np.array([r["press_d"] <= 3.0 for r in te]); fwd = np.array([r["fwd"] > 5.0 for r in te])
    def cal(p, s): return abs(y[s].mean() - p[s].mean())
    for nm, s in (("all", np.ones(len(te), bool)), ("pressed (opp <= 3 m)", press), ("forward > 5 m", fwd),
                  ("margin < -2", np.array([r["m0"] < -2 for r in te]))):
        print(f"   {nm:22s} n={s.sum():5d} observed {y[s].mean():.3f} base {pb[s].mean():.3f} new {pc[s].mean():.3f}"
              f"  |err| {cal(pb, s):.3f} -> {cal(pc, s):.3f}")
    tm = sorted(test); idx = {mm: np.array([i for i, r in enumerate(te) if r["match"] == mm]) for mm in tm}
    def lld(sel):
        yy = y[sel]; a = pb[sel]; b = pc[sel]
        return (-np.mean(yy * np.log(b) + (1 - yy) * np.log(1 - b))) - (-np.mean(yy * np.log(a) + (1 - yy) * np.log(1 - a)))
    loo = [lld(np.concatenate([idx[o] for o in tm if o != mm])) for mm in tm]
    rng = np.random.default_rng(11)
    boot = [lld(np.concatenate([idx[o] for o in rng.choice(tm, len(tm))])) for _ in range(2000)]
    lo, hi = np.percentile(boot, [5, 95])
    print(f"   log-loss difference: leave-one-match-out range {min(loo):+.5f}..{max(loo):+.5f}; bootstrap 90% [{lo:+.5f}, {hi:+.5f}]")
    gates = {"G1 log loss -2%": (llc - llb) / llb <= -0.02, "G2 Brier not worse": brc <= brb,
             "G3 pressed better, forward not worse >0.02": cal(pc, press) < cal(pb, press) and cal(pc, fwd) - cal(pb, fwd) <= 0.02,
             "G4 leave-one-match-out sign": all(v < 0 for v in loo), "G5 bootstrap excludes 0": hi < 0}
    for g_, v in gates.items(): print(f"  {g_:44s} {'PASS' if v else 'FAIL'}")
    if all(gates.values()):
        tha = fit(rows, feats, k, wx)
        print(f"DECISION: ACCEPT model {chosen}; refit on all {len(rows)}: PASS2_OFFSET={-tha[1] / tha[0]:.3f}"
              f" PASS2_SCALE={1 / tha[0]:.3f} PASS2_PRESS={tha[2]:.3f} (features {'motion' if chosen != 'A' else 'current'})")
    else:
        print("DECISION: KEEP the current pass model")


if __name__ == "__main__":
    main()
