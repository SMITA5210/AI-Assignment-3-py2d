"""Outcome labels for Agent1's keep-the-ball decisions (dribble, hold), from
replays + traces, and a validation report for those labels.

usage (repo root):
  python3 tools/keep_data.py validate 'V3F*' 'V8BF*'      # label checks on traced matches
  python3 tools/keep_data.py rows OUT.jsonl 'K1C*' [...]   # one row per decision, for fitting

Opponent patterns are globs on the opponent's team name in the log name
(runlogs/trace/<stamp>_ME_vs_<NAME>, logs/<stamp>_ME_vs_<NAME>.rcg).

Label (event-based, "who touches next"): a decision taken at cycle c shows
its own touch, if it kicks, at c+1 (the server runs the command between c
and c+1). Walk forward from c+1 up to CAP cycles:
  - a play-mode change decides it: restart awarded to us = kept, to them =
    lost (fault modes - foul, offside, back pass - award the other side);
    a goal for us = kept, for them = lost; end of a half = unknown;
  - otherwise the first touch decides it, skipping the decider's own touch
    at c+1: ours = kept, theirs = lost; both sides in the same cycle, or a
    touch and a restart that disagree = unknown (contested);
  - nothing within CAP cycles = unknown.
Touches are player-state flags KICK 0x2, CATCH 0x10, TACKLE 0x1000.
Window labels (fit_probs.py style, for comparison): the first event within
W cycles decides as above, no event = kept.
"""
import glob, json, math, os, re, sys
from collections import Counter, defaultdict

KICK, CATCH, TACKLE = 0x2, 0x10, 0x1000
TOUCH = KICK | CATCH | TACKLE
CAP = 20                      # cycles; a full-speed ball is long gone by then
KINDS = ("dribble", "hold")
RESTART = ("kick_in", "free_kick", "corner_kick", "goal_kick", "indirect_free_kick",
           "goalie_catch_ball")
FAULT = ("foul_charge", "foul_push", "offside", "back_pass", "free_kick_fault",
         "catch_fault")
RE_MODE = re.compile(r"\(playmode (\d+) (\S+?)\)")
RE_BALL = re.compile(r"\(\(b\) (\S+) (\S+) (\S+) (\S+)\)")
RE_PL = re.compile(r"\(\((l|r) (\d+)\) (\d+) (0x[0-9a-f]+) (\S+) (\S+) (\S+) (\S+) (\S+)")


def parse_rcg(path):
    """Server params, team names, per-cycle ball/players, touches and play modes."""
    g = {"sp": {}, "names": {}, "ball": {}, "players": {}, "touch": defaultdict(list),
         "modes": {}}
    tackling = set()      # players whose TACKLE flag was already set last cycle
    for line in open(path, errors="replace"):
        if line.startswith("(server_param"):
            for k, v in re.findall(r"\((\w+) ([^()\s]+)\)", line):
                try: g["sp"][k] = float(v)
                except ValueError: pass
        elif line.startswith("(team"):
            p = line.split(); g["names"] = {"l": p[2], "r": p[3]}
        elif line.startswith("(playmode"):
            m = RE_MODE.match(line)
            if m: g["modes"][int(m.group(1))] = m.group(2)
        elif line.startswith("(show"):
            c = int(line.split()[1])
            g["ball"][c] = tuple(map(float, RE_BALL.search(line).groups()))
            pl = []
            now_tackling = set()
            for side, unum, _, st, x, y, vx, vy, body in RE_PL.findall(line):
                f = int(st, 16)
                if f == 0: continue
                k = (side, int(unum))
                pl.append((side, int(unum), f, float(x), float(y), float(body)))
                # the TACKLE flag stays set for the whole tackle_cycles freeze:
                # only its first cycle is a touch (KICK and CATCH last one cycle)
                if f & TACKLE: now_tackling.add(k)
                if (f & (KICK | CATCH)) or ((f & TACKLE) and k not in tackling):
                    g["touch"][c].append(k)
            tackling = now_tackling
            g["players"][c] = pl
    return g


def mode_award(mode):
    """Side a play-mode change hands the ball to: 'l'/'r', or ('goal', side),
    or None when it says nothing about possession."""
    m = re.match(r"(\w+?)_(l|r)$", mode)
    if not m:
        return None
    kind, side = m.group(1), m.group(2)
    other = "r" if side == "l" else "l"
    if kind == "goal": return ("goal", side)
    if kind in RESTART: return side
    if kind in FAULT: return other
    return None


def label_event(g, c, me_side, unum):
    """(label, reason, cycles to resolve): label 1 kept, 0 lost, None unknown."""
    for t in range(c + 1, c + CAP + 1):
        verdict = []
        mode = g["modes"].get(t)
        if mode and mode != "play_on":
            a = mode_award(mode)
            if isinstance(a, tuple):
                verdict.append((1 if a[1] == me_side else 0, "goal"))
            elif a is not None:
                verdict.append((1 if a == me_side else 0, "restart"))
            else:
                return None, "mode:" + mode, t - c
        ts = [x for x in g["touch"].get(t, []) if not (t == c + 1 and x == (me_side, unum))]
        sides = {s for s, _ in ts}
        if len(sides) == 2:
            return None, "contested", t - c
        if sides:
            verdict.append((1 if me_side in sides else 0, "touch"))
        if verdict:
            if len({v for v, _ in verdict}) > 1:
                return None, "conflict", t - c
            return verdict[0][0], verdict[-1][1], t - c
    return None, "no_event", CAP


def label_window(g, c, me_side, unum, W):
    """fit_probs.py-style: first event within W decides, none = kept."""
    for t in range(c + 1, c + W + 1):
        mode = g["modes"].get(t)
        if mode and mode != "play_on":
            a = mode_award(mode)
            if isinstance(a, tuple): return 1 if a[1] == me_side else 0
            if a is not None: return 1 if a == me_side else 0
        ts = [x for x in g["touch"].get(t, []) if x != (me_side, unum)]
        if any(s != me_side for s, _ in ts): return 0
        if ts: return 1
    return 1


def decisions(pattern, team="ME"):
    """Yield (tag, g, me_side, trace row, note) for every traced keep decision in PlayOn."""
    for d in sorted(glob.glob(f"runlogs/trace/*_{team}_vs_{pattern}")):
        tag = os.path.basename(d)
        rcg = f"logs/{tag}.rcg"
        if not os.path.exists(rcg): continue
        g = parse_rcg(rcg)
        me_side = {v: k for k, v in g["names"].items()}.get(team)
        if me_side is None: continue
        for p in sorted(glob.glob(f"{d}/{team}/*.jsonl")):
            for line in open(p):
                r = json.loads(line)
                if not r.get("kickable") or r.get("mode") != 2: continue
                for n in r.get("notes", []):
                    if n["note"] in KINDS:
                        yield tag, g, me_side, r, n
                        break


def validate(patterns):
    for kind in KINDS:
        st = Counter(); delay = Counter(); agree = {W: Counter() for W in (5, 8, 12)}
        ball_err = []; own_next = 0; n = 0; matches = set()
        for pat in patterns:
            for tag, g, side, r, note in decisions(pat):
                if note["note"] != kind: continue
                n += 1; matches.add(tag)
                c, u = r["c"], r["unum"]
                sg = 1.0 if side == "l" else -1.0
                if c in g["ball"]:
                    bx, by = g["ball"][c][0] * sg, g["ball"][c][1] * sg
                    ball_err.append(math.hypot(bx - r["ball"][0], by - r["ball"][1]))
                if (side, u) in g["touch"].get(c + 1, []): own_next += 1
                y, why, dt = label_event(g, c, side, u)
                st[(y, why)] += 1
                if y is not None:
                    delay["1-3" if dt <= 3 else "4-8" if dt <= 8 else "9-12" if dt <= 12 else "13-20"] += 1
                for W in agree:
                    agree[W][(y, label_window(g, c, side, u, W))] += 1
        if not n: continue
        ball_err.sort()
        print(f"\n== {kind}: {n} decisions in {len(matches)} matches ({', '.join(patterns)})")
        print(f"   cycle alignment: trace ball vs replay ball at the same cycle: median "
              f"{ball_err[len(ball_err) // 2]:.3f} m, 99th pct {ball_err[int(len(ball_err) * .99)]:.3f} m")
        print(f"   decider's own touch shown at c+1: {own_next / n:.0%}")
        known = sum(v for (y, _), v in st.items() if y is not None)
        print(f"   event label: kept {sum(v for (y, _), v in st.items() if y == 1)}"
              f"  lost {sum(v for (y, _), v in st.items() if y == 0)}"
              f"  unknown {n - known} ({(n - known) / n:.1%})")
        for (y, why), v in sorted(st.items(), key=lambda t: -t[1]):
            print(f"      {str(y):5s} {why:22s} {v}")
        print("   cycles to the deciding event (known labels):", dict(sorted(delay.items())))
        for W, a in agree.items():
            lost_e = sum(v for (e, w), v in a.items() if e == 0)
            miss = a[(0, 1)]
            unk_kept = a[(None, 1)]
            print(f"   window {W:2d}: agrees with event label on {sum(v for (e, w), v in a.items() if e is not None and e == w)}/{known};"
                  f" event-lost but window-kept {miss}/{lost_e};"
                  f" event-unknown labelled kept {unk_kept}, lost {a[(None, 0)]}")


def rows(out_path, patterns, team="ME"):
    """One row per collector keep_ctx note (tools/collect/agent.py), joined
    with the decision note and the labels. Exploratory rows are marked by
    explore_draw / deviated / prop, as logged."""
    n = Counter()
    with open(out_path, "w") as out:
        for pat in patterns:
            for d in sorted(glob.glob(f"runlogs/trace/*_{team}_vs_{pat}")):
                tag = os.path.basename(d)
                rcg = f"logs/{tag}.rcg"
                if not os.path.exists(rcg):
                    n["no_replay"] += 1; continue
                g = parse_rcg(rcg)
                side = {v: k for k, v in g["names"].items()}.get(team)
                for p in sorted(glob.glob(f"{d}/{team}/*.jsonl")):
                    for line in open(p):
                        r = json.loads(line)
                        notes = r.get("notes", [])
                        ctx = next((x for x in notes if x["note"] == "keep_ctx"), None)
                        if ctx is None: continue
                        if r.get("mode") != 2 or not r.get("kickable"):
                            n["skipped_not_playon_kickable"] += 1; continue
                        dec = next((x for x in notes if x["note"] not in ("model", "keep_ctx")), {})
                        if dec.get("note") != ctx["kind"]:
                            n["decision_note_mismatch"] += 1
                        y, why, dt = label_event(g, r["c"], side, r["unum"])
                        row = {"match": tag, "c": r["c"], "unum": r["unum"],
                               **{k: v for k, v in ctx.items() if k != "note"},
                               "model_p": dec.get("p"), "model_m": dec.get("m"),
                               "label": y, "why": why, "dt": dt,
                               **{f"label_w{W}": label_window(g, r["c"], side, r["unum"], W)
                                  for W in (5, 8, 12)}}
                        out.write(json.dumps(row) + "\n")
                        n["rows"] += 1
                        n["explore_draw"] += bool(ctx.get("explore_draw"))
                        n["deviated"] += bool(ctx.get("deviated"))
                        n["label_unknown"] += y is None
    return n


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(sys.argv[2:])
    elif sys.argv[1] == "rows":
        print(dict(rows(sys.argv[2], sys.argv[3:])))
    else:
        raise SystemExit(__doc__)
