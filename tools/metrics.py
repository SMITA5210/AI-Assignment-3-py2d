"""Per-team match metrics from .rcg replays, beyond the scoreline.

usage (repo root):  python3 tools/metrics.py logs/<match>.rcg [...]
                    python3 tools/metrics.py --team ME logs/*_ME_vs_F*.rcg   (totals for one team)

Everything comes from the replay, so it works for any two teams (self-play
included). Flags used (rcssserver player state, verified on real replays):
KICK 0x2, CATCH 0x10, TACKLE 0x1000; play-mode suffix _l/_r names the team
the restart is awarded to (or, for fouls/offside, the team at fault).

  possession   share of touches-to-touches time each team had the ball
  passes       a touch followed by a different teammate's touch (completed)
               or an opponent's touch / restart against (failed)
  shots        a touch whose ball path heads for the goal mouth from <= 35 m
  big chances  shots from inside 20 m
  turnovers    possession changes by an opponent touch, by own half and theirs
"""
import argparse, re, sys
from collections import Counter, defaultdict

KICK, CATCH, TACKLE = 0x2, 0x10, 0x1000
RE_SHOW = re.compile(r"^\(show (\d+) ")
RE_BALL = re.compile(r"\(\(b\) (\S+) (\S+) (\S+) (\S+)\)")
RE_PLAYER = re.compile(r"\(\((l|r) (\d+)\) \d+ (0x[0-9a-f]+) (\S+) (\S+)")
RE_MODE = re.compile(r"^\(playmode (\d+) (\S+?)\)")
HALF, GOAL_HALF, DECAY = 52.5, 7.01, 0.94


def heads_for_goal(bx, by, vx, vy, side):
    """Ball path crosses the goal line of the team NOT on `side`, inside the posts."""
    gx = HALF if side == "l" else -HALF
    if (gx - bx) * vx <= 0 or abs(vx) < 1e-6:
        return False
    speed = (vx * vx + vy * vy) ** 0.5
    along = abs(gx - bx) * speed / abs(vx)     # path length to the goal line
    if along > speed / (1 - DECAY):            # stops short (geometric series)
        return False
    return abs(by + vy * (gx - bx) / vx) <= GOAL_HALF + 1.0


def analyse(path):
    names, score = {}, (0, 0)
    touches = []            # (cycle, side, unum, ball x, ball y)
    modes = []              # (cycle, mode)
    ball_at = {}
    tackling = set()        # TACKLE stays set for tackle_cycles: count its first cycle only
    for line in open(path, errors="replace"):
        if line.startswith("(team"):
            p = line.split(); names = {"l": p[2], "r": p[3]}; score = (int(p[4]), int(p[5].rstrip(")")))
        elif line.startswith("(playmode"):
            m = RE_MODE.match(line)
            if m: modes.append((int(m.group(1)), m.group(2)))
        elif line.startswith("(show"):
            c = int(RE_SHOW.match(line).group(1))
            b = RE_BALL.search(line)
            bx, by, vx, vy = map(float, b.groups())
            ball_at[c] = (bx, by, vx, vy)
            now = set()
            for side, unum, st, x, y in RE_PLAYER.findall(line):
                f = int(st, 16)
                if f & TACKLE: now.add((side, unum))
                if (f & (KICK | CATCH)) or ((f & TACKLE) and (side, unum) not in tackling):
                    touches.append((c, side, int(unum), bx, by))
            tackling = now

    out = {s: Counter() for s in ("l", "r")}
    restart_side = {}
    for c, mode in modes:
        m = re.match(r"(kick_in|free_kick|corner_kick|goal_kick|indirect_free_kick)_(l|r)$", mode)
        if m: restart_side[c] = m.group(2)
        m = re.match(r"(foul_charge|foul_push|offside|back_pass)_(l|r)$", mode)
        if m: out[m.group(2)][m.group(1)] += 1

    # possession time and turnovers from touch sequence
    last = None
    for i, (c, side, unum, bx, by) in enumerate(touches):
        if last is not None:
            pc, pside, punum, pbx, pby = last
            out[pside]["poss_cycles"] += c - pc
            if side == pside and unum != punum:
                out[pside]["passes_ok"] += 1
            elif side != pside:
                out[pside]["passes_lost_or_turnover"] += 1
                own_half = (pbx < 0) if pside == "l" else (pbx > 0)
                out[pside]["turnover_own_half" if own_half else "turnover_their_half"] += 1
        # shots: next cycle's ball velocity after this touch
        nxt = ball_at.get(c + 1)
        if nxt:
            nbx, nby, nvx, nvy = nxt
            gx = HALF if side == "l" else -HALF
            dist = ((gx - bx) ** 2 + by ** 2) ** 0.5
            if dist <= 35.0 and heads_for_goal(nbx, nby, nvx, nvy, side):
                out[side]["shots"] += 1
                if dist <= 20.0: out[side]["big_chances"] += 1
        last = (c, side, unum, bx, by)
    for c, s in restart_side.items():          # restart against the last toucher = lost ball
        prev = [t for t in touches if t[0] <= c]
        if prev and prev[-1][1] != s:
            out[prev[-1][1]]["ball_out_lost"] += 1

    total = sum(out[s]["poss_cycles"] for s in "lr") or 1
    res = {}
    for s in "lr":
        o = out[s]
        att = o["passes_ok"] + o["passes_lost_or_turnover"]
        res[names.get(s, s)] = {
            "goals": score[0] if s == "l" else score[1],
            "possession": round(100.0 * o["poss_cycles"] / total, 1),
            "touch_chains_ok": o["passes_ok"], "chains_lost": o["passes_lost_or_turnover"],
            "shots": o["shots"], "big_chances": o["big_chances"],
            "turnovers_own_half": o["turnover_own_half"], "turnovers_their_half": o["turnover_their_half"],
            "ball_out_lost": o["ball_out_lost"],
            "fouls": o["foul_charge"] + o["foul_push"], "offside": o["offside"], "back_pass": o["back_pass"],
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--team", help="sum this team's rows over all logs")
    a = ap.parse_args()
    tot, opp, n = Counter(), Counter(), 0
    for p in a.logs:
        r = analyse(p)
        if a.team:
            if a.team not in r: continue
            n += 1
            tot.update(r[a.team])
            other = [k for k in r if k != a.team]
            if other: opp.update(r[other[0]])
        else:
            print(p.rsplit("/", 1)[-1])
            for team, m in r.items():
                print(f"   {team:<10} " + "  ".join(f"{k}={v}" for k, v in m.items()))
    if a.team:
        print(f"{a.team}: {n} matches, per-match averages (opponents in brackets)")
        for k in tot:
            print(f"   {k:<22} {tot[k] / max(n, 1):7.2f}   ({opp[k] / max(n, 1):.2f})")


if __name__ == "__main__":
    main()
