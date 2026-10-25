"""Summarise .rcg replays: score, referee events per offending team, card bits.

Playmode suffix _l/_r names the team at fault (verified: foul_charge_l was
followed by free_kick_r, back_pass_r by indirect_free_kick_l). Card bits are
rcssserver's player-state flags YELLOW 0x40000, RED 0x80000 (unconfirmed
until seen).
"""
import re, sys
from collections import Counter

def stats(path):
    names, score, ev, cards = {}, None, Counter(), {}
    prev = None
    for line in open(path, errors="replace"):
        if line.startswith("(team"):
            _, _, l, r, sl, sr = line.strip("()\n").split()[:6]
            names = {"l": l, "r": r}; score = (int(sl), int(sr))
        elif line.startswith("(playmode"):
            mode = line.split()[2].rstrip(")")
            if mode != prev:
                m = re.match(r"(foul_\w+|offside|back_pass|free_kick_fault|catch_fault|goal)_(l|r)$", mode)
                if m:
                    ev[(names.get(m.group(2), m.group(2)), m.group(1))] += 1
            prev = mode
        elif line.startswith("(show"):
            for side, unum, state in re.findall(r"\(\((l|r) (\d+)\) \d+ (0x[0-9a-f]+)", line):
                s = int(state, 16)
                c = 2 if s & 0x80000 else (1 if s & 0x40000 else 0)
                key = (names.get(side, side), int(unum))
                cards[key] = max(cards.get(key, 0), c)
    out = {"teams": names, "score": score, "events": dict(ev),
           "yellow": Counter(k[0] for k, v in cards.items() if v == 1),
           "red": Counter(k[0] for k, v in cards.items() if v == 2)}
    return out

if __name__ == "__main__":
    for p in sys.argv[1:]:
        s = stats(p)
        print(p.split("/")[-1], s["teams"], s["score"])
        print("   events:", s["events"], " yellow:", dict(s["yellow"]), " red:", dict(s["red"]))
