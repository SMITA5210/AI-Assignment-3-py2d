"""
Per-cycle decision trace.

    python3 a3.py match A B        then    python3 a3.py why

Use --no-trace to stop tracing.
Budget records timing regardless of tracing.
"""

import json
import os
import threading

ENABLED = os.environ.get("TRACE", "") not in ("", "0", "false", "False")
TRACE_DIR = os.path.join(os.environ.get("TRACE_DIR", "runlogs/trace"),
                         os.environ.get("TRACE_STAMP", "default"))

_files = {}
_lock = threading.Lock()
_broken = set()

# Notes are keyed by THREAD
_notes = {}


def note(agent, what, **fields):
    """Attach a note to this cycle's trace line for this player."""
    if ENABLED:
        _notes.setdefault(threading.get_ident(), []).append(
            dict(note=what, **fields))


def _take_notes():
    return _notes.pop(threading.get_ident(), None)


def _handle(team, unum):
    key = (team, unum)
    if key not in _files:
        d = os.path.join(TRACE_DIR, str(team))
        os.makedirs(d, exist_ok=True)
        _files[key] = open(os.path.join(d, f"{unum}.jsonl"), "w",
                           buffering=1)
    return _files[key]


def _describe(action):
    """Turn a PlayerAction protobuf into a dict."""
    kind = action.WhichOneof("action")
    if kind is None:
        return {"kind": "none"}
    body = getattr(action, kind)
    out = {"kind": kind}
    for fd, val in body.ListFields():
        if hasattr(val, "x") and hasattr(val, "y"):
            out[fd.name] = [round(val.x, 2), round(val.y, 2)]
        elif isinstance(val, float):
            out[fd.name] = round(val, 3)
        else:
            out[fd.name] = val
    return out


def record(agent, actions, budget=None):
    """Called by the framework after on_ball/off_ball returns.
    `agent` is the RAW agent
    `budget` is the guard.Budget the decision ran inside (CPU time)
    """
    if not ENABLED:
        return
    try:
        wm = agent.wm
        me = wm.self

        opps = [o for o in wm.opponents if o.uniform_number > 0]
        nearest = min(opps, key=lambda o: o.dist_from_self, default=None)

        row = {
            "c": wm.cycle,
            "unum": me.uniform_number,
            "mode": wm.game_mode_type,
            "pos": [round(me.position.x, 2), round(me.position.y, 2)],
            "ball": [round(wm.ball.position.x, 2), round(wm.ball.position.y, 2)],
            "kickable": me.is_kickable,
            "stamina": round(me.stamina),
            "reach": [wm.intercept_table.self_reach_steps,
                      wm.intercept_table.first_teammate_reach_steps,
                      wm.intercept_table.first_opponent_reach_steps],
            "nearest_opp": (
                [nearest.uniform_number, round(nearest.dist_from_self, 1)]
                if nearest else None),
            "actions": [_describe(a) for a in actions],
        }

        if budget is None:
            pass
        elif isinstance(budget, (int, float)):
            row["ms"] = round(float(budget), 2)
        else:
            row["ms"] = round(budget.cpu[-1], 2) if budget.cpu else 0.0
            row["wall_ms"] = round(budget.wall[-1], 2) if budget.wall else 0.0
            left = budget.remaining_ms
            if left != float("inf"):
                row["left_ms"] = round(left, 1)

        notes = _take_notes()
        if notes:
            row["notes"] = notes

        with _lock:
            _handle(wm.our_team_name, me.uniform_number).write(
                json.dumps(row) + "\n")

    except Exception as e:
        # Stop tracing
        key = getattr(getattr(agent, "wm", None), "our_team_name", "?")
        if key not in _broken:
            _broken.add(key)
            import traceback
            print(f"[trace] record failed - tracing OFF for {key}: {e}",
                  flush=True)
            traceback.print_exc()
        globals()["ENABLED"] = False