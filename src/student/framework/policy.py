"""
Which actions student code may send.

POLICY=off|warn|strip|raise
"""
import os

# Always allowed: raw commands, plus the body_/neck_/view_/focus_ families.
_RAW = {"dash", "turn", "kick", "tackle", "catch", "move", "turn_neck",
        "change_view", "log", "debug_client"}
_ALLOWED_PREFIXES = ("body_", "neck_", "view_", "focus_")

# No communication. Under fullstate every player already sees everything,
_NO_COMMS = {"say", "attention_to", "attention_to_of", "point_to",
             "point_to_of"}
_NO_COMMS_WHY = ("no communication in this assignment. Every player already "
                 "sees the whole field.")

# bhv_* actions with a specific reason. Any other bhv_* is banned too.
_BANNED = {
    "bhv_do_heard_pass_recieve":
        "this only follows a teammate's message, and coordination is what "
        "you are graded on. Decide for yourself.",
    "bhv_do_force_kick":
        "a clearance decision (whether, and where). Generate and evaluate it.",
    "bhv_emergency":
        "stamina management is yours. Control dash power directly.",
    "bhv_go_to_point_look_ball":
        "use body_go_to_point; looking does nothing under fullstate.",
    "bhv_scan_field":
        "does nothing under fullstate - your world model is already exact.",
    "bhv_body_neck_to_ball": "use body_turn_to_ball.",
    "bhv_body_neck_to_point": "use body_turn_to_point.",
    "bhv_neck_body_to_ball": "use body_turn_to_ball.",
    "bhv_neck_body_to_point": "use body_turn_to_point.",
    "bhv_before_kick_off": "kickoff is handled for you.",
    "bhv_goalie_free_kick": "the goalie is run for you.",
}
# helios_* used by the framework's own goalie and set-play code
_FRAMEWORK_HELIOS = {"helios_goalie", "helios_goalie_move",
                     "helios_goalie_kick", "helios_set_play"}

MODE = os.environ.get("POLICY", "strip")
_violations = {}                  # action name -> count, for summary()


class PolicyViolation(Exception):
    pass


def _name(action):
    """Which action a PlayerAction protobuf holds, e.g. 'dash'."""
    try:
        return action.WhichOneof("action") or type(action).__name__
    except Exception:
        return type(action).__name__


def verdict(name):
    """-> (allowed, reason). reason is None when allowed."""
    if name in _NO_COMMS:
        return False, _NO_COMMS_WHY
    if name in _RAW or name.startswith(_ALLOWED_PREFIXES):
        return True, None
    if name in _BANNED:
        return False, _BANNED[name]
    if name in _FRAMEWORK_HELIOS:
        return False, "the goalie and set plays are run for you."
    if name.startswith("helios_"):
        return False, ("helios_* is the reference planner.")
    if name.startswith("bhv_"):
        return False, "bhv_* is not available to student code."
    return False, "not a recognised student action."


def check(agent, actions):
    """Return the actions that should actually be sent."""
    if MODE == "off":
        return actions
    keep, bad = [], []
    for a in actions:
        name = _name(a)
        ok, why = verdict(name)
        if ok:
            keep.append(a)
        else:
            bad.append((name, why))
            _violations[name] = _violations.get(name, 0) + 1

    for name, why in bad:
        agent.logger.error(f"POLICY VIOLATION: {name} is not permitted. {why}")
        from src.student.framework import trace
        trace.note(agent, "POLICY_VIOLATION", action=name, reason=why)

    if bad and MODE == "raise":
        name, why = bad[0]
        raise PolicyViolation(f"{name}: {why}")
    return keep if MODE == "strip" else actions


def summary():
    if not _violations:
        return "policy: clean"
    return "policy: " + ", ".join(f"{k}x{v}"
                                  for k, v in sorted(_violations.items()))
