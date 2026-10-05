"""
Framework-only, no need to read this.

The world model arrives from the proxy as protobuf with bare .x/.y floats.
Converted into plain objects with pyrusgeom Vector2D positions.

The snapshot is rebuilt each cycle. A piece of it kept in `self` holds last
cycle's numbers.
"""
from pyrusgeom.geom_2d import Vector2D

def _is_vec(msg) -> bool:
    return getattr(msg, "DESCRIPTOR", None) is not None \
        and msg.DESCRIPTOR.name == "RpcVector2D"


class _View:
    """One protobuf message, converted."""

    __slots__ = ("__dict__", "_msg", "_name")

    def __init__(self, msg):
        d = self.__dict__
        object.__setattr__(self, "_msg", msg)
        object.__setattr__(self, "_name", msg.DESCRIPTOR.name)
        for f in msg.DESCRIPTOR.fields:
            d[f.name] = _convert(getattr(msg, f.name))

    def __getattr__(self, name):
        # only reached on a miss, so a better message costs nothing
        kind = object.__getattribute__(self, "_name")
        fields = ", ".join(sorted(self.__dict__))
        raise AttributeError(
            f"{kind} has no field {name!r}. Available: {fields}")

    def __setattr__(self, name, value):
        raise AttributeError("the world model is read-only")

    def __dir__(self):
        return sorted(self.__dict__)

    def __repr__(self):
        return f"<{object.__getattribute__(self, '_name')}>"


def _convert(val):
    if _is_vec(val):
        return Vector2D(val.x, val.y)
    if getattr(val, "DESCRIPTOR", None) is not None:
        return _View(val)
    if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
        try:
            return [_convert(v) for v in val]
        except TypeError:
            return val
    return val


def wrap(msg):
    return _View(msg)


class AgentView:
    """What student code receives as `agent`."""

    __slots__ = ("_agent", "_wm")

    def __init__(self, agent):
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_wm", _View(agent.wm))

    @property
    def wm(self):
        return object.__getattribute__(self, "_wm")

    @property
    def strategy(self):
        return object.__getattribute__(self, "_agent").strategy

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_agent"), name)

    def __setattr__(self, name, value):
        # Deliberately not a passthrough: a stray `agent.wm = ...` would
        # corrupt framework state silently.
        raise AttributeError(
            f"cannot set {name!r} on agent. Use `self` for state that should "
            "persist across cycles for this player, or agent.memory.")


def _unwrap(obj):
    """Framework only. Used by helpers in interface.py"""
    cls = type(obj)
    if cls is AgentView:
        return object.__getattribute__(obj, "_agent")
    if cls is _View:
        return object.__getattribute__(obj, "_msg")
    return obj
