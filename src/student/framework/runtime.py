"""
src/student/framework/runtime.py -- runs student code. Framework only.

    load_student()   arm the sandbox, then import the submission
    Budget           the match-wide thinking-time budget, used as
                     `with budget:` around each student call
    BudgetView       the read-only agent.budget that students see
    arm()            installs the sandbox; load_student calls it

What a student ACTION may be is decided in policy.py, not here.

Environment:
    STUDENT_FILE        path to the submission's agent.py
                        (unset = src/student/agent.py)
    STUDENT_AGENT       which class to use (a3.py always sets it)
    SANDBOX=0           don't arm the sandbox
    BUDGET_MATCH_MS     cpu ms each player may think for the WHOLE match
                        (0 = no limit, just measure)
    BUDGET_CEILING_MS   one decision is cut off after this (wall ms)
    BUDGET_HARD_MS      a decision still running this long forfeits
    BUDGET_MAX_CUTS     a player cut off this many times is benched
"""
import importlib.util
import inspect
import json
import os
import sys
import threading
import time


class SandboxViolation(RuntimeError):
    pass


class BudgetExceeded(BaseException):
    """BaseException so that a genuine `except Exception:` doesn't swallow it.
    For anyone who tries to evade this by catching a BaseException: 
    you will be heavily penalized."""


STUDENT_FILES = set()

# loading the submission

def load_student(path=None, name=None):
    """Arm the sandbox, import the submission, return (class, class name).

    A submission is one file, agent.py.
    """
    path = path or os.environ.get("STUDENT_FILE")
    name = name or os.environ.get("STUDENT_AGENT")

    if path:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise RuntimeError(f"STUDENT_FILE={path!r} does not exist")
        arm([path])
        spec = importlib.util.spec_from_file_location("submission", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = mod
        spec.loader.exec_module(mod)
    else:
        from src.student import agent as mod
        name = name or "Agent1"

    if name:
        if not hasattr(mod, name):
            raise RuntimeError(f"{name!r} not found in {mod.__file__}")
        return getattr(mod, name), name

    agents = [n for n, c in inspect.getmembers(mod, inspect.isclass)
              if c.__module__ == mod.__name__
              and callable(getattr(c, "on_ball", None))
              and callable(getattr(c, "off_ball", None))]
    if len(agents) != 1:
        raise RuntimeError(
            f"{os.path.basename(mod.__file__)} defines {len(agents)} agent "
            f"classes {agents} - keep exactly one with on_ball and off_ball")
    return getattr(mod, agents[0]), agents[0]


# Each player gets BUDGET_MATCH_MS of cpu time for the whole match and
# spends it however it likes. The match budget is CPU time. 
# However, there 1s a 1s cap on one decision (wall clock)
#  because the server waits on wall clock. 
# A decision still running at 1.5s forfeits, and a player cut off 5 times
# is benched.
_cpu = time.process_time


def _env_ms(key, default):
    return float(os.environ.get(key) or default)


class Budget:
    """One player's budget. Students see it through BudgetView, as
    agent.budget."""

    def __init__(self, team, unum, match_ms=None, ceiling_ms=None,
                 hard_ms=None, max_cuts=None, startup_ms=0.0, out_dir=None):
        self.team, self.unum = team, unum
        self.match_ms = (_env_ms("BUDGET_MATCH_MS", 0)
                         if match_ms is None else float(match_ms))
        self.ceiling_ms = (_env_ms("BUDGET_CEILING_MS", 1000)
                           if ceiling_ms is None else float(ceiling_ms))
        self.hard_ms = (_env_ms("BUDGET_HARD_MS", 1500)
                        if hard_ms is None else float(hard_ms))
        self.max_cuts = int(_env_ms("BUDGET_MAX_CUTS", 5)
                            if max_cuts is None else max_cuts)

        self.startup_ms = startup_ms   # cpu ms spent loading the submission
        self.spent = startup_ms   # cpu ms used so far
        self.cpu = []             # cpu ms per decision
        self.wall = []            # wall ms per decision
        self.killed = 0           # decisions cut off by the ceiling
        self.broke_at = None      # decision number where the budget ran out
        self._t0 = self._c0 = None
        self._ceiling_at = None   # perf_counter time of this decision's 1s cap
        self._decisions_since_save = 0

        self._dir = out_dir or os.path.join(
            "runlogs", "timing", os.environ.get("TRACE_STAMP", "default"))
        try:
            os.makedirs(self._dir, exist_ok=True)
        except OSError:
            pass                  # _write_json then fails quietly
        self._path = os.path.join(self._dir, f"{team}_{unum}.json")

    @property
    def remaining_ms(self):
        """cpu ms left for the rest of the match (inf when unlimited)."""
        if not self.match_ms:
            return float("inf")
        running = (_cpu() - self._c0) * 1000.0 if self._c0 is not None else 0.0
        return max(0.0, self.match_ms - self.spent - running)

    @property
    def decision_remaining_ms(self):
        """wall ms left of this decision's ceiling."""
        if self._ceiling_at is None:
            return self.ceiling_ms
        return max(0.0, (self._ceiling_at - time.perf_counter()) * 1000.0)

    @property
    def exhausted(self):
        return bool(self.match_ms) and self.spent >= self.match_ms

    @property
    def benched(self):
        return bool(self.max_cuts) and self.killed >= self.max_cuts

    def __enter__(self):
        self._t0, self._c0 = time.perf_counter(), _cpu()
        # cut the decision off at the ceiling, or at budget ceiling
        limit_ms = self.ceiling_ms
        if self.match_ms:
            limit_ms = min(limit_ms, self.match_ms - self.spent)
        self._ceiling_at = self._t0 + self.ceiling_ms / 1000.0
        _watchdog.start(self._t0 + limit_ms / 1000.0,
                        self._t0 + self.hard_ms / 1000.0,
                        self._forfeit)
        return self

    def __exit__(self, exc_type, *_):
        _watchdog.stop()
        used = (_cpu() - self._c0) * 1000.0
        self.cpu.append(used)
        self.wall.append((time.perf_counter() - self._t0) * 1000.0)
        self.spent += used
        self._t0 = self._c0 = self._ceiling_at = None

        important = exc_type is BudgetExceeded
        if exc_type is BudgetExceeded:
            self.killed += 1
            if self.killed == self.max_cuts:
                sys.stderr.write(
                    f"BENCHED: {self.team} #{self.unum} was cut off "
                    f"{self.killed} times and makes no more decisions this "
                    "match.\n")
        if self.broke_at is None and self.exhausted:
            important = True
            self.broke_at = len(self.cpu)
            sys.stderr.write(
                f"BUDGET SPENT: {self.team} #{self.unum} used its whole match "
                f"budget ({self.match_ms:.0f} ms) after {self.broke_at} "
                "decisions.\n")

        # there is no end-of-match hook (the process is just killed), so
        # save often, and right away when something worth reporting happens
        self._decisions_since_save += 1
        if important or self._decisions_since_save >= 20:
            self.save()
        return False              # never swallow the student's exception

    def save(self):
        """Write this player's timing to runlogs/timing/<stamp>/ for a3.py."""
        self._decisions_since_save = 0
        _write_json(self._path, {
            "team": self.team, "unum": self.unum,
            "match_ms": self.match_ms, "ceiling_ms": self.ceiling_ms,
            "hard_ms": self.hard_ms, "max_cuts": self.max_cuts,
            "startup_ms": round(self.startup_ms, 3),
            "spent_ms": round(self.spent, 3),
            "broke_at": self.broke_at, "killed": self.killed,
            "cpu_ms": [round(v, 3) for v in self.cpu],
            "wall_ms": [round(v, 3) for v in self.wall]})

    def _forfeit(self):
        """Called just before the process exits on a stuck decision."""
        self.save()
        _write_json(os.path.join(
            self._dir, f"FORFEIT_{self.team}_{self.unum}.json"), {
            "team": self.team, "unum": self.unum,
            "reason": "decision stuck past the hard deadline",
            "hard_ms": self.hard_ms, "spent_ms": round(self.spent, 3),
            "decisions": len(self.cpu)})


class BudgetView:
    """What students get as agent.budget. Read-only, so a student can't
    hand themselves more time by resetting the counters."""

    __slots__ = ("_budget",)

    def __init__(self, budget):
        object.__setattr__(self, "_budget", budget)

    @property
    def remaining_ms(self):
        """cpu ms left for the rest of the match (inf when unlimited)."""
        return object.__getattribute__(self, "_budget").remaining_ms

    @property
    def decision_remaining_ms(self):
        """wall ms left of the 1s cap on this one decision."""
        return object.__getattribute__(self, "_budget").decision_remaining_ms

    def __setattr__(self, name, value):
        raise AttributeError("agent.budget is read-only")

    def __repr__(self):
        return (f"<budget: {self.remaining_ms:.0f}ms left in match, "
                f"{self.decision_remaining_ms:.0f}ms left this decision>")


def _write_json(path, data):
    try:
        with open(path + ".tmp", "w") as fh:
            json.dump(data, fh)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


class _Watchdog:
    """Interrupts a decision that runs past its deadline.
    Does not stop a blocking C call like time.sleep(30). 
    For those there is the hard deadline: the process exits 
    and the player forfeits.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread_id = None
        self._soft = self._hard = None
        self._fired = False
        self._on_forfeit = None
        self._loop_thread = None
        import ctypes
        self._ctypes = ctypes
        self._set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
        self._set_async_exc.argtypes = (ctypes.c_ulong, ctypes.py_object)

    def start(self, soft_deadline, hard_deadline, on_forfeit):
        with self._lock:
            self._thread_id = threading.get_ident()
            self._soft, self._hard = soft_deadline, hard_deadline
            self._fired = False
            self._on_forfeit = on_forfeit
        if self._loop_thread is None:
            self._loop_thread = threading.Thread(
                target=self._loop, daemon=True, name="a3-watchdog")
            self._loop_thread.start()

    def stop(self):
        with self._lock:
            self._thread_id = None

    def _loop(self):
        while True:
            time.sleep(0.02)
            with self._lock:
                if self._thread_id is None:
                    continue
                now = time.perf_counter()
                if not self._fired and now >= self._soft:
                    self._fired = True
                    self._set_async_exc(self._ctypes.c_ulong(self._thread_id),
                                        self._ctypes.py_object(BudgetExceeded))
                    continue
                if not (self._fired and now >= self._hard):
                    continue
                on_forfeit = self._on_forfeit
            try:
                on_forfeit()
            except Exception:
                pass
            sys.stderr.write("BUDGET FORFEIT: decision stuck in a call that "
                             "can't be interrupted. Exiting.\n")
            sys.stderr.flush()
            os._exit(3)


_watchdog = _Watchdog()

# Each player runs in its own process with no network, files, other
# processes or shared state. 
def _each(names, reason):
    return dict.fromkeys(names.split(), reason)


# Modules student code may not import, and why. Does not need to be exhaustive;
# supplement the mechanism by manually reading the student code as well.
BLOCKED_IMPORTS = {
    **_each("ctypes _ctypes cffi", "native code"),
    **_each("socket _socket ssl http urllib urllib3 requests ftplib smtplib "
            "telnetlib poplib imaplib xmlrpc webbrowser asyncio selectors "
            "grpc", "network"),
    **_each("subprocess _posixsubprocess pty tty multiprocessing "
            "_multiprocessing concurrent threading runpy",
            "processes and threads"),
    **_each("mmap shelve dbm sqlite3 posix_ipc sysv_ipc tempfile shutil "
            "fcntl signal resource", "shared state between players"),
    **_each("marshal pickle dill cloudpickle", "loading code"),
    **_each("gc inspect traceback atexit", "interpreter internals"),
    **_each("imp importlib pkgutil", "dynamic import"),
    **_each("src.behaviors src.decision_makers src.strategy src.interfaces "
            "src.utils src.communication src.student.framework base_agent "
            "player_agent coach_agent server start",
            "framework internals - use src.student.interface"),
    **_each("service_pb2 service_pb2_grpc google.protobuf",
            "raw protobuf - use the Action API"),
    **_each("scipy", "not allowed - plain Python and numpy only"),
}


def blocked_import(name):
    """-> why `import name` is not permitted, or None if it is."""
    parts = (name or "").split(".")
    for i in range(len(parts), 0, -1):
        why = BLOCKED_IMPORTS.get(".".join(parts[:i]))
        if why:
            return why
    return None


_BLOCKED_EVENTS = {
    "socket.__new__", "socket.bind", "socket.connect", "socket.connect_ex",
    "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto",
    "os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.exec",
    "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.link",
    "os.symlink", "os.truncate", "os.chmod", "os.chown", "os.chdir",
    "os.putenv", "os.unsetenv", "os.kill", "os.killpg", "os.setuid",
    "subprocess.Popen", "shutil.copyfile", "shutil.move", "shutil.rmtree",
    "tempfile.mkstemp", "pickle.find_class", "marshal.load", "marshal.loads",
    "sys.settrace", "sys.setprofile", "mmap.__new__",
    "cpython.run_file", "cpython.run_command", "cpython.run_module",
}
_OK_FROM_IMPORTLIB = {"marshal.load", "marshal.loads", "compile", "exec"}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def _stack_files():
    """Filenames on the call stack, innermost first, skipping this file."""
    f = sys._getframe(1)
    while f is not None:
        name = f.f_code.co_filename
        if not name.endswith("runtime.py"):
            yield name
        f = f.f_back


def _is_importlib(name):
    return "importlib" in name


def _student_wrote_import():
    for name in _stack_files():
        if name.startswith("<") or _is_importlib(name):
            continue
        return name in STUDENT_FILES
    return False


def _student_responsible():
    for name in _stack_files():
        if name in STUDENT_FILES:
            return True
        if _is_importlib(name):
            return False
    return False


def _raised_by_importlib():
    return _is_importlib(next(_stack_files(), ""))


def _hook(event, args):
    if event == "import":
        why = blocked_import(args[0])
        if why and _student_wrote_import():
            raise SandboxViolation(
                f"import {args[0]!r} is not permitted ({why})")
        return

    if event == "open":
        path, mode, flags = args
        writing = (any(c in mode for c in "wxa+") if mode is not None
                   else bool(flags & _WRITE_FLAGS))
        if writing and _student_responsible():
            raise SandboxViolation(
                f"writing {path!r} is not permitted. To record your "
                "reasoning, use note().")
        return

    if event.startswith("ctypes."):
        if _student_responsible():
            raise SandboxViolation(event + ": native code is not permitted")
        return

    if event in _OK_FROM_IMPORTLIB and _raised_by_importlib():
        return

    if event in _BLOCKED_EVENTS and _student_responsible():
        raise SandboxViolation(event + " is not permitted")


_armed = False


def arm(student_files=()):
    """Mark these files as student code and install the hook (once)."""
    global _armed
    STUDENT_FILES.update(os.path.abspath(p) for p in student_files if p)
    if _armed or os.environ.get("SANDBOX", "1") == "0":
        return

    sys.dont_write_bytecode = True
    sys.addaudithook(_hook)
    _armed = True
