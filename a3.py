"""
a3 - the assignment harness, in one file.

    python3 a3.py doctor              is this machine set up right
    python3 a3.py match MINE FULL --right full      play the reference team
    python3 a3.py why                 what did my players do
    python3 a3.py watch               see it in a browser


Match settings live in one file in rcssserver's own format, read here and
passed to the server with --include=. Override one setting for one run
with `-o server::half_time=30`.

Every command takes --help. Python 3.8+, standard library only.
"""
import argparse, ast, glob, gzip, json, os, re, shutil
import socket, subprocess, sys, time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "a3.py"
if not PROG.endswith(".py") and PROG != "a3":
    PROG = "a3.py"
RUN = "python3 " + PROG

LOGS = os.path.join(HERE, "logs")
RUNLOGS = os.path.join(HERE, "runlogs")
TRACE_ROOT = os.path.join(RUNLOGS, "trace")
DEFAULT_CONF = "base.conf"
if sys.platform == "darwin":
    SERVER_BIN = os.path.join(HERE, "scripts", "rcssserver", "rcssserver")
else:
    SERVER_BIN = os.path.join(
        HERE, "scripts", "rcssserver", "squashfs-root", "AppRun"
    )
PROXY_BIN = os.path.join(HERE, "scripts", "proxy", "sample_player")
GIBBS = os.path.join(HERE, "gibbs")


FORMATIONS = {11: "4-3-3-cyrus-base", 7: "4-3-3-7aside", 5: "4-3-3-5aside"}

COLOR_MAP = {"dim": "\033[2m", "b": "\033[1m", "r": "\033[31m", "g": "\033[32m",
     "y": "\033[33m", "c": "\033[36m", "x": "\033[0m"}
if not sys.stdout.isatty():
    COLOR_MAP = {k: "" for k in COLOR_MAP}


def die(msg, code=1):
    sys.stderr.write("!! " + msg + "\n"); raise SystemExit(code)


def say(msg):
    sys.stderr.write("   " + msg + "\n")


# config formatting for rcsserver versus ours
CONF_LINE = re.compile(r"^\s*([A-Za-z_][\w:]*)\s*=\s*(.*?)\s*(?:#.*)?$")
A3_LINE = re.compile(r"^\s*#\s*(a3::\w+)\s*=\s*(.*?)\s*$")


def read_conf(path):
    conf = {}
    if not os.path.exists(path):
        return conf
    for line in open(path):
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            m = A3_LINE.match(line)
            if m:
                conf[m.group(1)] = m.group(2).strip().strip("'\"")
            continue
        m = CONF_LINE.match(line)
        if m:
            conf[m.group(1)] = m.group(2).strip().strip("'\"")
    return conf


def team_spec(spec):
    """
    Specifying file path and agent class name.

        full                    HELIOS reference
        subs/x/agent.py         agent.py file, class Agent1
        subs/x/agent.py:Alpha   agent.py file, class Alpha
        :Alpha                  src/student/agent.py, class Alpha
        <omitted>               src/student/agent.py, class Agent1

    -> (mode, file, cls)
    """
    if not spec:
        return "starter", None, "Agent1"
    if spec.strip().lower() == "full":
        return "full", None, None
    path, cls = spec, "Agent1"
    if ":" in spec:
        head, tail = spec.rsplit(":", 1)
        if tail.isidentifier():
            path, cls = head, tail
    path = path.strip() or None
    if path:
        p = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(p):
            p = os.path.join(p, "agent.py")
        if not os.path.isfile(p):
            die(f"no such submission: {path}\n     a team spec is `full`, "
                "a path to agent.py, or `:ClassName`")
        path = p
    return "starter", path, cls


def team_size(a, conf=None):
    if getattr(a, "size", None):
        return int(a.size)
    if conf is None:
        conf = read_conf(os.path.join(HERE, getattr(a, "conf", DEFAULT_CONF)))
    return int(conf_get(conf, "a3::team_size", 0) or 0) or 11


def conf_get(conf, key, default=None):
    if key in conf:
        return conf[key]
    tail = key.split("::")[-1]
    for k, v in conf.items():
        if k.split("::")[-1] == tail:
            return v
    return default


# RUN MATCH
# AppImages are extracted, not FUSE-mounted
# no_proxy set
# --use-different-rpc-port so every agent is its own process
# wait for every python brain before kickoff
# every output path carries a per-match stamp
# game_log_fixed_name is passed with single quotes inside the
#   value so rcsserver parses it as a string

def _popen(argv, env, out_path):
    fh = open(out_path, "wb")
    return subprocess.Popen(argv, env=env, cwd=HERE, stdout=fh,
                            stderr=subprocess.STDOUT), fh


def _port_busy(port):
    """Return False if the port isn't available."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def _preflight(a):
    """Catch errors in match command."""
    if a.left == a.right:
        die("the two team names must differ")
    if a.dry_run:
        return
    for path, what in ((SERVER_BIN, "server"), (PROXY_BIN, "proxy")):
        if not (os.path.exists(path) and os.access(path, os.X_OK)):
            die(f"no {what} at {path}\n     run `{RUN} doctor`")
    busy = [q for q in (a.port, a.port + 1, a.port + 2) if _port_busy(q)]
    if busy:
        ports = ", ".join(str(q) for q in busy)
        die(f"udp port(s) {ports} already in use - another match is "
            "running, or the last one did not clean up\n"
            f"     try -p {a.port + 10}")


class _Setup:
    """All flag/config context."""
    __slots__ = ("conf", "conf_path", "size", "half", "stamp",
                 "server_log", "rcg")


def _settings(a):
    s = _Setup()
    s.conf_path = os.path.join(HERE, a.conf)
    s.conf = read_conf(s.conf_path)
    for ov in a.override:
        if "=" not in ov:
            die(f"-o wants key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        s.conf[k] = v

    s.size = team_size(a, s.conf)
    s.half = a.half or conf_get(s.conf, "server::half_time")
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    s.stamp = a.stamp or f"{now}_{a.left}_vs_{a.right}"

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RUNLOGS, exist_ok=True)
    s.server_log = os.path.join(RUNLOGS, s.stamp + "_server.log")
    s.rcg = os.path.join(LOGS, s.stamp + ".rcg")
    return s


def _dedupe_args(sargs):
    seen = {}
    for arg in sargs:
        seen[arg.split("=", 1)[0]] = arg
    out = []
    for arg in sargs:
        key = arg.split("=", 1)[0]
        if key in seen:
            out.append(seen.pop(key))
    return out


def _server_argv(a, s, rest):
    sargs = [
        "--server::auto_mode=true",
        f"--server::port={a.port}",
        f"--server::coach_port={a.port + 1}",
        f"--server::olcoach_port={a.port + 2}",
        "--server::game_log_dir=" + LOGS,
        "--server::text_log_dir=" + LOGS,
        "--server::nr_extra_halfs=0",
        "--server::penalty_shoot_outs=false",
        "--server::game_log_fixed=true",
        f"--server::game_log_fixed_name='{s.stamp}'",
        "--server::game_log_dated=false",
        "--server::fullstate_l=true",
        "--server::fullstate_r=true",
    ]
    if os.path.exists(s.conf_path):
        sargs.insert(0, "--include=" + s.conf_path)
    if s.half:
        sargs.append(f"--server::half_time={s.half}")
    for ov in a.override:
        sargs.append("--" + ov)
    sargs += list(rest)
    return _dedupe_args(sargs)


def _announce(a, s, teams, sargs):
    (lname, lmode, lcls, _), (rname, rmode, rcls, _) = teams
    lwho = "full" if lmode == "full" else lcls
    rwho = "full" if rmode == "full" else rcls
    print(f"== {lname}({lwho}) vs {rname}({rwho})  "
          f"({s.size}-a-side, port {a.port})")
    if not a.size and not conf_get(s.conf, "a3::team_size"):
        say(f"no a3::team_size in {a.conf} - defaulting to 11. "
            "Set it, or pass -n.")
    if a.dry_run:
        print(" ".join([SERVER_BIN] + sargs))
        return True
    return False


def _base_env():
    env = dict(os.environ)
    env["no_proxy"] = env["NO_PROXY"] = "localhost,127.0.0.1,::1"
    return env


class _Procs:
    def __init__(self):
        self.procs, self.handles, self.output_paths = [], [], {}

    def start(self, argv, env, out_path):
        p, fh = _popen(argv, env, out_path)
        self.procs.append(p)
        self.handles.append(fh)
        return p

    def cleanup(self, *_):
        say("cleaning up")
        for p in self.procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(0.5)
        for p in self.procs:
            try:
                p.kill()
            except Exception:
                pass
        for h in self.handles:
            try:
                h.close()
            except Exception:
                pass


def _team_env(env, a, s, mode, cls, sub, form):
    tenv = dict(env)
    tenv.update({
        "USE_STARTER": "0" if mode == "full" else "1",
        "TEAM_SIZE": str(s.size),
        "STUDENT_AGENT": cls or "Agent1",
        "FORMATION": form,
        "TRACE_STAMP": s.stamp,
        "TRACE": "1" if a.trace else "0",
        "POLICY": a.policy,
        "SANDBOX": "0" if a.no_sandbox else "1",
        # whole-match thinking time per player, and the hang cut-off
        "BUDGET_MATCH_MS": str(float(
            conf_get(s.conf, "a3::budget_match_s", 0) or 0) * 1000.0),
        "BUDGET_CEILING_MS": str(conf_get(s.conf, "a3::ceiling_ms", 1000)),
        "BUDGET_HARD_MS": str(conf_get(s.conf, "a3::hard_ms", 1500)),
        "BUDGET_MAX_CUTS": str(conf_get(s.conf, "a3::max_cuts", 5)),
    })
    if sub and mode != "full":
        tenv["STUDENT_FILE"] = sub
    return tenv


def _spawn_teams(processes, env, a, s, teams):
    form = FORMATIONS.get(s.size, FORMATIONS[11])
    for name, mode, cls, sub in teams:
        out = os.path.join(RUNLOGS, f"{s.stamp}_{name}.out")
        processes.output_paths[name] = out
        processes.start([sys.executable, "start.py", "-t", name,
                   "--use-different-rpc-port",
                   "--server-port", str(a.port),
                   "--log-dir", os.path.join(
                       RUNLOGS, f"{s.stamp}_{name}")],
                  _team_env(env, a, s, mode, cls, sub, form), out)


def _wait_for_brains(processes, want, tries=30):
    """Kick off only once every brain has registered."""
    sys.stderr.write(f"-- waiting for {want} brains ")
    up = 0
    for _ in range(tries):
        up = 0
        for out in processes.output_paths.values():
            try:
                up += open(out, errors="replace").read().count("Agent created")
            except OSError:
                pass
        if up >= want:
            break
        sys.stderr.write("."); sys.stderr.flush()
        time.sleep(1)
    sys.stderr.write(f" {up}/{want} up\n")
    if up >= want:
        return True
    say(f"only {up}/{want} brains came up - the match would be invalid")
    for nm, out in processes.output_paths.items():
        sys.stderr.write(f"---- {nm} ----\n")
        try:
            sys.stderr.write("".join(
                open(out, errors="replace").readlines()[-20:]))
        except OSError:
            sys.stderr.write("(no output)\n")
    return False


def _wait_for_end(srv, rcg, end_at_iters=6, step=5):
    """Watch replay size, call match done once it stops changing."""
    sys.stderr.write("-- playing ")
    last, replay_stable_for, t0 = -1, 0, time.time()
    while replay_stable_for < end_at_iters:
        time.sleep(step)
        try:
            sz = os.path.getsize(rcg)
        except OSError:
            sz = 0
        if sz == last:
            replay_stable_for += 1
        else:
            replay_stable_for = 0
            sys.stderr.write("."); sys.stderr.flush()
        last = sz
        if srv.poll() is not None:
            break
    sys.stderr.write(" done\n")
    return t0, time.time()


def _result(rcg, outs):
    """Validate the replay, then void it if any brain printed a traceback."""
    v = validate(rcg)
    for nm, out in outs.items():
        try:
            if "Traceback (most recent call last)" in open(
                    out, errors="replace").read():
                v["ok"] = False
                v["problems"].append(f"EXCEPTION[{nm}]")
        except OSError:
            pass
    return v


def _budget(s):
    """How much planning time the match actually cost."""
    rows = read_timing(os.path.join(TIMING_ROOT, s.stamp))
    if not rows:
        return
    budget = float(conf_get(s.conf, "a3::budget_match_s", 0) or 0) * 1000.0
    for team, st in sorted(summarise(rows).items()):
        spend = st["spend_ms"]
        cut = f"  {st['killed']} cut short" if st["killed"] else ""
        if budget:
            pct = 100.0 * spend / budget
            col = COLOR_MAP["r"] if pct > 100 else (COLOR_MAP["y"] if pct > 80 else "")
            print(f"   planning   : {col}{team:<9}{pct:3.0f}% of "
                  f"{budget:.0f}ms{COLOR_MAP['x']}  worst #{st['spend_unum']}{cut}")
        else:
            print(f"   planning   : {team:<9}{spend:.0f}ms   "
                  f"worst #{st['spend_unum']}{cut}")


def _report(a, s, processes, t0, t1):
    print()
    print(f"   wall clock : {int(t1 - t0)}s")
    if os.path.exists(s.rcg):
        print("   replay     : " + os.path.basename(s.rcg))
        v = _result(s.rcg, processes.output_paths)
        tag = (COLOR_MAP["g"] + "valid" + COLOR_MAP["x"]) if v["ok"] else \
              (COLOR_MAP["r"] + "INVALID: " + ", ".join(v["problems"]) + COLOR_MAP["x"])
        print(f"   result     : {v['left']} {v['lscore']}-{v['rscore']} "
              f"{v['right']}   [{tag}]")
    else:
        print(f"   replay     : <none produced - see {s.server_log}>")

    for f in forfeits(s.stamp):
        print(f"   {COLOR_MAP['r']}FORFEIT    : {f['team']} #{f['unum']} never "
              f"returned from a decision (hard deadline "
              f"{f.get('hard_ms', 0):.0f}ms){COLOR_MAP['x']}")
    _budget(s)
    print()
    print(f"   watch it   : {RUN} watch")
    if a.trace:
        print(f"   why        : {RUN} why -t {a.left}   (or -t {a.right})")


def cmd_match(a, rest):
    _preflight(a)
    s = _settings(a)
    sargs = _server_argv(a, s, rest)

    lmode, lfile, lcls = team_spec(a.left_spec)
    rmode, rfile, rcls = team_spec(a.right_spec)
    teams = ((a.left, lmode, lcls, lfile), (a.right, rmode, rcls, rfile))

    if _announce(a, s, teams, sargs):
        return 0 # dry run

    env = _base_env()
    processes = _Procs()
    srv = processes.start([SERVER_BIN] + sargs, env, s.server_log)
    try:
        time.sleep(3)
        if srv.poll() is not None:
            say("server died:")
            sys.stderr.write("".join(open(s.server_log, errors="replace")
                                     .readlines()[:20]))
            return 1
        _spawn_teams(processes, env, a, s, teams)
        if not _wait_for_brains(processes, s.size * 2 + 2):
            return 1
        t0, t1 = _wait_for_end(srv, s.rcg)
    finally:
        processes.cleanup()

    _report(a, s, processes, t0, t1)
    return 0


# TRACE 

def _is_match_dir(d):
    return bool(glob.glob(os.path.join(d, "*", "*.jsonl")))


def trace_matches(root=TRACE_ROOT):
    root = os.environ.get("TRACE_DIR", root)
    if not os.path.isdir(root):
        return []
    subs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]
    if _is_match_dir(root) and not any(_is_match_dir(d) for d in subs):
        return [root]
    return sorted([d for d in subs if _is_match_dir(d)],
                  key=os.path.getmtime, reverse=True)


def teams_in(match):
    return sorted(d for d in os.listdir(match)
                  if glob.glob(os.path.join(match, d, "*.jsonl")))


def pick_match(team=None, name=None):
    for m in trace_matches():
        if name and name not in os.path.basename(m):
            continue
        if team and team not in teams_in(m):
            continue
        return m
    what = []
    if team:
        what.append(f"team {team!r}")
    if name:
        what.append(f"name containing {name!r}")
    filters = (" with " + " and ".join(what)) if what else ""
    die(f"no traced match{filters} under {TRACE_ROOT}.\n     Run with "
        f"--trace, or try `{RUN} why --list`.")


def load_trace(match, team=None):
    by_cycle = defaultdict(list)
    for path in glob.glob(os.path.join(match, team or "*", "*.jsonl")):
        t = os.path.basename(os.path.dirname(path))
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            row["team"] = t
            by_cycle[row["c"]].append(row)
    for rows in by_cycle.values():
        rows.sort(key=lambda r: (r["team"], r["unum"]))
    return by_cycle


def _fmt_action(a):
    kind = a.get("kind", "?")
    args = {k: v for k, v in a.items() if k != "kind"}
    if not args:
        return kind
    bits = []
    for k, v in args.items():
        bits.append(f"{k}=({v[0]},{v[1]})"
                    if isinstance(v, list) and len(v) == 2 else
                    f"{k}={v}")
    return f"{kind}({', '.join(bits)})"


def show_cycle(cycle, rows):
    if not rows:
        return
    ball = rows[0]["ball"]
    print(f"\n{COLOR_MAP['b']}cycle {cycle}{COLOR_MAP['x']}   {COLOR_MAP['dim']}ball "
          f"({ball[0]}, {ball[1]})   mode {rows[0]['mode']}{COLOR_MAP['x']}")
    for r in rows:
        acts = " ; ".join(_fmt_action(a) for a in r.get("actions", [])) or \
            f"{COLOR_MAP['r']}NOTHING (player will coast to a stop){COLOR_MAP['x']}"
        flag = f"{COLOR_MAP['g']}*{COLOR_MAP['x']}" if r.get("kickable") else " "
        reach = r.get("reach", [0, 0, 0])
        opp = r.get("nearest_opp")
        left = (f"  left {r['left_ms']:.0f}ms"
                if "left_ms" in r else "")
        near = f"opp{opp[0]}@{opp[1]}m" if opp else "-"
        x, y = r["pos"]
        print(f"  {flag} {COLOR_MAP['c']}{r['team']}#{r['unum']:<2}{COLOR_MAP['x']} "
              f"({x:>6.1f},{y:>6.1f}) {COLOR_MAP['dim']}reach me/tm/opp "
              f"{reach[0]:>2}/{reach[1]:>2}/{reach[2]:>2}  {near:<12} "
              f"{r.get('ms', 0):>5.1f}ms cpu{left}{COLOR_MAP['x']}")
        print("      " + acts)
        for n in r.get("notes", []):
            extra = " ".join(f"{k}={v}"
                             for k, v in n.items() if k != "note")
            print(f"      {COLOR_MAP['y']}. {n['note']} {extra}{COLOR_MAP['x']}")


def _labels(row):
    return [n.get("note") for n in row.get("notes", [])]


def t_grep(by_cycle, needle, limit):
    hits = [c for c, rows in by_cycle.items()
            if needle.lower() in json.dumps(rows).lower()]
    print(f"{len(hits)} cycle(s) match {needle!r}")
    for c in sorted(hits)[:limit]:
        show_cycle(c, by_cycle[c])


def t_slow(by_cycle, ms, limit):
    """`ms` is CPU time. Wall time is in the trace too but is not worth
    thresholding on a shared box - see the note above TIMING_ROOT."""
    hits = [(c, max(r.get("ms", 0) for r in rows))
            for c, rows in by_cycle.items()
            if any(r.get("ms", 0) > ms for r in rows)]
    hits.sort(key=lambda h: -h[1])
    print(f"{len(hits)} cycle(s) over {ms}ms")
    for c, worst in hits[:limit]:
        print(f"  cycle {c:>5}  worst {worst:.1f}ms")


def t_chase(by_cycle, limit):
    hits = []
    for c, rows in by_cycle.items():
        per = Counter(r["team"] for r in rows
                      if any(a.get("kind") == "body_intercept"
                             for a in r.get("actions", [])))
        hits += [(c, t, n) for t, n in per.items() if n >= 2]
    print(f"{len(hits)} cycle(s) with 2+ teammates intercepting")
    for c, t, n in sorted(hits)[:limit]:
        print(f"  cycle {c:>5}  {t}: {n} players intercepting")
    if hits:
        print(f"\n{COLOR_MAP['dim']}Every player decides independently{COLOR_MAP['x']}")
        show_cycle(hits[0][0], by_cycle[hits[0][0]])


def t_breakdown(by_cycle):
    per = defaultdict(Counter)
    squad = defaultdict(set)
    for rows in by_cycle.values():
        for r in rows:
            squad[r["team"]].add(r["unum"])
            for lab in _labels(r):
                per[r["team"]][lab] += 1
    if not per:
        print("no labelled decisions. Your agent needs to call\n"
              "    from src.student.interface import note\n"
              '    note(agent, "chase")')
        return
    for team in sorted(per):
        total = sum(per[team].values())
        # only one player can be on the ball, so formation has an arithmetic
        # floor.
        n = max(2, len(squad[team]))
        min_formation = 100.0 * (n - 1) / n
        print(f"\n{COLOR_MAP['b']}{team}{COLOR_MAP['x']}  ({total} decisions)")
        for lab, pcount in per[team].most_common():
            pct = 100.0 * pcount / total
            mark = f"  {COLOR_MAP['y']}<- standing around{COLOR_MAP['x']}" \
                if lab == "formation" and pct > min_formation + 8 else ""
            bar = "#" * int(pct / 2)
            print(f"   {lab:<14} {pcount:>6}  {pct:>5.1f}%  {bar}{mark}")
        print(f"   {COLOR_MAP['dim']}one player holds the ball at a time, so "
              f"formation sits near {min_formation:.0f}% if no other off-ball instr given.{COLOR_MAP['x']}")


def t_stack(by_cycle, label):
    per = defaultdict(Counter)
    for rows in by_cycle.values():
        count = Counter()
        for r in rows:
            if label in _labels(r):
                count[r["team"]] += 1
        for team, n in count.items():
            per[team][n] += 1
    if not per:
        print(f"nobody was labelled {label!r}")
        return
    for team in sorted(per):
        hist = per[team]
        stacked = sum(v for k, v in hist.items() if k >= 2)
        print(f"\n{COLOR_MAP['b']}{team}{COLOR_MAP['x']}  cycles with N players on {label!r}:")
        for n in sorted(hist):
            warn = f"  {COLOR_MAP['r']}<- stacked{COLOR_MAP['x']}" if n >= 2 else ""
            print(f"   {n} player(s): {hist[n]:>5}{warn}")
        if stacked:
            print(f"   {COLOR_MAP['y']}{stacked} cycle(s) with 2+ on the same job, "
                  f"worst {max(hist)}{COLOR_MAP['x']}")


def t_possession(by_cycle):
    per_team = defaultdict(lambda: defaultdict(list))
    for c, rows in by_cycle.items():
        for r in rows:
            per_team[r["team"]][c].append(r)
    for team in sorted(per_team):
        cycles = per_team[team]
        n = len(cycles)
        ours = sum(any(r.get("kickable") for r in rs) for rs in cycles.values())
        theirs = sum(any(r.get("reach", [0, 0, 1])[2] == 0 for r in rs)
                     for rs in cycles.values())
        loose = max(0, n - ours - theirs)
        print(f"{team:<12} on the ball {100.0 * ours / n:5.1f}%   "
              f"opponent on the ball {100.0 * theirs / n:5.1f}%   "
              f"loose {100.0 * loose / n:5.1f}%   ({n} cycles)")


def cmd_why(a, rest):
    if a.list:
        ms = trace_matches()
        if not ms:
            print("no traced matches under " + TRACE_ROOT)
        for m in ms:
            print(f"  {os.path.basename(m):<48} {', '.join(teams_in(m))}")
        return 0

    match = pick_match(team=a.team, name=a.match)
    teams = ", ".join(teams_in(match))
    print(f"{COLOR_MAP['dim']}match: {os.path.basename(match)}  teams: {teams}"
          f"{COLOR_MAP['x']}")

    by_cycle = load_trace(match, a.team)
    if not by_cycle:
        die("that match has no trace rows yet - still starting?")

    if a.grep:
        return t_grep(by_cycle, a.grep, a.limit) or 0
    if a.slow is not None:
        return t_slow(by_cycle, a.slow, a.limit) or 0
    if a.chase:
        return t_chase(by_cycle, a.limit) or 0
    if a.stack:
        return t_stack(by_cycle, a.stack) or 0
    if a.possession:
        return t_possession(load_trace(match)) or 0   # needs both teams
    if a.cycles:
        lo = a.cycles[0]
        hi = a.cycles[1] if len(a.cycles) > 1 else lo
        for c in range(lo, hi + 1):
            if c in by_cycle:
                show_cycle(c, by_cycle[c])
        return 0
    return t_breakdown(by_cycle) or 0


# RCG PARSING 
GOALIE_BIT = 0x8
RE_SHOW = re.compile(r"^\(show\s+(-?\d+)\s+(.*)\)\s*$")
RE_TEAM = re.compile(r'^\(team\s+-?\d+\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)')
RE_PLAYMODE = re.compile(r"^\(playmode\s+-?\d+\s+(\S+?)\)")
RE_BALL = re.compile(r"\(\(?b\)?\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)"
                     r"\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\)")
RE_PLAYER = re.compile(
    r"\(\((l|r)\s+(\d+)\)\s+(\d+)\s+(0x[0-9a-fA-F]+|\d+)\s+"
    r"(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+"
    r"(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)")
RE_STAMINA = re.compile(r"\(s\s+(-?[\d.eE+-]+)")
RE_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}|\d{14})")
MANIFEST = ".manifest.json"


def _open_rcg(path):
    return (gzip.open(path, "rt", errors="replace") if path.endswith(".gz")
            else open(path, "rt", errors="replace"))


def _r2(v):
    return round(float(v), 2)


def walk_rcg(path):
    """Yield ('team'|'playmode'|'show'|'param', payload) for a log."""
    with _open_rcg(path) as fin:
        header = fin.readline().strip()
        if not header.startswith("ULG"):
            raise ValueError(f"{path}: not a text rcg log (got {header!r}). "
                             "Set --server::game_log_version=6.")
        for line in fin:
            if line.startswith("(server_param"):
                yield "param", line
            elif line.startswith("(team"):
                m = RE_TEAM.match(line)
                if m:
                    yield "team", (m.group(1), m.group(2),
                                   int(m.group(3)), int(m.group(4)))
            elif line.startswith("(playmode"):
                m = RE_PLAYMODE.match(line)
                if m:
                    yield "playmode", m.group(1)
            else:
                m = RE_SHOW.match(line.rstrip("\n"))
                if m:
                    yield "show", (int(m.group(1)), m.group(2))


def parse_players(body):
    """-> {(side, unum): (state, x, y, body, neck, stamina)} for connected only."""
    out = {}
    for chunk in body.split("((")[1:]:
        mp = RE_PLAYER.search("((" + chunk)
        if not mp:
            continue
        side, unum, _t, state, x, y, _vx, _vy, bd, nk = mp.groups()
        st = int(state, 16) if state.startswith("0x") else int(state)
        if st == 0:
            continue                          # not connected
        ms = RE_STAMINA.search(chunk)
        out[(side, int(unum))] = (st, x, y, bd, nk,
                                  float(ms.group(1)) if ms else 0.0)
    return out


# VALIDATE if the match ran properly 

def validate(path, expect_cycles=None, expect_players=None):
    v = {"file": path, "ok": True, "problems": [], "cycles": 0,
         "left": "?", "right": "?", "lscore": 0, "rscore": 0,
         "expect_players": expect_players, "half_time": None,
         "first_full": {"l": None, "r": None},
         "last_full": {"l": None, "r": None},
         "max_players": {"l": 0, "r": 0}, "final_players": {"l": 0, "r": 0}}
    counts = {"l": [], "r": []}
    cycles = []
    try:
        for kind, payload in walk_rcg(path):
            if kind == "param":
                m = re.search(r"\(half_time (-?[\d.]+)\)", payload)
                if m:
                    v["half_time"] = float(m.group(1))
                m = re.search(r"\(simulator_step (-?[\d.]+)\)", payload)
                if m:
                    v["step"] = float(m.group(1))
            elif kind == "team":
                v["left"], v["right"], v["lscore"], v["rscore"] = payload
            elif kind == "show":
                cyc, body = payload
                ps = parse_players(body)
                cycles.append(cyc)
                for side in ("l", "r"):
                    counts[side].append(
                        sum(1 for (s, _u) in ps if s == side))
    except (ValueError, OSError) as e:
        v["ok"] = False
        v["problems"].append(f"UNREADABLE: {e}")
        return v

    v["cycles"] = len(cycles)
    if not cycles:
        v["ok"] = False
        v["problems"].append("NO_SHOW_DATA (match never kicked off)")
        return v

    if v["left"] == v["right"]:
        v["ok"] = False
        v["problems"].append("DUPLICATE_NAMES (server merged both teams)")

    want = expect_players
    if want is None:
        want = max(max(counts["l"] or [0]), max(counts["r"] or [0]))
        v["expect_players"] = want

    for side in ("l", "r"):
        seq = counts[side]
        v["max_players"][side] = max(seq) if seq else 0
        v["final_players"][side] = seq[-1] if seq else 0
        full = [i for i, n in enumerate(seq) if n >= want]
        if full:
            v["first_full"][side] = cycles[full[0]]
            v["last_full"][side] = cycles[full[-1]]
        name = v["left"] if side == "l" else v["right"]
        if not full:
            v["ok"] = False
            v["problems"].append(
                f"NEVER_FULL[{name}] "
                f"(peaked at {v['max_players'][side]}/{want})")
            continue

        tail = len(seq) - 1 - full[-1]
        if tail > 0.02 * len(seq) and seq[-1] < want:
            v["ok"] = False
            v["problems"].append(
                f"DISCONNECT[{name}] (dropped to {seq[-1]}/{want} at cycle "
                f"{cycles[full[-1]]}, {tail} cycles before the end)")

        if full[0] > 0.05 * len(seq):
            v["ok"] = False
            v["problems"].append(
                f"LATE_START[{name}] (not full until cycle {cycles[full[0]]})")

    exp = expect_cycles
    if exp is None and v["half_time"]:
        exp = int(v["half_time"] * 1000.0 / v.get("step", 100.0)) * 2
    if exp and v["cycles"] < 0.9 * exp:
        v["ok"] = False
        v["problems"].append(
            f"ABORTED ({v['cycles']} of ~{exp} cycles)")
    v["expect_cycles"] = exp
    return v


def _spark(seq, width=48):
    if not seq:
        return ""
    bars = " .:-=+*#"
    hi = max(seq) or 1
    step = max(1, len(seq) // width)
    return "".join(bars[min(len(bars) - 1,
                            int(1.0 * max(seq[i:i + step]) / hi * (len(bars) - 1)))]
                   for i in range(0, len(seq), step))


def cmd_validate(a, rest):
    paths = []
    for p in a.logs:
        paths += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    results = [validate(p, a.expect_cycles, a.expect_players) for p in paths]
    if a.json:
        print(json.dumps(results, indent=2))
        return 0 if all(r["ok"] for r in results) else 1
    for v in results:
        tag = (COLOR_MAP["g"] + "VALID  " + COLOR_MAP["x"]) if v["ok"] else (COLOR_MAP["r"] + "INVALID" + COLOR_MAP["x"])
        print(f"{tag} {os.path.basename(v['file'])}")
        of = (f" of ~{v['expect_cycles']}" if v.get("expect_cycles")
              else "")
        print(f"        {v['left']} {v['lscore']}-{v['rscore']} {v['right']}"
              f"   {v['cycles']} cycles{of}")
        for p in v["problems"]:
            print(f"        {COLOR_MAP['r']}! {p}{COLOR_MAP['x']}")
        if a.verbose:
            print(f"        expected {v['expect_players']} players a side")
            for side in ("l", "r"):
                print(f"        {side}: max {v['max_players'][side]} "
                      f"final {v['final_players'][side]} "
                      f"full {v['first_full'][side]}..{v['last_full'][side]}")
    bad = [r for r in results if not r["ok"]]
    if len(results) > 1:
        print(f"\n{len(results) - len(bad)}/{len(results)} valid")
    return 1 if bad else 0


# WATCH replay
INDEX = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>matches</title>
<style>
 body{{font:15px/1.6 system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 20px}}
 h1{{font-size:18px;font-weight:600;margin-bottom:4px}}
 p{{color:#666;margin-top:0}} ul{{list-style:none;padding:0}}
 li{{padding:8px 0;border-bottom:1px solid #eee}}
 a{{color:#0645ad;text-decoration:none}} a:hover{{text-decoration:underline}}
 .m{{color:#999;font-size:13px;margin-left:8px}}
</style></head><body>
<h1>matches</h1><p>{n} replay(s). Large files load synchronously &mdash; the
tab will hang for a few seconds.</p>
<ul>{items}</ul></body></html>
"""


def to_replay(rcg_path, out_path):
    """.rcg -> GIBBS .replay. Returns teams in PITCH order and final score."""
    left = right = "unknown"
    lscore = rscore = 0
    playmode = "before_kick_off"
    cycles = 0
    goal_width = 14.02
    with open(out_path, "w") as fout:
        header_written = False
        pending = []
        for kind, payload in walk_rcg(rcg_path):
            if kind == "param":
                m = re.search(r"\(goal_width ([\d.]+)\)", payload)
                if m:
                    goal_width = float(m.group(1))
            elif kind == "team":
                left, right, lscore, rscore = payload
            elif kind == "playmode":
                playmode = payload
            elif kind == "show":
                cycle, body = payload
                if not header_written:
                    fout.write(f'T "{left}" "{right}" {goal_width}\n')
                    header_written = True
                out = [f"S {cycle} {playmode} {lscore} {rscore}"]
                mb = RE_BALL.search(body)
                out.append("b " + " ".join(str(_r2(x)) for x in mb.groups())
                           if mb else "b 0 0 0 0")
                ps = parse_players(body)
                for side in ("l", "r"):
                    for u in range(1, 12):
                        p = ps.get((side, u))
                        if p:
                            st, x, y, bd, nk, sta = p
                            tag = side.upper() if st & GOALIE_BIT else side
                            out.append(
                                f"{tag} {u} {_r2(x)} {_r2(y)} {_r2(bd)} "
                                f"{_r2(nk)} #{round(sta, 1)}")
                        else:
                            out.append(f"{side} {u} -60 -45 0 0 #0")
                pending.append("\n".join(out))
                cycles += 1
                if len(pending) >= 500:
                    fout.write("\n".join(pending) + "\n")
                    pending = []
        if pending:
            fout.write("\n".join(pending) + "\n")
    return {"cycles": cycles, "left": left, "right": right,
            "lscore": lscore, "rscore": rscore}


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]", "-", name) or "unknown"


def replay_name(rcg_name, rcg_path, info):
    """<when>_<LEFT>_<l>-<r>_<RIGHT>.replay, from the log's own data."""
    m = RE_STAMP.match(rcg_name)
    when = m.group(1) if m else time.strftime(
        "%Y-%m-%d_%H-%M-%S", time.localtime(os.path.getmtime(rcg_path)))
    left, right = _safe(info["left"]), _safe(info["right"])
    return (f"{when}_{left}_{info['lscore']}-{info['rscore']}_{right}"
            ".replay")


def patch_player_js(gibbs):
    js = os.path.join(gibbs, "scripts", "player.js")
    if not os.path.exists(js):
        die(f"{js} not found - is --gibbs pointing at the replay viewer?")
    src = open(js).read()
    if 'type: "post"' in src:
        shutil.copy(js, js + ".orig")
        open(js, "w").write(src.replace('type: "post"', 'type: "get"'))
        say("patched player.js (post -> get)")


def write_index(gibbs):
    rep = os.path.join(gibbs, "replays")
    files = sorted((f for f in os.listdir(rep) if f.endswith(".replay")),
                   reverse=True)
    items = "".join(
        f'<li><a href="play.html?f=./replays/{f}">{f}</a>'
        f'<span class="m">{os.path.getsize(os.path.join(rep, f)) / 1e6:.1f}'
        ' MB</span></li>' for f in files)
    open(os.path.join(gibbs, "index.html"), "w").write(
        INDEX.format(n=len(files), items=items or "<li>none yet</li>"))
    php = open(os.path.join(gibbs, "play.php")).read()
    opts = "".join(f'<option value="./replays/{f}">{f}</option>'
                   for f in files)
    html = re.sub(r"<\?php.*?\?>", opts, php, flags=re.S)
    html = html.replace("</body>", """
<script>
(function(){
  var f = new URLSearchParams(location.search).get('f');
  if (!f) return;
  var s = document.getElementById('fileSelect');
  if (s) { s.value = f; if (window.Start) { Start(f); Play(); } }
})();
</script>
</body>""")
    open(os.path.join(gibbs, "play.html"), "w").write(html)
    return len(files)


def sync_replays(logs, gibbs):
    """Convert any .rcg without an up-to-date .replay."""
    rep = os.path.join(gibbs, "replays")
    os.makedirs(rep, exist_ok=True)
    mpath = os.path.join(rep, MANIFEST)
    try:
        manifest = json.load(open(mpath))
    except (OSError, ValueError):
        manifest = {}
    made = 0
    for name in sorted(os.listdir(logs)):
        if not (name.endswith(".rcg") or name.endswith(".rcg.gz")):
            continue
        src = os.path.join(logs, name)
        mtime = os.path.getmtime(src)
        rec = manifest.get(name)
        if (rec and rec.get("mtime", 0) >= mtime
                and os.path.exists(os.path.join(rep, rec["replay"]))):
            continue
        tmp = os.path.join(rep, f".converting-{os.getpid()}.replay")
        try:
            info = to_replay(src, tmp)
        except Exception as e:
            print(f"   !! {name}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        if info["cycles"] == 0:
            print(f"   -- {name}: no show data (match never kicked off?)")
            os.remove(tmp)
            continue
        dst = replay_name(name, src, info)
        os.replace(tmp, os.path.join(rep, dst))
        if rec and rec.get("replay") != dst:
            old = os.path.join(rep, rec["replay"])
            if os.path.exists(old):
                os.remove(old)
        manifest[name] = {"replay": dst, "mtime": mtime}
        print(f"   -> {dst}  ({info['cycles']} cycles)")
        made += 1
    tmp = mpath + ".tmp"
    json.dump(manifest, open(tmp, "w"), indent=1, sort_keys=True)
    os.replace(tmp, mpath)
    return made


def cmd_watch(a, rest):
    import functools, http.server, socketserver, threading
    gibbs = os.path.expanduser(a.gibbs)
    logs = os.path.expanduser(a.logs)
    for p in (logs, gibbs):
        if not os.path.isdir(p):
            die("no such directory: " + p)
    patch_player_js(gibbs)
    print("scanning " + logs)
    sync_replays(logs, gibbs)
    print(f"{write_index(gibbs)} replay(s) ready")
    if a.no_serve:
        return 0

    if a.follow:
        def loop():
            while True:
                time.sleep(20)
                if sync_replays(logs, gibbs):
                    write_index(gibbs)
        threading.Thread(target=loop, daemon=True).start()

    class _S(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=gibbs)
    with _S((a.host, a.port), handler) as httpd:
        print(f"\nserving on http://localhost:{a.port}/   (ctrl-c to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


# TIMING
# The budget is MATCH-WIDE and TEAMWIDE. Read it through
# agent.budget.remaining_ms and decide how to spend it; see interface.py.

TIMING_ROOT = os.path.join(RUNLOGS, "timing")


def _pct(xs, p):
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))
    return sorted(xs)[i]


def read_timing(stamp_dir):
    """-> [{team, unum, spent_ms, cpu_ms[], wall_ms[], ...}] for one match."""
    out = []
    for p in sorted(glob.glob(os.path.join(stamp_dir, "*.json"))):
        try:
            out.append(json.load(open(p)))
        except (OSError, ValueError):
            continue
    return out


def forfeits(stamp):
    """Teams that hit the hard budget deadline and exited."""
    d = stamp if os.path.isdir(stamp) else os.path.join(TIMING_ROOT, stamp)
    out = []
    for p in sorted(glob.glob(os.path.join(d, "FORFEIT_*.json"))):
        try:
            out.append(json.load(open(p)))
        except (OSError, ValueError):
            pass
    return out


def summarise(rows):
    """Per team.
    
    Per-decision percentiles are kept for diagnosis.
    """
    per = defaultdict(lambda: {"cpu": [], "wall": [],
                               "spend": [], "broke": [], "killed": 0,
                               "match_ms": 0.0, "n": 0})
    for r in rows:
        t = per[r.get("team", "?")]
        t["cpu"] += r.get("cpu_ms", [])
        t["wall"] += r.get("wall_ms", [])
        t["spend"].append((r.get("spent_ms", 0.0) or 0.0, r.get("unum", "?")))
        if r.get("broke_at"):
            t["broke"].append((r.get("unum", "?"), r["broke_at"],
                               len(r.get("cpu_ms", []))))
        t["killed"] += r.get("killed", 0) or 0
        t["match_ms"] = r.get("match_ms", 0.0) or t["match_ms"]
        t["n"] += 1
    out = {}
    for team, t in per.items():
        worst = max(t["spend"]) if t["spend"] else (0.0, "?")
        s = {"players": t["n"], "decisions": len(t["cpu"]),
             "spend_ms": worst[0], "spend_unum": worst[1],
             "match_ms": t["match_ms"], "broke": t["broke"],
             "killed": t["killed"]}
        for key in ("cpu", "wall"):
            for p in (50, 95, 99, 100):
                s[f"{key}_p{p}"] = _pct(t[key], p)
        out[team] = s
    return out



# diagnose setup problems
def _check(label, ok, detail="", fix=""):
    mark = (COLOR_MAP["g"] + " ok " + COLOR_MAP["x"]) if ok else (COLOR_MAP["r"] + "FAIL" + COLOR_MAP["x"])
    print(f"  [{mark}] {label}")
    if detail:
        print("         " + detail)
    if not ok and fix:
        print(f"         {COLOR_MAP['y']}fix: {fix}{COLOR_MAP['x']}")
    return ok


_EXPECTED = {
    "src/student/interface.py": ("StarterAgent", "Action", "Tools", "note"),
    "src/student/framework/runtime.py": ("load_student", "Budget", "arm"),
    "src/student/framework/policy.py": ("check", "verdict"),
    "src/student/framework/world.py": ("AgentView", "_unwrap"),
    "src/student/framework/trace.py": ("note", "record"),
}

_RUNNER = "src/decision_makers/play_on_decision_maker.py"
_RUNNER_CALLS = ("load_student", "AgentView", "check", "Budget")


def _names_in(path):
    try:
        tree = ast.parse(open(path, errors="replace").read(), filename=path)
    except (OSError, SyntaxError):
        return None
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            out.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((al.asname or al.name).split(".")[0] for al in n.names)
    return out


def _wiring():
    ok = True
    for rel, wanted in sorted(_EXPECTED.items()):
        p = os.path.join(HERE, rel)
        names = _names_in(p)
        if names is None:
            ok &= _check(rel, False, "missing or will not parse",
                         "put it back - see INTEGRATION.md")
            continue
        missing = [w for w in wanted if w not in names]
        ok &= _check(rel, not missing,
                     "defines " + ", ".join(sorted(wanted)) if not missing
                     else "missing: " + ", ".join(missing))

    names = _names_in(os.path.join(HERE, _RUNNER))
    if names is None:
        ok &= _check(_RUNNER, False, "missing or will not parse")
    else:
        missing = [c for c in _RUNNER_CALLS if c not in names]
        ok &= _check(_RUNNER + " is patched", not missing,
                     "calls " + ", ".join(_RUNNER_CALLS) if not missing
                     else "does not reference: " + ", ".join(missing),
                     "this is py2d's original file - student code never runs")

    # agent.py: a class with both methods, without executing anything
    p = os.path.join(HERE, "src", "student", "agent.py")
    try:
        tree = ast.parse(open(p, errors="replace").read(), filename=p)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        bodies = {c.name: {f.name for f in c.body
                           if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
                  for c in classes}
        playable = [n for n, fns in bodies.items()
                    if {"on_ball", "off_ball"} <= fns
                    or any(b.id in bodies or getattr(b, "id", "") == "StarterAgent"
                           for b in next(c for c in classes if c.name == n).bases
                           if isinstance(b, ast.Name))]
        ok &= _check("src/student/agent.py has an agent class", bool(playable),
                     ", ".join(playable) or "no class with on_ball/off_ball",
                     "define one, subclassing StarterAgent")
        if playable and "Agent1" not in playable:
            say(f"no class named Agent1 - `{RUN} match` defaults to it, so "
                f"pass --left :{playable[0]} or rename.")
    except (OSError, SyntaxError) as e:
        ok &= _check("src/student/agent.py", False, str(e)[:60])
    return ok


def cmd_doctor(a, rest):
    ok = True
    print("\npython")
    ok &= _check("python 3.11", sys.version_info[:2] == (3, 11),
                 f"{sys.executable}  {sys.version.split()[0]}",
                 "conda env create -f environment.yml && conda activate a3")

    import platform
    ok &= _check("linux x86_64", sys.platform.startswith("linux")
                 and platform.machine() == "x86_64",
                 f"{sys.platform} {platform.machine()}",
                 "use WSL2 on Windows, or the Docker image on a Mac")
    libc, ver = platform.libc_ver()
    new_enough = libc == "glibc" and tuple(
        int(x) for x in ver.split(".")[:2]) >= (2, 29)
    ok &= _check("glibc 2.29+", new_enough, f"{libc} {ver}",
                 "needs Ubuntu 20.04+ / Debian 11+, or the Docker image")
    print("\nimports")
    for mod in ("grpc", "numpy", "scipy", "pyrusgeom"):
        try:
            __import__(mod)
            ok &= _check("import " + mod, True)
        except ImportError as e:
            ok &= _check("import " + mod, False, str(e),
                         "conda env create -f environment.yml && conda activate a3")

    print("\ngenerated protobuf")
    for f in ("service_pb2.py", "service_pb2_grpc.py"):
        ok &= _check(f, os.path.exists(os.path.join(HERE, f)), "", "./generate.sh")

    print("\nbinaries")
    proxy_dir = os.path.join(HERE, "scripts", "proxy")
    ok &= _check("scripts/proxy", os.path.isdir(proxy_dir), proxy_dir,
                 "missing from your copy - re-extract the assignment bundle")
    ok &= _check("rcssserver extracted", os.access(SERVER_BIN, os.X_OK),
                 SERVER_BIN,
                 "./rcssserver-*.AppImage --appimage-extract   "
                 "# libfuse2 is unavailable, so FUSE mounting will not work")
    for b in ("sample_player", "sample_coach", "sample_trainer"):
        p = os.path.join(HERE, "scripts", "proxy", b)
        ok &= _check("proxy/" + b, os.access(p, os.X_OK), "",
                     "./<name>.AppImage --appimage-extract")

    ok &= _check("start.py", os.path.isfile(os.path.join(HERE, "start.py")),
                 "", "this is what a3 launches for each team")

    print("\nframework wiring")
    ok &= _wiring()

    _code = ("from src.student.framework.runtime import load_student\n"
             "print(load_student()[1])\n")
    _r = subprocess.run([sys.executable, "-c", _code], cwd=HERE,
                        capture_output=True, text=True)
    _loaded = _r.returncode == 0
    ok &= _check("student agent imports and loads", _loaded,
                 _r.stdout.strip() if _loaded
                 else (_r.stderr.strip().splitlines() or [""])[-1][:70],
                 "python3 -c 'from src.student.framework.runtime "
                 "import load_student; load_student()'")
    if not _loaded and _r.stderr:
        sys.stderr.write(_r.stderr)

    print("\nnetwork")
    np_ = os.environ.get("no_proxy", "")
    proxied = any(os.environ.get(k) for k in
                  ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"))
    ok &= _check("no_proxy covers localhost",
                 not proxied or ("localhost" in np_ and "127.0.0.1" in np_),
                 "no_proxy=" + (np_ or "<unset>") if proxied
                 else "no HTTP proxy set",
                 'export no_proxy="localhost,127.0.0.1,::1"   '
                 "# or gRPC routes localhost through the HTTP proxy")

    print("\nconfig")
    cp = os.path.join(HERE, a.conf)
    conf = read_conf(cp)
    ok &= _check("config readable", bool(conf),
                 f"{cp}  ({len(conf)} settings)",
                 f"restore {a.conf} from the assignment bundle")
    if conf:
        for key in ("a3::team_size", "server::fullstate_l",
                    "server::fullstate_r", "server::synch_mode",
                    "server::game_log_version", "player::player_types",
                    "player::subs_max"):
            v = conf_get(conf, key)
            ok &= _check(key, v is not None,
                         f"= {v}" if v is not None else "not set",
                         f"add {key} to {a.conf}")
        if (conf_get(conf, "server::synch_mode") or "").lower() in ("true", "on", "1"):
            say("synch_mode is on. Nothing here can test it - if the proxy "
                "mishandles (think)/(done) the match hangs at cycle 0. "
                "Confirm with one short match.")

    print("\nreplay viewer")
    g = os.path.expanduser(a.gibbs)
    ok &= _check("replay viewer", os.path.isfile(
                     os.path.join(g, "scripts", "player.js")), g,
                 "the gibbs/ folder is missing from your copy")

    print("\n" + (COLOR_MAP["g"] + "all good" + COLOR_MAP["x"] if ok else
                  COLOR_MAP["r"] + "problems above - fix and re-run" + COLOR_MAP["x"]))
    return 0 if ok else 1


# CLI
STUDENT_COMMANDS = ("doctor", "match", "why", "watch")

INSTRUCTOR_COMMANDS = ("validate",)
INSTRUCTOR = (os.path.exists(os.path.join(HERE, ".a3_instructor"))
              or os.environ.get("A3_INSTRUCTOR") == "1")

COMMANDS = STUDENT_COMMANDS + (INSTRUCTOR_COMMANDS if INSTRUCTOR else ())

MATCH_EPILOG = """\
a team spec says where that team's code comes from:

  --right full                  the HELIOS reference that ships with py2d
  --left  subs/x/agent.py       that submission, class Agent1
  --left  subs/x/agent.py:Alpha that submission, class Alpha
  --left  :Agent3               src/student/agent.py, class Agent3
  (omitted)                     src/student/agent.py, class Agent1

the two team names must differ, or the server treats them as one team.
"""


def main():
    ap = argparse.ArgumentParser(
        prog=PROG, description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--conf", default=DEFAULT_CONF, metavar="FILE",
                        help="match settings, rcssserver format "
                             "(default: %(default)s)")
    common.add_argument("--dry-run", action="store_true",
                        help="say what would happen, do nothing")
    sub = ap.add_subparsers(dest="cmd",
                            metavar="{" + ",".join(COMMANDS) + "}")

    def parser(name, help_text, epilog=None):
        kw = {"help": help_text} if name in COMMANDS else {}
        return sub.add_parser(
            name, parents=[common], epilog=epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter, **kw)

    p = parser("doctor", "is this machine set up right")
    p.add_argument("--gibbs", default=GIBBS)
    p.set_defaults(fn=cmd_doctor)

    p = parser("match", "run one match", epilog=MATCH_EPILOG)
    p.add_argument("left"); p.add_argument("right")
    p.add_argument("-n", "--size", type=int)
    p.add_argument("-t", "--half", type=int)
    p.add_argument("-p", "--port", type=int, default=6000)
    p.add_argument("--left", dest="left_spec", metavar="SPEC",
                   help="where the left team's code comes from: `full`, a "
                        "path to agent.py, or `:ClassName`. "
                        "Default: src/student/agent.py, class Agent1")
    p.add_argument("--right", dest="right_spec", metavar="SPEC",
                   help="same, for the right team")

    p.add_argument("--trace", dest="trace", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    p.add_argument("--no-trace", dest="trace", action="store_false",
                   help="do not record decisions (default: record)")
    p.add_argument("--policy", default="strip",
                   choices=("off", "warn", "strip", "raise"))
    p.add_argument("--no-sandbox", action="store_true")
    p.add_argument("--stamp")
    p.add_argument("-o", "--override", action="append", default=[], metavar="K=V")
    p.set_defaults(fn=cmd_match)

    p = parser("why", "what did my players do")
    p.add_argument("cycles", nargs="*", type=int)
    p.add_argument("-t", "--team")
    p.add_argument("--match", metavar="SUBSTR")
    p.add_argument("--list", action="store_true")
    p.add_argument("-g", "--grep")
    p.add_argument("--slow", type=float, metavar="MS")
    p.add_argument("--chase", action="store_true")
    p.add_argument("--stack", metavar="LABEL")
    p.add_argument("--possession", action="store_true")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.set_defaults(fn=cmd_why)

    p = parser("watch", "see the matches in a browser")
    p.add_argument("--logs", default=LOGS)
    p.add_argument("--gibbs", default=GIBBS)

    p.add_argument("--host", default=os.environ.get("A3_WATCH_HOST",
                                                    "127.0.0.1"))
    p.add_argument("-p", "--port", type=int, default=8080)
    p.add_argument("--follow", action="store_true",
                   help="re-scan for new logs every 20s")
    p.add_argument("--no-serve", action="store_true")
    p.set_defaults(fn=cmd_watch)

    p = parser("validate", "is this .rcg real, or a harness artifact")
    p.add_argument("logs", nargs="+")
    p.add_argument("--expect-cycles", type=int)
    p.add_argument("--expect-players", type=int)
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_validate)


    known, rest = ap.parse_known_args()

    leftovers = []
    it = iter(rest)
    for tok in it:
        if tok == "--dry-run":
            known.dry_run = True
        elif tok == "--conf":
            known.conf = next(it, known.conf)
        elif tok.startswith("--conf="):
            known.conf = tok.split("=", 1)[1]
        else:
            leftovers.append(tok)
    rest = leftovers
    if rest and rest[0] == "--":
        rest = rest[1:]
    if not known.cmd:
        ap.print_help()
        return 0
    return known.fn(known, rest)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        pass
    except KeyboardInterrupt:
        sys.exit(130)