# DATA COLLECTOR - not a submission. v8b (tools/versions/v8b) plus logging of
# keep-the-ball context and bounded exploration; see EXPLORE_* below and
# tools/collect/README.txt. Training matches only, never evaluation.

import math
from itertools import permutations

import numpy as np
from pyrusgeom.geom_2d import Sector2D, Vector2D
from src.student.interface import StarterAgent, Action, Tools, note


# Design parameters of the model and search. Nothing about the environment is
# fixed here: physics, pitch and rule limits are derived from server_params in
# _Model, and how long a decision may think from agent.budget every cycle.
MAX_DEPTH = 3           # plies of our own actions in the on-ball search
TOP_M = 6               # options expanded per ply (beam width)
THINK_CAP_MS = 60.0     # never think longer than this, even with budget spare
SHARE = 0.9             # a decision may spend this much of (budget left / cycles left):
                        # below 1, so the budget cannot run out while shares are kept
WALL_FRACTION = 0.25    # share of the 1s decision cap a search may use
CHEAP_SHARE_MS = 1.0    # below this share, fall back to the starter rules

PASS_RING = 3.0         # pass targets: each teammate plus a ring this wide
PASS_RING_N = 8
PASS_SPEEDS = (1.0, 0.8, 0.6)   # fractions of ball_speed_max tried per target
PASS_MIN = 3.0
# rcssserver player.cpp, Player::kick: the kick gets a random vector of length
# up to kick_rand * (power/max_power) * (pos_rate + speed_rate), each rate in
# [0.5, 1] -- "[0, 2*kick_rand]" in the source. We take the upper bound.
KICK_NOISE_RATE = 2.0
RECV_SLACK, RECV_SCALE = 1.0, 1.5    # receiver needs a cycle to turn/start

# Success-probability curves: P = sigmoid((margin - OFFSET) / SCALE), margin in
# cycles. Fitted by MAP (L12: maximum likelihood plus pseudo-counts from the old
# hand-set curve as the prior) with tools/fit_probs.py, outcomes read from the
# replays. Second round, 44 matches: the first fit's play had tried passes at
# low margins, so this data covers them (selection bias of the first round).
# Held-out log-likelihood per sample, hand-set -> fitted: pass -0.939 -> -0.245
# (n=3293), dribble -0.304 -> -0.043 (n=6636), hold -0.257 -> -0.040 (n=5174).
# Shots: 22 samples, mostly against weak opponents, so the hand-set prior
# (0.5, 1.0) is kept. The dribble offset sits at the fit's grid edge (-4):
# dribbles almost never fail, so how far it can be pushed is unknown.
PASS_OFFSET, PASS_SCALE = -3.2, 0.82
SHOT_OFFSET, SHOT_SCALE = 0.5, 1.0
DRIBBLE_OFFSET, DRIBBLE_SCALE = -4.0, 1.09
HOLD_OFFSET, HOLD_SCALE = -1.1, 0.46

SHOT_N = 9
SHOT_MIN_SPEED = 2.0    # a one-touch shot slower than this is not worth it

DRIBBLE_DIST = 5.0
DRIBBLE_SPEED = 0.52    # m/cycle while the ball moves: mean of 6737 dribble
                        # cycles in 10 traced matches (turns are counted apart)
DRIBBLE_DIRS = 12

HOLD_COST = 0.02        # holding wastes time; grows with consecutive holds
CLEAR_DIST = 30.0       # clear() lands roughly here, wide on the ball's side
RACE_SCALE = 2.0        # cycles of head start -> P(we win a loose ball)
LEAD_DIST = 7.0         # lead passes: this far ahead of the receiver
SPACE_DISTS = (12.0, 20.0, 28.0)              # long balls into space
SPACE_ANGLES = (-60.0, -30.0, 0.0, 30.0, 60.0)
CLEAR_KEEP = 0.5        # clear() is planned only when keeping the ball is this unlikely
CONTROL_DELAY = 1.0     # cycles a receiver needs before its own next kick
RESPONSE = 0.6          # how far opponents close in while the ball travels
PLY_COST = 0.01         # cost of each extra ply (sign-neutral discount)
RISK = 0.2              # weight of losing the ball later, in value()
# Evaluation functions attack() = P(our possession scores from here) and
# danger() = P(their possession scores from where they won it), fitted by MAP
# with tools/fit_value.py on 99 replays against full (26533 / 6332 touches;
# pseudo-counts from the old hand-set weights as the prior, strength chosen
# on held-out matches). Held-out log-likelihood per touch, hand-set -> fitted:
# attack -0.149 -> -0.126, danger -0.546 -> -0.532. The hand-set curve
# overrated the ball in midfield about four-fold (observed 0.04 at 40-50 m).
ATTACK_FIELD, ATTACK_CHANCE, ATTACK_LEN = 0.08, 1.05, 13.0
DANGER_FIELD, DANGER_CHANCE, DANGER_LEN = 0.3, 0.6, 14.0

# exploration safety limits (collector only)
EXPLORE_P = 0.25        # chance an eligible decision is swapped for a random keep action
EXPLORE_P_FLOOR = 0.3   # never explore an action the current model gives less than this
EXPLORE_PRESS = 2.0     # "pressured": an opponent within this many tackle_dist of the ball
EXPLORE_MAX = 15        # explored decisions per player per match, at most
EXPLORE_GAP = 10        # cycles between two explored decisions of one player

TURN_ANGLE = 45.0       # facing further away than this costs turn cycles
COVER_DIST = 5.0
CROWD_R = 6.0
FOUL_SAFETY = 0.3       # extra metres around rcssserver's charge geometry
DEFEND_FULL_X = 10.0


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _xy(v):
    return np.array([v.x(), v.y()], dtype=float)


def _ring(center, radius, n):
    """Tools.points_around, as an (n, 2) array."""
    a = np.radians(np.arange(n) * 360.0 / n)
    return center + radius * np.stack([np.cos(a), np.sin(a)], axis=1)


def _norm_deg(a):
    return (a + 180.0) % 360.0 - 180.0


class _Model:
    """Simulator constants for this match, read once from server_params,
    and the ball-flight / player-reach model built on them."""

    def __init__(self, sp):
        self.decay = sp.ball_decay
        self.ball_max = sp.ball_speed_max
        self.max_power = sp.max_power
        self.ball_accel_max = sp.ball_accel_max
        self.speed = sp.real_speed_max or sp.player_speed_max
        self.kickable = sp.kickable_area
        self.catchable = sp.catchable_area
        self.tackle = sp.tackle_dist
        self.hl = sp.pitch_half_length
        self.hw = sp.pitch_half_width
        self.goal_half = sp.goal_width / 2.0
        self.pen_x = sp.their_penalty_area_line_x
        self.pen_hw = sp.penalty_area_half_width
        step = sp.simulator_step or 100.0
        self.cycles = int(sp.half_time * 1000.0 / step) * max(1, sp.nr_normal_halfs)

        # Limits derived from the simulator instead of hand-set:
        # look ahead until a full-speed ball has slowed to "stopped"
        self.horizon = int(math.ceil(math.log(sp.stopped_ball_vel / self.ball_max)
                                     / math.log(self.decay)))
        self.t = np.arange(1, self.horizon + 1, dtype=float)
        self.travel = (1.0 - self.decay ** self.t) / (1.0 - self.decay)
        # the farthest any kick can reach within that horizon
        self.pass_max = float(self.ball_max * self.travel[-1])
        # a shot is only a shot while it still outruns a sprinting player
        # at the goal line: v_max * decay^n >= speed, summed over n cycles
        self.shot_max = (self.ball_max - self.speed) / (1.0 - self.decay)
        # aim with the whole ball inside the post
        self.post_clear = sp.goal_post_radius + sp.ball_size
        # receivers can move this far while a pass is being struck
        self.offside_margin = self.speed
        # mark from where a tackle reaches
        self.mark_dist = sp.tackle_dist
        self.max_moment, self.inertia = sp.max_moment, sp.inertia_moment
        self.tackle_cycles = sp.tackle_cycles
        # largest random velocity error per component on a full-power kick
        self.kick_noise = sp.kick_rand * KICK_NOISE_RATE
        ring = _ring(np.zeros(2), PASS_RING, PASS_RING_N)
        lead = _ring(np.zeros(2), LEAD_DIST, 12)[[11, 0, 1]]   # -30, 0, +30 deg
        self.ring = np.vstack([np.zeros((1, 2)), ring, lead])
        sa = np.radians(np.repeat(SPACE_ANGLES, len(SPACE_DISTS)))
        sr = np.tile(SPACE_DISTS, len(SPACE_ANGLES))
        self.space = np.stack([sr * np.cos(sa), sr * np.sin(sa)], axis=1)

    # ball and players

    def steps_for(self, dist, speed):
        """Cycles a ball kicked at `speed` takes to cover `dist`; inf when
        it stops short."""
        r = 1.0 - dist * (1.0 - self.decay) / speed
        safe = np.where(r > 0, r, 1.0)
        n = np.ceil(np.log(safe) / math.log(self.decay) - 1e-9)
        return np.where(r > 0, np.maximum(n, 1.0), np.inf)

    def speed_for(self, dist, n):
        """Launch speed that covers `dist` in exactly `n` cycles (as kick())."""
        return dist * (1.0 - self.decay) / (1.0 - self.decay ** n)

    def one_step_speed(self, dirs, ball_vel, krate):
        """Fastest launch one touch gives along each unit vector in `dirs`:
        Tools.calc_max_velocity, vectorised."""
        a = min(self.max_power * krate, self.ball_accel_max)
        ub = dirs @ ball_vel
        disc = ub * ub - ball_vel @ ball_vel + a * a
        root = np.sqrt(np.maximum(disc, 0.0))
        hi, lo = ub + root, ub - root
        s = np.where(hi > self.ball_max, self.ball_max, hi)
        bad = (disc < 0) | (hi <= 0) | (lo > self.ball_max)
        return np.where(bad, 0.0, s)

    def on_target(self, dist, speed, lo, hi):
        """P(the kick's sideways error at `dist` stays within [-lo, hi]).

        rcssserver adds a uniform random vector of up to kick_noise per
        component to the kick, so the launch direction is off by about
        u / speed and the ball strays dist * u / speed sideways."""
        a = dist * self.kick_noise / np.maximum(speed, 1e-6)
        lo, hi = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
        inside = np.minimum(lo, a) + np.minimum(hi, a)
        return np.where(a > 1e-9, np.clip(inside / np.maximum(2 * a, 1e-9), 0, 1), 1.0)

    def turns(self, off_deg, speed):
        """Turn cycles to face `off_deg` away while moving at `speed`: one
        turn rotates at most max_moment / (1 + inertia_moment * speed)."""
        off = np.abs(np.asarray(off_deg, float))
        per = self.max_moment / (1.0 + self.inertia * np.asarray(speed, float))
        return np.where(off > TURN_ANGLE, np.ceil(off / per), 0.0)

    def reach(self, dist, radius):
        """Cycles a player needs to get within `radius` of a point `dist`
        away, running flat out."""
        return np.maximum(0.0, dist - radius) / self.speed

    def paths(self, origin, dirs, speeds, T):
        """(K, T, 2) ball positions for cycles 1..T after each launch."""
        moved = speeds[:, None] * self.travel[None, :T]
        return origin[..., None, :] + dirs[:, None, :] * moved[..., None]

    def margin(self, paths, arrive, delay, opps, goalie, radius):
        """Opponents' best response: for each path, the smallest
        (opponent reach time - ball time) over every opponent and every
        cycle up to `arrive`. Negative means someone gets there first."""
        K, T, _ = paths.shape
        if not len(opps):
            return np.full(K, 99.0)
        diff = paths[:, :, None, :] - opps[None, None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=-1))
        rad = np.full(dist.shape, radius)
        if goalie.any():
            inbox = ((paths[..., 0] > self.pen_x)
                     & (np.abs(paths[..., 1]) < self.pen_hw))
            rad = np.where(inbox[..., None] & goalie[None, None, :],
                           max(radius, self.catchable), rad)
        t = self.t[None, :T, None] + np.reshape(delay, (-1, 1, 1))
        m = self.reach(dist, rad) - t
        valid = self.t[None, :T] <= np.reshape(arrive, (-1, 1))
        m = np.where(valid[..., None], m, 99.0)
        return m.min(axis=(1, 2))

    def advance(self, opps, target, n):
        """Opponents after closing in on `target` for n cycles."""
        if not len(opps):
            return opps
        vec = target - opps
        d = np.maximum(np.linalg.norm(vec, axis=1), 1e-6)
        step = np.minimum(np.maximum(d - self.kickable, 0.0),
                          n * self.speed * RESPONSE)
        return opps + vec / d[:, None] * step[:, None]

    # values, in goals: +1 is a goal for us

    def attack(self, x, y):
        """Worth of having the ball at (x, y), P(this possession scores):
        territory, plus a scoring chance that needs both closeness and an
        open angle on goal."""
        dx = self.hl - x
        d = np.hypot(dx, y)
        angle = np.abs(np.arctan2(self.goal_half - y, dx)
                       - np.arctan2(-self.goal_half - y, dx))
        chance = np.exp(-d / ATTACK_LEN) * np.minimum(1.0, angle)
        return ATTACK_FIELD * (1.0 - d / (2 * self.hl)) + ATTACK_CHANCE * chance

    def danger(self, x, y):
        """Cost of losing the ball at (x, y), P(their possession scores)."""
        d = np.hypot(x + self.hl, y)
        return (DANGER_CHANCE * np.exp(-d / DANGER_LEN)
                + DANGER_FIELD * (self.hl - x) / (2 * self.hl))

    def value(self, x, y):
        """Worth of keeping the ball at (x, y): its attacking worth, less
        the risk of losing it there later."""
        return self.attack(x, y) - RISK * self.danger(x, y)

    def time_cost(self, origin):
        """Living cost of one more cycle on the ball at `origin` (L08): the
        value a cycle of dribbling straight at goal would have added there.
        Taking time is then only worth it for something better."""
        to_goal = np.array([self.hl, 0.0]) - origin
        u = to_goal / max(float(np.linalg.norm(to_goal)), 1e-6)
        ahead = origin + u * DRIBBLE_SPEED
        return max(0.0, float(self.value(ahead[0], ahead[1])
                              - self.value(origin[0], origin[1])))

    def legal(self, pts, margin=1.5):
        """In bounds and outside our own penalty area (a pass there can
        become a back-pass to the goalie)."""
        x, y = pts[:, 0], pts[:, 1]
        inside = (np.abs(x) < self.hl - margin) & (np.abs(y) < self.hw - margin)
        our_box = (x < -self.pen_x) & (np.abs(y) < self.pen_hw)
        return inside & ~our_box

    # option generators

    def shot_targets(self, origin):
        ys = np.linspace(-self.goal_half + self.post_clear,
                         self.goal_half - self.post_clear, SHOT_N)
        targets = np.stack([np.full(SHOT_N, self.hl), ys], axis=1)
        vec = targets - origin
        dist = np.linalg.norm(vec, axis=1)
        return targets, vec / dist[:, None], dist

    def shot_p(self, origin, targets, dirs, dist, speed, delay, opps, goalie):
        """P(goal) for shots launched at `speed` after `delay` cycles:
        (p, opponent margin, kick-noise factor)."""
        speed = np.broadcast_to(np.asarray(speed, float), dist.shape)
        n = self.steps_for(dist, np.maximum(speed, 1e-6))
        ok = (np.isfinite(n) & (n <= self.horizon) & (speed > 0)
              & (dist <= self.shot_max))
        n = np.where(ok, n, 1.0)
        m = self.margin(self.paths(origin, dirs, speed, int(n.max())),
                        n, delay, opps, goalie, self.kickable)
        # how far the ball may stray sideways and still go in
        lo = targets[:, 1] - (-self.goal_half + self.post_clear)
        hi = (self.goal_half - self.post_clear) - targets[:, 1]
        f = self.on_target(dist, speed, lo, hi)
        p = np.where(ok, _sig((m - SHOT_OFFSET) / SHOT_SCALE) * f, 0.0)
        return p, m, f

    def pass_targets(self, origin, mates, line_x):
        """Points on and around each receiver, with every launch speed
        tried: (targets, receiver index, n, speed, dirs, dist) or None."""
        if not len(mates):
            return None
        line = max(line_x, origin[0]) - self.offside_margin
        onside = (mates[:, 0] <= line) | (mates[:, 0] <= 0.0)
        pts = (mates[:, None, :] + self.ring[None, :, :]).reshape(-1, 2)
        recv = np.repeat(np.arange(len(mates)), len(self.ring))
        # long balls into space, for whichever teammate is nearest
        space = origin + self.space
        near = np.linalg.norm(space[:, None, :] - mates[None], axis=2).argmin(axis=1)
        pts, recv = np.vstack([pts, space]), np.concatenate([recv, near])
        vec = pts - origin
        dist = np.linalg.norm(vec, axis=1)
        keep = (onside[recv] & self.legal(pts)
                & (dist >= PASS_MIN) & (dist <= self.pass_max))
        if not keep.any():
            return None
        pts, recv, vec, dist = pts[keep], recv[keep], vec[keep], dist[keep]
        k = len(pts)
        speeds = np.repeat(np.array(PASS_SPEEDS) * self.ball_max, k)
        pts, recv = np.tile(pts, (len(PASS_SPEEDS), 1)), np.tile(recv, len(PASS_SPEEDS))
        vec, dist = np.tile(vec, (len(PASS_SPEEDS), 1)), np.tile(dist, len(PASS_SPEEDS))
        n = self.steps_for(dist, speeds)
        ok = np.isfinite(n) & (n <= self.horizon)
        if not ok.any():
            return None
        pts, recv, vec, dist, n = pts[ok], recv[ok], vec[ok], dist[ok], n[ok]
        speed = self.speed_for(dist, n)
        return pts, recv, n, speed, vec / dist[:, None], dist

    def pass_p(self, origin, cand, delay, mates, opps, goalie):
        """P(the receiver gets it first) for each pass candidate:
        (p, opponent margin, the other factors) with p = sig(margin) * other."""
        pts, recv, n, speed, dirs, dist = cand
        m = self.margin(self.paths(origin, dirs, speed, int(n.max())),
                        n, delay, opps, goalie, self.kickable)
        rd = np.linalg.norm(mates[recv] - pts, axis=1)
        slack = np.where(rd > self.kickable, RECV_SLACK, 0.0)
        late = n + delay - (self.reach(rd, self.kickable) + slack)
        # kick noise: the receiver can still reach a ball that strays as far
        # as he can run in his spare cycles
        give = self.kickable + self.speed * np.maximum(late, 0.0)
        f = _sig(late / RECV_SCALE) * self.on_target(dist, speed, give, give)
        return _sig((m - PASS_OFFSET) / PASS_SCALE) * f, m, f

    def dribbles(self, origin, angles, turns, opps, goalie):
        """Dribbles of DRIBBLE_DIST at each absolute angle (degrees), each
        starting after `turns` cycles of turning: (ends, P(keep it),
        cycles, ev), illegal end points dropped."""
        a = np.radians(angles)
        dirs = np.stack([np.cos(a), np.sin(a)], axis=1)
        ends = origin + DRIBBLE_DIST * dirs
        ok = self.legal(ends)
        if not ok.any():
            return ends[:0], np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        dirs, ends, turns = dirs[ok], ends[ok], np.asarray(turns, float)[ok]
        nd = int(math.ceil(DRIBBLE_DIST / DRIBBLE_SPEED))
        steps = np.minimum(np.arange(1, nd + 1) * DRIBBLE_SPEED, DRIBBLE_DIST)
        paths = origin + dirs[:, None, :] * steps[None, :, None]
        mg = self.margin(paths, np.full(len(ends), nd), turns, opps, goalie,
                         self.tackle)
        p = _sig((mg - DRIBBLE_OFFSET) / DRIBBLE_SCALE)
        ev = (p * self.value(ends[:, 0], ends[:, 1])
              - (1.0 - p) * self.danger(origin[0], origin[1]))
        return ends, p, nd + turns, ev, mg

    def race(self, point, mates, opps):
        """P(one of `mates` gets to a loose ball at `point` before any
        opponent does)."""
        mine = (self.reach(np.linalg.norm(mates - point, axis=1), self.kickable).min()
                if len(mates) else 99.0)
        theirs = (self.reach(np.linalg.norm(opps - point, axis=1), self.kickable).min()
                  if len(opps) else 99.0)
        return float(_sig((theirs - mine) / RACE_SCALE))

    def shot_ev(self, origin, targets, p):
        """A goal, or the ball given away somewhere along its way. Out of
        range is not an option at all."""
        mid = (targets + origin) / 2.0
        ev = p - (1.0 - p) * self.danger(mid[:, 0], mid[:, 1])
        far = np.linalg.norm(targets - origin, axis=1) > self.shot_max
        return np.where(far, -np.inf, ev)

    def pass_ev(self, origin, pts, p):
        mid = (pts + origin) / 2.0
        return (p * self.value(pts[:, 0], pts[:, 1])
                - (1.0 - p) * self.danger(mid[:, 0], mid[:, 1]))


class _Clock:
    """Stops a search before either this decision's share of the match
    budget or the per-decision wall cap runs out."""

    def __init__(self, budget, share_ms):
        self.budget = budget
        self.share = share_ms
        self.cpu0 = budget.remaining_ms
        self.wall0 = budget.decision_remaining_ms

    def used(self):
        wall = self.wall0 - self.budget.decision_remaining_ms
        if math.isinf(self.cpu0):
            return wall
        return max(wall, self.cpu0 - self.budget.remaining_ms)

    def ok(self):
        return self.used() < self.share


class Agent1(StarterAgent):
    """Two-sided search on the ball, a shared role allocation off it.

    Every player sees the same exact world, so each one computes the same
    allocation for the whole team and then plays its own part."""

    def __init__(self):
        super().__init__()
        self._m = None
        self._holds = 0
        self._explored = 0
        self._last_explore = -10 ** 6

    # entry points: never let an exception reach the framework

    def on_ball(self, agent) -> None:
        try:
            self._on_ball(agent)
        except Exception as e:
            note(agent, "error", where="on_ball", err=repr(e)[:120])
            Action.clear(agent)

    def off_ball(self, agent) -> None:
        try:
            self._off_ball(agent)
        except Exception as e:
            note(agent, "error", where="off_ball", err=repr(e)[:120])
            Action.go_to_point(agent, Tools.home_position(agent))
            Action.turn_to_ball(agent)

    # shared plumbing

    def _model(self, agent):
        if self._m is None:
            m = self._m = _Model(agent.server_params)
            # the derived limits, once, so they can be checked in the trace
            note(agent, "model", horizon=m.horizon, pass_max=round(m.pass_max, 1),
                 shot_max=round(m.shot_max, 1), post_clear=round(m.post_clear, 3),
                 mark=round(m.mark_dist, 2), kick_noise=round(m.kick_noise, 3),
                 cycles=m.cycles)
        return self._m

    def _share_ms(self, agent, m):
        """This decision's slice of the remaining match budget."""
        b = agent.budget
        cap = min(THINK_CAP_MS, b.decision_remaining_ms * WALL_FRACTION)
        rem = b.remaining_ms
        if math.isinf(rem):
            return cap
        left = m.cycles - agent.wm.cycle
        if left <= 0:
            left = 100
        return min(cap, SHARE * rem / left)

    @staticmethod
    def _opponents(wm):
        opps = [o for o in wm.opponents if o.uniform_number > 0]
        pos = np.array([_xy(o.position) for o in opps]).reshape(-1, 2)
        goalie = np.array([bool(o.is_goalie) for o in opps], dtype=bool)
        return opps, pos, goalie

    @staticmethod
    def _mates(wm):
        """Outfield teammates (the goalie never receives: back-pass rule)."""
        mates = [t for t in wm.teammates
                 if t.uniform_number > 0 and not t.is_goalie]
        pos = np.array([_xy(t.position) for t in mates]).reshape(-1, 2)
        return mates, pos

    # ON THE BALL

    def _on_ball(self, agent):
        m = self._model(agent)
        share = self._share_ms(agent, m)
        if share < CHEAP_SHARE_MS:
            note(agent, "cheap", share=round(share, 2))
            return super().on_ball(agent)
        clock = _Clock(agent.budget, share)

        wm = agent.wm
        ball = _xy(wm.ball.position)
        bvel = _xy(wm.ball.velocity)
        me_pos = _xy(wm.self.position)
        _, opps, goalie = self._opponents(wm)
        mates, mpos = self._mates(wm)
        line_x = wm.offside_line_x
        krate = wm.self.kick_rate

        # can we keep it one more cycle? (a second touch, or a hold)
        if len(opps):
            close = m.reach(np.linalg.norm(opps - ball, axis=1), m.tackle).min()
        else:
            close = 99.0
        keep1 = float(_sig((close - 1.0 - HOLD_OFFSET) / HOLD_SCALE))

        options = []    # (ev, kind, target, extra)

        # shots: one touch now, or a full-power kick() that may take 3
        targets, sdirs, sdist = m.shot_targets(ball)
        v1 = m.one_step_speed(sdirs, bvel, krate)
        p_one, m_one, f_one = m.shot_p(ball, targets, sdirs, sdist, v1, 0.0, opps, goalie)
        p_one = np.where(v1 >= SHOT_MIN_SPEED, p_one, 0.0)
        p_full, m_full, f_full = m.shot_p(ball, targets, sdirs, sdist, m.ball_max, 2.0,
                                          opps, goalie)
        p_full = p_full * keep1 ** 2
        ev_one, ev_full = m.shot_ev(ball, targets, p_one), m.shot_ev(ball, targets, p_full)
        i, j = int(np.argmax(ev_one)), int(np.argmax(ev_full))
        # extra carries the raw model terms, logged so the curves can be fitted
        options.append((float(ev_one[i]), "shoot", targets[i],
                        ("one", float(p_one[i]), float(m_one[i]), float(f_one[i]))))
        options.append((float(ev_full[j]), "shoot", targets[j],
                        ("full", float(p_full[j]), float(m_full[j]),
                         float(f_full[j] * keep1 ** 2))))

        # expandable options: passes and dribbles
        exp_pts, exp_p, exp_n, exp_ev, exp_kind, exp_info = [], [], [], [], [], []
        exp_m, exp_f = [], []

        cand = m.pass_targets(ball, mpos, line_x)
        if cand is not None:
            pts, recv, n, speed, dirs, dist = cand
            two = speed > m.one_step_speed(dirs, bvel, krate) + 1e-6
            delay = np.where(two, 1.0, 0.0)
            p, pm, pf = m.pass_p(ball, cand, delay, mpos, opps, goalie)
            pf = pf * np.where(two, keep1, 1.0)
            p = p * np.where(two, keep1, 1.0)
            exp_pts.append(pts)
            exp_p.append(p)
            exp_n.append(n + delay)
            exp_ev.append(m.pass_ev(ball, pts, p))
            exp_m.append(pm)
            exp_f.append(pf)
            exp_kind += ["pass"] * len(pts)
            exp_info += [(int(r), int(k)) for r, k in zip(recv, n)]

        # dribbles all round; facing elsewhere costs turn cycles first
        ang = np.arange(DRIBBLE_DIRS) * 360.0 / DRIBBLE_DIRS
        turns = m.turns(_norm_deg(ang - wm.self.body_direction),
                        np.linalg.norm(_xy(wm.self.velocity)))
        dend, dp, dn, dev, dm = m.dribbles(ball, ang, turns, opps, goalie)
        if len(dend):
            exp_pts.append(dend)
            exp_p.append(dp)
            exp_n.append(dn)
            exp_ev.append(dev)
            exp_m.append(dm)
            exp_f.append(np.ones(len(dend)))
            exp_kind += ["dribble"] * len(dend)
            exp_info += [None] * len(dend)

        # hold passes the turn: the same choices next cycle, with the
        # defence two cycles closer.
        later = self._value(m, ball, mpos, m.advance(opps, ball, 2.0),
                            goalie, line_x, 1, clock, body=wm.self.body_direction)
        # every option pays the living cost of the cycles it takes
        wait = m.time_cost(ball)
        hold_ev = (keep1 * ((later - PLY_COST) - HOLD_COST * (1 + self._holds))
                   - (1.0 - keep1) * float(m.danger(ball[0], ball[1])) - wait)
        options.append((hold_ev, "hold", ball, (keep1, float(close - 1.0))))
        cx = min(ball[0] + CLEAR_DIST, m.hl - 5.0)
        cy = math.copysign(m.hw * 0.6, ball[1] if ball[1] != 0 else 1.0)
        if keep1 < CLEAR_KEEP:
            # clear() picks its own direction; this landing point is a guess,
            # so it is an escape under pressure, not a planned pass
            cp = m.race(np.array([cx, cy]), mpos, opps)
            clear_ev = cp * float(m.value(cx, cy)) - (1 - cp) * float(m.danger(cx, cy))
            options.append((clear_ev, "clear", np.array([cx, cy]), None))

        depth = 1
        if exp_pts:
            pts = np.vstack(exp_pts)
            p = np.concatenate(exp_p)
            n = np.concatenate(exp_n)
            ev = np.concatenate(exp_ev) - wait * n
            raw_m, raw_f = np.concatenate(exp_m), np.concatenate(exp_f)
            mid = (pts + ball) / 2.0
            lose = (1.0 - p) * m.danger(mid[:, 0], mid[:, 1])

            # iterative deepening: re-score the best few with what the
            # player on the ball could do next against the closing defence
            for d in range(2, MAX_DEPTH + 1):
                if not clock.ok():
                    break
                new_ev = ev.copy()
                done = True
                for k in np.argsort(-ev)[:TOP_M]:
                    if not clock.ok():
                        done = False
                        break
                    if exp_kind[k] == "pass":
                        r = exp_info[k][0]
                        child = np.vstack([np.delete(mpos, r, axis=0), me_pos[None]])
                    else:
                        child = mpos
                    v = self._value(m, pts[k], child, m.advance(opps, pts[k], n[k]),
                                    goalie, line_x, d - 1, clock)
                    best_next = max(float(m.value(pts[k][0], pts[k][1])), v - PLY_COST)
                    new_ev[k] = p[k] * best_next - lose[k] - wait * n[k]
                if not done:
                    break
                ev, depth = new_ev, d

            self._last_expand = (pts, ev, p, exp_kind)
            k = int(np.argmax(ev))
            options.append((float(ev[k]), exp_kind[k], pts[k],
                            (exp_info[k], float(p[k]), float(raw_m[k]), float(raw_f[k]))))

        self._last_options = options
        ev, kind, target, extra = max(options, key=lambda o: o[0])

        # DATA COLLECTION (tools/collect only, never in a submitted version):
        # sometimes swap the policy's choice for a keep-the-ball action, under
        # pressure and outside our defensive third, and log the context.
        ctx = self._keep_context(m, ball, opps, wm)
        pressured = ctx["d1"] <= EXPLORE_PRESS * m.tackle
        cands = []
        if keep1 >= EXPLORE_P_FLOOR:
            cands.append((hold_ev, "hold", ball, (keep1, float(close - 1.0))))
        for i in range(len(dend)):
            if dp[i] >= EXPLORE_P_FLOOR:
                cands.append((float(dev[i]), "dribble", dend[i],
                              (None, float(dp[i]), float(dm[i]), 1.0)))
        c = wm.cycle
        eligible = (pressured and ball[0] >= -m.hl / 3.0 and kind not in ("shoot", "clear")
                    and len(cands) > 0 and self._explored < EXPLORE_MAX
                    and c - self._last_explore >= EXPLORE_GAP)
        policy_kind, policy_target = kind, target

        def same(a_kind, a_target, b_kind, b_target):
            return a_kind == b_kind and np.allclose(a_target, b_target)

        # rng seed: per decision (player, cycle), NOT a match seed; matches
        # are identified by their log name, and are not reproducible anyway
        rng_seed = int(wm.self.uniform_number) * 100003 + int(c)
        draw, prop = False, 1.0
        if eligible:
            rng = np.random.default_rng(rng_seed)
            if rng.random() < EXPLORE_P:
                draw = True
                ev, kind, target, extra = cands[int(rng.integers(len(cands)))]
                self._explored += 1
                self._last_explore = c
            # P(the executed action) under the collector's rule:
            # (1 - EXPLORE_P) * [it is the policy's choice]
            #   + EXPLORE_P / |cands| * [it is a candidate]
            prop = ((1.0 - EXPLORE_P) * same(kind, target, policy_kind, policy_target)
                    + EXPLORE_P / len(cands) * any(same(kind, target, o[1], o[2])
                                                   for o in cands))
        deviated = not same(kind, target, policy_kind, policy_target)
        self._holds = self._holds + 1 if kind == "hold" else 0
        self._act(agent, m, kind, target, extra, ev, depth, mates)
        if kind in ("dribble", "hold") or eligible:
            ctx["policy_kind"] = policy_kind
            ctx["policy_at"] = [round(float(policy_target[0]), 1), round(float(policy_target[1]), 1)]
            ctx["cands"] = [[o[1], round(float(o[2][0]), 1), round(float(o[2][1]), 1),
                             round(float(o[3][1] if o[1] == "dribble" else o[3][0]), 3)]
                            for o in cands]
            if kind == "dribble":
                v = target - ball
                a = math.degrees(math.atan2(v[1], v[0]))
                ctx["dir"] = round(a, 1)
                ctx["turns"] = float(m.turns(_norm_deg(a - wm.self.body_direction),
                                             np.linalg.norm(_xy(wm.self.velocity))))
                if ctx["opp_bearing"] is not None:
                    ctx["opp_rel"] = round(abs(_norm_deg(ctx["opp_bearing"] - a)), 1)
            # kind/at: the executed action; explore_draw: the 25% coin came up
            # (the drawn candidate may equal the policy's choice); deviated:
            # executed != policy choice; prop: P(executed action)
            note(agent, "keep_ctx", kind=kind,
                 at=[round(float(target[0]), 1), round(float(target[1]), 1)],
                 eligible=bool(eligible), explore_draw=draw, deviated=bool(deviated),
                 prop=round(float(prop), 4), n_cands=len(cands), rng_seed=rng_seed,
                 keep1=round(keep1, 3), **ctx)

    def _keep_context(self, m, ball, opps, wm):
        """Pressure features, computed with this agent's own model constants
        so a fitted curve can be used by the agent unchanged."""
        if len(opps):
            d = np.sort(np.linalg.norm(opps - ball, axis=1))
            near = opps[int(np.argmin(np.linalg.norm(opps - ball, axis=1)))] - ball
            bearing = round(math.degrees(math.atan2(near[1], near[0])), 1)
        else:
            d, bearing = np.array([99.0, 99.0]), None
        d1 = float(d[0]); d2 = float(d[1]) if len(d) > 1 else 99.0
        x = float(ball[0])
        return {"bx": round(x, 2), "by": round(float(ball[1]), 2),
                "region": "def" if x < -m.hl / 3 else ("att" if x > m.hl / 3 else "mid"),
                "d1": round(d1, 2), "d2": round(d2, 2),
                "n_tackle": int((d <= m.tackle).sum()),
                "n_press": int((d <= EXPLORE_PRESS * m.tackle).sum()),
                "umargin": round((d1 - m.tackle) / m.speed, 2),
                "opp_bearing": bearing,
                "body": round(float(wm.self.body_direction), 1)}

    def _value(self, m, origin, mates, opps, goalie, line_x, depth, clock, body=None):
        """Best EV for us when one of our players controls the ball at
        `origin`: a shot, or a pass or dribble searched `depth` plies deep.
        `body`: the ball carrier's facing if known (then dribbles pay their
        turn cycles from a standstill); unknown after a pass or dribble."""
        best = float(m.value(origin[0], origin[1]))
        targets, sdirs, sdist = m.shot_targets(origin)
        sp, _, _ = m.shot_p(origin, targets, sdirs, sdist, m.ball_max, CONTROL_DELAY,
                            opps, goalie)
        best = max(best, float(m.shot_ev(origin, targets, sp).max()))

        pts, p, n, ev, lose, recv = [], [], [], [], [], []
        cand = m.pass_targets(origin, mates, line_x)
        if cand is not None:
            pp, _, _ = m.pass_p(origin, cand, CONTROL_DELAY, mates, opps, goalie)
            mid = (cand[0] + origin) / 2.0
            pts.append(cand[0]); p.append(pp); n.append(cand[2] + CONTROL_DELAY)
            ev.append(m.pass_ev(origin, cand[0], pp))
            lose.append((1 - pp) * m.danger(mid[:, 0], mid[:, 1]))
            recv.append(cand[1])
        # carrying on with the ball: a fan of directions towards goal
        to_goal = math.degrees(math.atan2(-origin[1], m.hl - origin[0]))
        ang = to_goal + np.linspace(-60.0, 60.0, 5)
        turns = (np.zeros(5) if body is None
                 else m.turns(_norm_deg(ang - body), 0.0))
        dend, dp, dn, dev, _ = m.dribbles(origin, ang, turns, opps, goalie)
        if len(dend):
            pts.append(dend); p.append(dp); n.append(dn); ev.append(dev)
            lose.append((1 - dp) * m.danger(origin[0], origin[1]))
            recv.append(np.full(len(dend), -1))
        if not pts:
            return best
        pts, p, n = np.vstack(pts), np.concatenate(p), np.concatenate(n)
        wait = m.time_cost(origin)
        ev, lose, recv = np.concatenate(ev) - wait * n, np.concatenate(lose), np.concatenate(recv)
        best = max(best, float(ev.max()))
        if depth <= 1:
            return best

        for k in np.argsort(-ev)[:TOP_M]:
            if not clock.ok():
                break
            if recv[k] >= 0:
                child = np.vstack([np.delete(mates, recv[k], axis=0), origin[None]])
            else:
                child = mates
            v = self._value(m, pts[k], child, m.advance(opps, pts[k], n[k]),
                            goalie, line_x, depth - 1, clock)
            nxt = max(float(m.value(pts[k][0], pts[k][1])), v - PLY_COST)
            best = max(best, float(p[k] * nxt - lose[k] - wait * n[k]))
        return best

    def _act(self, agent, m, kind, target, extra, ev, depth, mates):
        at = [round(float(target[0]), 1), round(float(target[1]), 1)]
        tv = Vector2D(float(target[0]), float(target[1]))
        # m: opponent margin in cycles, f: the curve's other factors, so that
        # p = sigmoid((m - offset) / scale) * f can be refitted from traces
        if kind == "shoot":
            how, p, mg, f = extra
            note(agent, "shoot", ev=round(ev, 3), how=how, p=round(p, 3),
                 m=round(mg, 2), f=round(f, 3), at=at, depth=depth)
            if how == "one":
                Action.kick_one_step(agent, tv)
            else:
                Action.kick(agent, tv, 1)
        elif kind == "pass":
            (recv, n), p, mg, f = extra
            note(agent, "pass", ev=round(ev, 3), p=round(p, 3), m=round(mg, 2),
                 f=round(f, 3), to=int(mates[recv].uniform_number), n=n, at=at,
                 depth=depth)
            Action.kick(agent, tv, n, accept_late=2)
        elif kind == "dribble":
            _, p, mg, _ = extra
            note(agent, "dribble", ev=round(ev, 3), p=round(p, 3), m=round(mg, 2),
                 at=at, depth=depth)
            Action.dribble(agent, tv)
        elif kind == "hold":
            keep, mg = extra
            note(agent, "hold", ev=round(ev, 3), p=round(keep, 3), m=round(mg, 2),
                 depth=depth)
            Action.hold(agent)
        else:
            note(agent, "clear", ev=round(ev, 3), depth=depth)
        Action.clear(agent)      # the line that cannot fail

    # OFF THE BALL

    def _off_ball(self, agent):
        m = self._model(agent)
        share = self._share_ms(agent, m)
        if share < CHEAP_SHARE_MS:
            note(agent, "cheap", share=round(share, 2))
            return super().off_ball(agent)

        wm = agent.wm
        me = wm.self.uniform_number
        ball = _xy(wm.ball.position)
        bvel = _xy(wm.ball.velocity)
        path = np.vstack([ball, ball + bvel[None, :] * m.travel[:, None]])

        # the whole outfield squad, in one fixed order every player shares
        squad = [t for t in wm.teammates if t.uniform_number > 0 and not t.is_goalie]
        ids = [t.uniform_number for t in squad] + [me]
        pos = np.array([_xy(t.position) for t in squad] + [_xy(wm.self.position)])
        body = np.array([t.body_direction for t in squad] + [wm.self.body_direction])
        speed = np.array([np.linalg.norm(_xy(t.velocity)) for t in squad]
                         + [np.linalg.norm(_xy(wm.self.velocity))])
        order = np.argsort(ids)
        ids = [ids[i] for i in order]
        pos, body, speed = pos[order], body[order], speed[order]
        mine = ids.index(me)

        opps, opos, goalie = self._opponents(wm)
        obody = np.array([o.body_direction for o in opps])
        ospeed = np.array([np.linalg.norm(_xy(o.velocity)) for o in opps])
        our_t = self._steps_to_ball(m, path, pos, body, speed)
        opp_t = (self._steps_to_ball(m, path, opos, obody, ospeed) if len(opps)
                 else np.array([999]))

        holder = None
        if wm.kickable_teammate_existance:
            state, chaser = "ours", None
            d = np.linalg.norm(pos - ball, axis=1)
            if d.min() <= m.kickable + 0.3:
                holder = int(np.argmin(d))
            origin = ball
        else:
            chaser = min(range(len(ids)), key=lambda i: (our_t[i], ids[i]))
            if wm.kickable_opponent_existance:
                state = "theirs"
            else:
                state = "ours" if our_t[chaser] <= opp_t.min() else "theirs"
            origin = path[min(int(our_t[chaser]), m.horizon)]

        if chaser == mine:
            if wm.kickable_opponent_existance and self._tackle(
                    agent, m, ball, np.delete(pos, mine, axis=0), opos):
                return
            note(agent, "chase", steps=int(our_t[mine]), opp=int(opp_t.min()),
                 state=state)
            Action.intercept(agent)
            Action.go_to_point(agent, Tools.home_position(agent))
            return

        rest = [i for i in range(len(ids)) if i != chaser and i != holder]
        if state == "ours":
            self._support(agent, m, origin, pos, mine, opos, goalie, wm.offside_line_x)
        else:
            self._defend(agent, m, ball, pos, rest, mine, opps, opos)

    def _steps_to_ball(self, m, path, pos, body, speed):
        """Cycles each player needs to meet the ball on its current path
        (999 if not within the horizon). Same formula for everyone, so
        every player ranks the team identically."""
        d = np.linalg.norm(path[None, :, :] - pos[:, None, :], axis=2)
        to_ball = path[0] - pos
        face = np.degrees(np.arctan2(to_ball[:, 1], to_ball[:, 0]))
        turn = m.turns(_norm_deg(face - body), speed)
        reach = m.reach(d, m.kickable) + turn[:, None]
        ok = reach <= np.arange(path.shape[0])[None, :]
        return np.where(ok.any(axis=1), ok.argmax(axis=1), 999)

    def _tackle(self, agent, m, ball, mates, opos):
        """Tackle only when it very likely wins the ball, never as a foul,
        and send the ball towards their goal.

        rcssserver (Referee::checkFoul) calls even a foul=False tackle a
        charge, with probability foul_detect_probability, when an opponent
        on the ball stands no farther away than the ball, in front of the
        tackler along the line to the ball and within both players' widths
        of it. We cannot see whether he is dashing, so assume he is."""
        me = agent.wm.self
        p = float(me.tackle_probability)
        if p <= 0.0:
            return False
        sp = agent.server_params
        mine = _xy(me.position)
        to_ball = ball - mine
        bd = np.linalg.norm(to_ball)
        if bd > 1e-6:
            ux, uy = to_ball / bd
            width = 2.0 * sp.player_size + FOUL_SAFETY
            for o in agent.wm.opponents:
                rel = _xy(o.position) - mine
                along = rel[0] * ux + rel[1] * uy
                side = -rel[0] * uy + rel[1] * ux
                if (np.linalg.norm(rel) <= bd + FOUL_SAFETY and along >= -FOUL_SAFETY
                        and abs(side) <= width):
                    note(agent, "no_tackle", why="charge_risk",
                         opp=int(o.uniform_number))
                    return False
        # Expected value against carrying on the chase (the threshold comes
        # from the value model, not a set probability):
        #   won  - the ball flies off in the chosen direction and goes to
        #          whoever reaches it first (we are frozen, so not us)
        #   miss - we are frozen for tackle_cycles while he runs at our goal
        #   wait - he keeps it for now
        # Force, rcssserver Player::tackle (v12+): falls off linearly from
        # max_tackle_power straight ahead to max_back_tackle_power behind,
        # and by up to half again if the ball is behind the body.
        body = me.body_direction
        ball_off = abs(_norm_deg(math.degrees(math.atan2(to_ball[1], to_ball[0])) - body))
        rels = np.arange(-180.0, 180.0, 15.0)
        power = ((sp.max_back_tackle_power + (sp.max_tackle_power - sp.max_back_tackle_power)
                  * (1.0 - np.abs(rels) / 180.0))
                 * sp.tackle_power_rate * (1.0 - 0.5 * ball_off / 180.0))
        speed = np.minimum(power, sp.ball_accel_max)
        a = np.radians(body + rels)
        lands = ball + np.stack([np.cos(a), np.sin(a)], axis=1) * (speed / (1.0 - m.decay))[:, None]
        lands = np.clip(lands, [-m.hl, -m.hw], [m.hl, m.hw])
        v_dir = []
        for land in lands:
            win = m.race(land, mates, opos)
            v_dir.append(win * float(m.value(*land)) - (1 - win) * float(m.danger(*land)))
        k = int(np.argmax(v_dir))
        rel, v_won = float(rels[k]), v_dir[k]
        to_goal = np.array([-m.hl, 0.0]) - ball
        run = min(np.linalg.norm(to_goal), m.tackle_cycles * m.speed)
        gone = ball + to_goal / max(np.linalg.norm(to_goal), 1e-6) * run
        v_miss = -float(m.danger(*gone))
        v_wait = -float(m.danger(*ball))
        if p * v_won + (1 - p) * v_miss <= v_wait:
            note(agent, "no_tackle", why="ev", p=round(p, 2))
            return False

        note(agent, "tackle", p=round(p, 2), dir=rel, force=round(float(speed[k]), 2),
             won=round(v_won, 3), miss=round(v_miss, 3))
        Action.tackle(agent, float(rel), foul=False)
        return True

    def _support(self, agent, m, origin, pos, mine, opos, goalie, line_x):
        """Get open: a spot near home that a pass from the ball can reach
        before any opponent, as far forward as possible, not on top of a
        teammate."""
        home = _xy(Tools.home_position(agent))
        cands = np.vstack([home[None], _ring(home, 5.0, 8), _ring(home, 10.0, 8)])
        line = max(line_x, origin[0]) - 1.0
        ok = (np.abs(cands[:, 0]) < m.hl - 2) & (np.abs(cands[:, 1]) < m.hw - 2)
        ok &= (cands[:, 0] <= line) | (cands[:, 0] <= 0.0)
        cands = cands[ok]
        if not len(cands):
            cands = home[None]

        vec = cands - origin
        dist = np.maximum(np.linalg.norm(vec, axis=1), 1e-6)
        n = m.steps_for(dist, np.full(len(cands), m.ball_max))
        reachable = np.isfinite(n) & (dist <= m.pass_max)
        n = np.where(np.isfinite(n), np.minimum(n, m.horizon), m.horizon)
        speed = m.speed_for(dist, n)
        mg = m.margin(m.paths(origin, vec / dist[:, None], speed, int(n.max())),
                      n, CONTROL_DELAY, opos, goalie, m.kickable)
        p = np.where(reachable, _sig((mg - PASS_OFFSET) / PASS_SCALE), 0.0)

        others = np.vstack([np.delete(pos, mine, axis=0), origin[None]])
        crowd = np.exp(-np.linalg.norm(cands[:, None, :] - others[None], axis=2)
                       / CROWD_R).sum(axis=1)
        move = np.linalg.norm(cands - pos[mine], axis=1)
        score = (p * m.attack(cands[:, 0], cands[:, 1])
                 - 0.1 * crowd - 0.01 * move)
        k = int(np.argmax(score))
        note(agent, "support", p=round(float(p[k]), 2),
             at=[round(float(cands[k][0]), 1), round(float(cands[k][1]), 1)])
        Action.go_to_point(agent, Vector2D(float(cands[k][0]), float(cands[k][1])),
                           max_dash_power=self.dash_power(agent))
        Action.turn_to_ball(agent)

    def _defend(self, agent, m, ball, pos, rest, mine, opps, opos):
        """One cover between ball and goal, the rest mark the most dangerous
        free opponents. The cheapest assignment of players to jobs is the
        same for everyone, so nobody doubles up."""
        goal = np.array([-m.hl, 0.0])
        jobs, kinds = [], []
        to_goal = goal - ball
        d = np.linalg.norm(to_goal)
        if d > 1.0:
            jobs.append(ball + to_goal / d * min(COVER_DIST, 0.4 * d))
            kinds.append("cover")

        out = [i for i, o in enumerate(opps) if not o.is_goalie]
        if out:
            carrier = min(out, key=lambda i: np.linalg.norm(opos[i] - ball))
            threats = [i for i in out if i != carrier]
            threats.sort(key=lambda i: -float(m.danger(opos[i][0], opos[i][1])))
            for i in threats[:max(0, len(rest) - len(jobs))]:
                v = goal - opos[i]
                jobs.append(opos[i] + v / max(np.linalg.norm(v), 1e-6) * m.mark_dist)
                kinds.append("mark")

        job = None
        if jobs and mine in rest:
            cost = np.array([[np.linalg.norm(pos[r] - j) for j in jobs] for r in rest])
            best, best_perm = None, None
            for perm in permutations(range(len(rest)), min(len(rest), len(jobs))):
                c = sum(cost[perm[j], j] for j in range(len(perm)))
                if best is None or c < best - 1e-9:
                    best, best_perm = c, perm
            me_row = rest.index(mine)
            if me_row in best_perm:
                job = best_perm.index(me_row)

        power = 100.0 if ball[0] < DEFEND_FULL_X else self.dash_power(agent)
        if job is None:
            note(agent, "formation")
            Action.go_to_point(agent, Tools.home_position(agent),
                               max_dash_power=self.dash_power(agent))
        else:
            t = jobs[job]
            t = np.clip(t, [-m.hl + 1, -m.hw + 1], [m.hl - 1, m.hw - 1])
            note(agent, kinds[job], at=[round(float(t[0]), 1), round(float(t[1]), 1)])
            Action.go_to_point(agent, Vector2D(float(t[0]), float(t[1])),
                               max_dash_power=power)
        Action.turn_to_ball(agent)


class Agent2(StarterAgent):
    def on_ball(self, agent) -> None:
        return None

    def off_ball(self, agent) -> None:
        return None

# STARTER CODE ENDS HERE #
