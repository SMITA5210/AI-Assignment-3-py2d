# Hand-off: COL333 Assignment 3 agent (py2d 5-a-side), state at 2026-10-27

This file is for a new assistant session that continues this work. It records the
current state, the rules, how to run and evaluate, what has been tried, and the
planned next steps. Verify facts against the repo before relying on them (paths and
checksums are given so you can).

---

## 1. The task and its hard rules

- **Deliverable:** `src/student/agent.py`, class `Agent1` (4 outfield players; the
  goalie, set plays and penalties are run by the framework). Only `agent.py` plus a
  `report.txt` (<= 2 pages, 11pt Arial, first line = student name(s) + entry number)
  are submitted. `Agent2` in that file is a do-nothing dummy and **must stay unchanged**.
- **Grading:** about 70 % fixed head-to-head matches against several TA agents (`full` =
  HELIOS reference is one), about 30 % tournament (round robin 3/1/0, then knockout,
  best of 3). Tie-breakers: goal difference, goals scored, fewest yellow cards (direct
  red = 2.5), fewest fouls.
- **Method constraint (instructor):** use search / adversarial reasoning and concepts
  from the course; methods with a "fundamentally different mathematical underpinning"
  (e.g. learning: RL, neural nets) may be excluded. Optimisations are allowed.
  Do not hard-code constraints: derive them from `server_params`, `agent.budget` or data.
- **Budget:** 75 s CPU per player per match, 1 s wall per decision (5 cut-offs bench a
  player, 1.5 s = forfeit). The grader's machine may be slower: keep time use adaptive.
- **Packages:** only the starter environment; **scipy is banned** (numpy is fine).
  The sandbox (armed when the agent is given as `--left path/agent.py`) blocks file
  writes and many imports. The action policy strips `bhv_*`, `helios_*` and
  communication. Keep `note()` tracing and safe error handling (`except Exception`,
  never `BaseException`). A note field that is not JSON-safe makes the trace print a
  traceback, which voids the match.
- **Course concepts used and where (slides at `/mnt/d/OneDrive/Desktop/SEM 1(mtech)/AI/`):**
  L03 iterative deepening (p53); L04 alldiff / soft constraints (p8, p10); L05 beam
  search (p17), hill climbing (p11); L07 probability; L08 living reward (p8); L10 minimax,
  alpha-beta and move ordering (p46), evaluation functions (p43-45), expectimax (p47-51),
  MCTS/UCT (p53-64), multi-agent (p66); L12 MLE (p11-14), MAP (p9-10, p15), counting
  (p16-25), Laplace pseudo-counts as a prior (p27), decomposition (p28-29), EM (p31-38).

## 2. How the user wants to work (important)

- For any big design change: **inspect, show a concise plan, and wait for approval**.
  The user approves with short messages ("ok", "do it", "proceed").
- **Never promote** a candidate into `src/student/agent.py` without explicit approval.
  Never modify production or saved versions; build candidates as new files.
- **Predeclare** evaluation rules and thresholds in a plan file *before* running
  matches; **never change thresholds after seeing results**. Disclose any reuse of
  evaluation data for design.
- Keep each change attributable (one change per candidate where practical).
- Report honestly with numbers and uncertainty; say when something failed. Do not
  claim completion without evidence.
- Ask before deleting anything that is not a clearly regenerable cache.
- Git: see section 4 (instructor repo is read-only).

## 3. Environment and commands

- WSL2 Ubuntu, repo at `/home/smita/py2d`. Python: `/home/smita/miniconda3/envs/ai-a3/bin/python`
  (the default `python3` lacks protobuf; use the env path, or `PYTHON=...` for scripts).
- Health check: `$PY a3.py doctor` (expect "all good").
- One match: `$PY a3.py match ME OPP --left path/agent.py --right full -p 6000`
  (team names must differ; use `-p 6000/6010/6020` to run 3 in parallel).
  Output: replay `logs/<stamp>_ME_vs_OPP.rcg`, decision traces
  `runlogs/trace/<stamp>_ME_vs_OPP/ME/<unum>.jsonl`, team stdout `runlogs/<stamp>_ME.out`.
- Interleaved evaluation with stop rules (invalid match, forfeit, cut-off, traceback,
  policy violation): `PYTHON=$PY tools/eval_run.sh TAG file1:NAME1 file2:NAME2 ...`
  -> `runlogs/eval/TAG.tsv`. A 24-v-24 run takes about 25 minutes.
- Report with bootstrap: `python3 tools/eval_report.py runlogs/eval/TAG.tsv --boot BASE CAND`.
- Server randomness cannot be seeded (only `player::random_seed`, which fixes player
  types, exists); matches are not reproducible. Balance conditions by interleaving.
- Storage: each match writes about 27 MB (about 20 MB of framework debug logs under
  `runlogs/<match>_<team>/`). The WSL disk file lives on Windows C:, which filled up once.
  Old debug-log folders were deleted with the user's approval (traces, replays, `.out`,
  timing, eval files were kept). Do not run hundreds of matches without checking free
  space on C: (`df -h /mnt/c`).

## 4. Git rules

- `origin` = the user's own repo `https://github.com/SMITA5210/AI-Assignment-3-py2d.git`.
  **All commits and pushes go here only**, and only when the user asks.
- `upstream` = the instructor's repo `https://github.com/TheAbry/COL333-Assignment-3-py2d-fork.git`.
  **Read-only.** Its push URL is set to `DISABLED-read-only-instructor-repo` on purpose.
  To take upstream changes: `git fetch upstream && git merge upstream/main` (local), test,
  then push to `origin` if the user wants.
- Latest commits on `main` (= origin/main): `e5503ce` promote v14; `15e836d` research
  tools and versions; `d3cd2d6` merge of upstream (keeper rewrite `b990c80`, tackle
  docstring `b48f4c2`).
- `runlogs/` and `logs/` are git-ignored; `runlogs/eval/` and two `runlogs/keep/manifest_*`
  files were force-added. Commit messages end with a `Co-Authored-By` line.

## 5. Current state

| What | File | md5 (first 12) |
|---|---|---|
| **Production (promoted 2026-10-27)** | `src/student/agent.py` = v14 | `2166ad360dce` |
| Previous production | `tools/versions/v11/agent.py` | `db7aef6ffb15` |
| Research branch (all stage switches off = v14 behaviour) | `tools/versions/v15/agent.py` | `25903939c7a6` |
| v15 stage-1 snapshot (parent for stage 2) | `tools/versions/v15_s1/agent.py` | `1dcab1e3824f` |
| Rejected stage-2 pass fit | `tools/versions/v15_s2a/agent.py` | `dfad10c3c06d` |
| Old baseline | `tools/versions/v3/agent.py` (+ `v3_backup_prepromotion`) | `f52543749c3b` |

All versions v1-v15 are kept in `tools/versions/<v>/agent.py`.

**Under the current framework (new upstream keeper), production v14 vs `full`:**
24 matches 2-4-18, 0.42 points/match, 0.29 goals for, 1.75 against; v11 was 0-0-24
in the same interleaved run. (Under the old keeper v14 had 0.69 pts/match over 48.)
Every version still loses most matches to `full`.

## 6. How the agent works (v14; line numbers approximate)

- `_Model` (from `server_params`, line ~118): ball flight (`steps_for`, `paths`,
  decay 0.94), player reach with turn cost (`reach`, `turns`), **margin** = opponents'
  best interception time minus ball time (`margin`), success curves
  `P = sigmoid((margin - OFFSET) / SCALE) * other factors` for shots (`shot_p`),
  passes (`pass_p`), dribbles (`dribbles`); evaluation `attack`, `danger`,
  `value = attack - 0.2 * danger`; tackle outcome along the ball path (`touch_outcome`).
- Fitted constants (MAP, see comments in the file): `PASS -3.2/0.82`,
  `SHOT -1.85/1.372` (v14), `DRIBBLE -4.0/1.09`, `HOLD -1.1/0.46`.
- On the ball (`_on_ball`, `_value`): expected-value search over shots, passes (ring,
  lead and space points, 3 speeds), dribbles (12 directions), hold, clear; iterative
  deepening to 3 plies with a top-6 beam, under a per-decision time share
  (`_share_ms`, `_Clock`: 0.9 x budget left / cycles left, capped at 60 ms).
- Off the ball (`_off_ball`): consistent chaser election (`_steps_to_ball`), support
  spots near home (`_support`), defence = cover + marks by exhaustive alldiff assignment
  (`_defend`); `_goal_side` / `_defend_goalside` exist but `FIX_GOALSIDE = False`.
- Tackles (`_tackle`): charge rule as rcssserver computes it (opponent kickable, dashing
  assumed, facing within 90 deg of our line to the ball; `red_card_probability = 0` so no
  cards), outcome = who reaches the ball first along its real path (tackle push is added
  to the ball's velocity, tackler frozen `tackle_cycles` = 10), same-cycle carrier kick
  (`SAME_CYCLE_KICK = 0.66`, measured), **timing**: never tackle a controlled ball,
  tackle a loose ball an opponent would reach first (`TACKLE_TIMING`), and our goalie
  collecting it in our box = back-pass foul (`BACK_PASS_COST = 21/24`, measured).

## 7. Verified simulator facts worth knowing

- `ball_decay` 0.94, `ball_speed_max` 3, `ball_accel_max` 2.7, kickable area 1.085,
  catchable 1.3, `tackle_dist` 2, `tackle_cycles` 10, `max_back_tackle_power` 0,
  `tackle_power_rate` 0.027, `foul_detect_probability` 0.5, `red_card_probability` 0,
  `player_decay` 0.4, `illegal_defense_number` 0 (off); a match is 1500 cycles.
- Server physics (`MPObject::_inc`): accel capped, **added** to velocity, speed capped,
  `pos += vel`, then `vel *= decay`.
- Replay flags: KICK `0x2`, CATCH `0x10`, TACKLE `0x1000`, TACKLE_FAULT `0x2000`, yellow
  card `0x40000`. **The TACKLE flag stays set for all 10 freeze cycles: count it as a touch
  only in its first cycle** (fixed in `tools/keep_data.py` and `tools/metrics.py`).
- A touch shown at cycle c comes from the command decided at c-1.
- The goalie decision maker (`src/decision_makers/goalie_decision_maker.py`) is shared by
  both teams. Upstream's keeper rewrite (27 Oct) moves the keeper to a box corner after a
  catch and passes to an open teammate or kicks long; all evaluations before that used
  the old keeper and must not be pooled with new ones.

## 8. Tools (all in `tools/`)

| Tool | Purpose |
|---|---|
| `eval_run.sh`, `eval_report.py` | interleaved evaluation with stop rules; per-version tables and bootstrap |
| `test_agent.py` -> `test_v10.py` ... `test_v15.py` | chained offline tests; each version's test runs the earlier chain. Run `test_v14.py src/student/agent.py` for production, `test_v15.py` for the research branch |
| `keep_data.py` | replay parser, event-based outcome labels ("who touches next") |
| `metrics.py` | per-team match metrics from replays |
| `tackle_report.py`, `recoveries.py`, `goals.py` | tackle outcomes; what happens after recoveries; how goals were conceded |
| `fit_probs.py` (old), `fit_value.py`, `fit_shots.py`, `fit_keep.py`, `fit_passes.py` | MAP fits with predeclared held-out gates |
| `collect/` | data collector agent with bounded exploration (training matches only) |
| `versions/` | every agent version |

Evaluation records, plans and fit outputs: `runlogs/eval/` (`plan_*.txt` are the
predeclared rules; `*.tsv` are per-match results; `design_v15.txt` is the current plan).

## 9. Evaluation protocol in use

- Fresh matches only; versions interleaved in each batch of 3; opponent `full`.
- **Verdict rule (used for every promotion):** PASS if candidate points/match > baseline
  AND the 90 % match-bootstrap interval (4000 resamples, seed 7) of the difference excludes 0
  AND goals against/match not worse by more than 0.25; FAIL if points are lower or goals
  against worse by more than 0.5; otherwise INCONCLUSIVE. Plus operational validity (all
  valid, no tracebacks / policy violations / forfeits / error notes, planning < 100 %).
- Noise: points/match has a standard deviation of about 1.0 per match. Two identical teams
  trigger "worse by > 0.5" in 16.6 % of 12-v-12 runs, 6.9 % of 24-v-24, 2.9 % of 36-v-36.
  So per-stage screens are **24 v 24 regression warnings**, not automatic rejections.
- Offline fits use held-out matches split by whole match, gates fixed before fitting.
  **Caveat learned twice:** held-out gates cannot detect selection bias (data only contain
  actions the old policy chose). Always check smoke matches for behaviour shifts.

## 10. History: what worked and what did not (all vs `full`)

| Version | Change | Result |
|---|---|---|
| v1-v3 | search agent, MAP-fitted pass/dribble/hold curves, tackle force model | v3 0.12 pts/m (58 m) |
| v6 | defensive expectimax over joint positions | rejected (worse) |
| v7 / v7b | joint support runs | rejected (more goals conceded / no attack) |
| v8 / v8b | refitted position values + living cost | rejected |
| v10a/b | real charge rule / goal-side defence | rejected (tackles poked the ball back; shape unchanged) |
| **v11** | tackle outcome along the ball path, same-cycle kick | PASS vs v3, confirmation INCONCLUSIVE; promoted by the user on pooled evidence (0.33 vs 0.07 pts/m) |
| v12 | tackle timing | inconclusive vs v11 (tackling much cleaner) |
| v13 | back-pass awareness | inconclusive vs v11 |
| **v14** | fitted shot curve | PASS, then INCONCLUSIVE (edge), then PASS under the new keeper -> promoted |
| v15 stage 1 | switches only (= v14) | accepted (its 12-v-12 REJECT was a measured false alarm) |
| v15 stage 2 (s2a) | pass model with motion + pressure features | all offline gates passed, **but smoke 0-14**: selection-bias extrapolation (tight passes rated 0.87, chosen 28 %, succeeded 38 %) -> rejected, switched off |

Lessons: big tactical overhauls lost; small model-grounded fixes based on real server
rules won. Watch for selection bias in fits and for tool bugs (two were found: fixed-window
labels blaming later events, and the persistent TACKLE flag).

## 11. Open diagnosis (where the agent loses)

- After recovering the ball we usually lose it again quickly (median 0 cycles kept, about
  60 % lost within 5 cycles; HELIOS keeps theirs a median 12 cycles).
- Few shooting chances: about 99 % of our on-ball decisions have a best shot chance below 0.2.
- In shooting range the agent often **holds** instead of shooting: the hold option's
  lookahead assumes shooting next cycle with the defence barely closer (planned stage 3).
- Passes under pressure are overrated by the current model (observed 0.73 vs predicted 0.87
  when an opponent is within 3 m of the passer).
- About 0.17 back-pass fouls per match remain (close races teammate vs goalie).

## 12. Next steps (agreed plan; `runlogs/eval/design_v15.txt`)

Research branch `tools/versions/v15/agent.py` has switches `STAGE_EVAL`, `STAGE_LOOKAHEAD`,
`STAGE_RESPONSE`, `STAGE_CSP`, `STAGE_OFFBALL` (all `False`). Parent for comparisons is v14
(= production now). The user asked for each stage: implementation + tests + 3-match smoke
+ 24-v-24 screen (warning) + report with metrics, uncertainty, runtime, checksums, verdict.

1. **Stage 2b (next, proposed and awaiting the user's go):** keep the current pass curve
   (`PASS_OFFSET -3.2`, `PASS_SCALE 0.82`) and the static margin; add **only** the passer-
   pressure term `x` from `_Model.pass_features` (already in v15) and fit only its weight
   `PASS2_PRESS` with `tools/fit_passes.py` (adapt the model set). Gates: the existing five
   (G1-G5 in the script docstring) **plus a guard gate fixed before fitting: the new
   probability must never exceed the current one for any pass** (pressure can only make the
   agent more careful). Then tests, smoke (check pass success and choice distribution against
   v14 before screening), 24-v-24 screen vs v14.
2. Stage 3: lookahead consistency (hold valued with opponents closing at measured speed and
   tackle risk; shoot-now vs shoot-later on the same kick model); search reuse between cycles.
3. Stage 4: one-step expectimax over the nearest opponents' responses (press, block shot
   lane, cut pass lane, close receiver) with probabilities counted from HELIOS replays.
4. Stage 5: constraint-based support coordination (hard: legality, distinct spots, rest
   defence; soft: spacing, lanes, width), exhaustive with pruning.
5. Stage 6: off-ball lookahead with a shared evaluator (shot chance after receiving),
   through balls using receiver velocity, defensive anticipation of HELIOS passes.
6. Final: predeclared 24 x 3 evaluation (v15 vs v14 vs v11) and confirmation after a PASS;
   promotion only with the user's approval.
7. Still to do eventually: `report.txt` (needs the user's name and entry number); consider
   removing the `_last_options` / `_last_expand` debug attributes before submission.

## 13. Quick start for the new session

```bash
cd /home/smita/py2d
PY=/home/smita/miniconda3/envs/ai-a3/bin/python
git status && git log --oneline -3          # expect e5503ce on main, synced with origin
md5sum src/student/agent.py                 # 2166ad360dce... (v14)
$PY a3.py doctor                            # all good
$PY tools/test_v14.py src/student/agent.py  # production tests
$PY tools/test_v15.py                       # research branch tests
cat runlogs/eval/design_v15.txt             # the agreed stage plan
```
Then propose stage 2b's concrete plan to the user and wait for approval.
