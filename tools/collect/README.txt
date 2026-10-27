Keep-the-ball data collection (dribble + hold), training matches only.

agent.py   v8b (tools/versions/v8b/agent.py) plus, in Agent1._on_ball only:
           - a "keep_ctx" trace note after every dribble/hold decision and every
             decision eligible for exploration: ball x/y, region (thirds),
             nearest/second opponent distance d1/d2, opponents within tackle_dist
             and 2*tackle_dist, uncapped margin (d1 - tackle_dist)/speed,
             nearest opponent bearing, body, dribble direction/turn cycles,
             angle of the nearest opponent to the dribble direction, keep1,
             eligible/explored flags, selection probability, candidate count,
             and the cycle (the exploration RNG seed is unum*100003 + cycle);
           - bounded exploration (constants EXPLORE_*):
               eligible only if an opponent is within 2*tackle_dist of the ball,
               the ball is NOT in our defensive third (x >= -pitch_half_length/3),
               the policy did not choose a shot or a clearance,
               fewer than 15 explored decisions so far for this player,
               and >= 10 cycles since this player's last explored decision;
               candidates: hold and each dribble direction whose current model
               P(keep) >= 0.3 (never a hopeless action);
               with probability 0.25 the choice is replaced by a uniformly
               drawn candidate.
           No file writes; everything goes through note() into the trace.
run.sh     runs KC<first>..KC<last> vs full, 3 at a time, stops at the first
           invalid match / forfeit / cut-off / traceback / policy violation,
           writes runlogs/keep/manifest_KC*.txt (md5s, config, match stamps).
Labels and rows: tools/keep_data.py.  Fitting: tools/fit_keep.py (stage 3).
