# Scratch probe: starter behaviour plus notes recording runtime facts.
from src.student.interface import StarterAgent, note


class Agent1(StarterAgent):
    def __init__(self):
        super().__init__()
        self.first = True

    def _probe(self, agent):
        wm = agent.wm
        sp = agent.server_params
        if self.first:
            self.first = False
            note(agent, "params", half_time=sp.half_time, step=sp.simulator_step,
                 halfs=sp.nr_normal_halfs, catchable=round(sp.catchable_area, 3),
                 kickable=round(sp.kickable_area, 3), pen_x=sp.our_penalty_area_line_x,
                 pen_hw=sp.penalty_area_half_width, real_speed_max=sp.real_speed_max,
                 tm=[(t.uniform_number, t.is_goalie) for t in wm.teammates],
                 op=[(o.uniform_number, o.is_goalie) for o in wm.opponents],
                 me=wm.self.uniform_number, budget=agent.budget.remaining_ms)
        note(agent, "probe", off=round(wm.offside_line_x, 2),
             their_def=round(wm.their_defense_line_x, 2),
             their_def_p=round(wm.their_defense_player_line_x, 2),
             ball_x=round(wm.ball.position.x(), 2),
             reach=[t.ball_reach_steps for t in wm.teammates],
             oreach=[o.ball_reach_steps for o in wm.opponents],
             it=[wm.intercept_table.self_reach_steps,
                 wm.intercept_table.first_teammate_reach_steps,
                 wm.intercept_table.first_opponent_reach_steps],
             body=round(wm.self.body_direction, 1), krate=round(wm.self.kick_rate, 4),
             card=wm.self.card, foulp=round(wm.self.foul_probability, 3),
             tacklep=round(wm.self.tackle_probability, 3))

    def on_ball(self, agent):
        self._probe(agent)
        super().on_ball(agent)

    def off_ball(self, agent):
        self._probe(agent)
        super().off_ball(agent)
