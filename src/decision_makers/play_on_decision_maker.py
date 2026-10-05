"""play_on_decision_maker.py -- the one place student code is invoked."""
import os
import sys
import threading
import time
import traceback

from src.student.framework.policy import PolicyViolation, check
from src.student.framework.runtime import (Budget, BudgetExceeded,
                                           BudgetView, load_student)
from src.student.framework.world import AgentView
from src.student.framework import trace


class PlayOnDecisionMaker:
    def __init__(self, agent):
        self.use_student = agent.use_starter_code
        self.budget = None
        # one student decision at a time
        self._deciding = threading.Lock()
        self._startup_ms = 0.0
        if self.use_student:
            started = time.process_time()
            try:
                cls, name = load_student()
                self.student = cls()
            except Exception:
                tb = traceback.format_exc()
                agent.logger.error("STUDENT AGENT FAILED TO LOAD\n" + tb)
                sys.stderr.write("!! student agent failed to load\n" + tb)
                sys.stderr.flush()
                os._exit(1)
            # loading is charged to the budget like any other thinking
            self._startup_ms = (time.process_time() - started) * 1000.0
            agent.logger.info(f"student agent: {name}")
        else:
            from src.decision_makers.kick_decision_maker import KickDecisionMaker
            from src.decision_makers.move_decision_maker import MoveDecisionMaker
            self.kick_decision_maker = KickDecisionMaker(agent)
            self.move_decision_maker = MoveDecisionMaker(agent)

    def make_decision(self, agent):
        kickable = agent.wm.self.is_kickable

        if not self.use_student:
            if kickable:
                self.kick_decision_maker.make_decision(agent)
            else:
                self.move_decision_maker.make_decision(agent)
            return

        if self.budget is None:
            self.budget = Budget(agent.wm.our_team_name,
                                 agent.wm.self.uniform_number,
                                 startup_ms=self._startup_ms)
            agent.budget = BudgetView(self.budget)   # read-only for students

        before = len(agent.actions)

        if self.budget.exhausted or self.budget.benched:
            # no thinking time left for the rest of the match (or cut off too
            # often). The player emits nothing and coasts.
            trace.record(agent, [], self.budget)
            return

        if not self._deciding.acquire(blocking=False):
            agent.logger.error("BUDGET: previous decision still running, "
                               "skipping this cycle")
            trace.record(agent, [], self.budget)
            return

        view = AgentView(agent)
        try:
            with self.budget:
                if kickable:
                    self.student.on_ball(view)
                else:
                    self.student.off_ball(view)
        except BudgetExceeded:
            # BaseException, so `except Exception` below would miss it
            agent.logger.error("BUDGET: decision cut short at the ceiling")
        except Exception as e:
            agent.logger.error(f"STUDENT CODE RAISED {type(e).__name__}: {e}",
                               exc_info=True)
        finally:
            self._deciding.release()

        # Whatever the student produced, filter it before it reaches the
        # proxy, and trace what actually got sent rather than what was asked
        # for.
        produced = agent.actions[before:]
        del agent.actions[before:]
        try:
            kept = check(agent, produced)
        except PolicyViolation as e:
            # POLICY=raise: nothing from this decision is sent
            agent.logger.error(f"POLICY: {e}")
            kept = []
        agent.actions.extend(kept)
        trace.record(agent, kept, self.budget)