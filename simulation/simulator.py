"""
Event-driven simulation engine (Section III-D).

The simulation advances by events, not fixed time steps.
A decision epoch is triggered when B UAVs complete their tasks.

Event types:
- UAV_COMPLETE: a UAV finishes its search pass
- The simulation runs until all M targets are found.
"""

import heapq
import time
import numpy as np

from .config import (
    NUM_UAVS, NUM_TARGETS, BATCH_SIZE, GCS_CELL, CELL_SIZE,
    GRID_ROWS, GRID_COLS, SIM_START_OFFSET, E_INIT,
)
from .grid_map import GridMap
from .uav import UAV
from .target import Target, create_targets
from .perception import PerceptionModule
from .memory import MemoryModule
# Cross-package import: simulation drives the agent's per-cycle policy.
import agent.decision as _decision
from agent.decision import init_knowledge


class Event:
    """A simulation event."""
    def __init__(self, time: float, event_type: str, data: dict = None):
        self.time = time
        self.type = event_type
        self.data = data or {}

    def __lt__(self, other):
        return self.time < other.time


class Simulator:
    """Event-driven SAR simulation engine."""

    def __init__(self, grid_map: GridMap, rng: np.random.Generator):
        self.grid = grid_map
        self.rng = rng
        self.current_time = SIM_START_OFFSET  # targets have been moving since t=0

        # Base position
        self.base_pos = grid_map.cell_centre(*GCS_CELL)

        # Create UAV fleet — all start at GCS
        self.uavs = []
        for i in range(NUM_UAVS):
            uav = UAV(i, self.base_pos)
            self.uavs.append(uav)

        # Create targets (trajectories pre-computed)
        self.targets = create_targets(grid_map, rng)

        # Modules
        self.perception = PerceptionModule(grid_map)
        self.memory = MemoryModule()

        # Clear any stale per-run telemetry in the decision module before
        # (re-)initialising RAG knowledge with the current INITIAL_CLUE.
        _decision.reset_stats()

        # RAG: retrieve domain knowledge once from the clue.
        init_knowledge()

        # Search statistics: (r,c) -> {"count": int, "last_time": float}
        self.search_stats = {}

        # Event queue (min-heap by time)
        self.event_queue = []

        # Tracking
        self.idle_count = NUM_UAVS  # all start idle
        self.epoch_count = 0
        self.all_found = False
        self._max_time = None
        self._max_epochs = None

        # Results
        self.discovery_times = {}    # target_id -> time
        self.total_cells_searched = 0
        # One end-to-end agent computation time per decision epoch.  This
        # wraps the full policy call (prompt construction, LLM reasoning,
        # parsing/retries, feasibility filtering, and assignment).  When an
        # LLM response is replayed from checkpoint, its recorded API latency
        # is restored so cache speed does not under-report computation cost.
        self.agent_computation_times = []

        # Dispatch log for case-study visualisation:
        # list of {"t": dispatch_time, "uav_id": int, "cell": [r, c],
        #          "completion_time": float}
        self.dispatch_log = []

    def run(self, decision_fn, max_time: float = 36000.0,
            max_epochs: int = None, time_budget: float = None) -> dict:
        """
        Run until all targets are found or ``max_time`` is reached.  When
        ``max_epochs`` decision cycles have been dispatched, no new cycle is
        opened, but every already-dispatched UAV is allowed to finish.

        Args:
            decision_fn: callable(grid_map, memory, state, uavs, num_unfound)
                         -> dict {uav_id: (row, col)}
            max_time: maximum simulation time in seconds (default 10 hours)
            max_epochs: maximum number of decision epochs (None = unlimited)
            time_budget: if set, results will include "success_within_budget"
                         (fraction of M targets found by t <= time_budget).
                         Defaults to max_time when None.

        Returns:
            dict with results
        """
        self._max_time = float(max_time)
        self._max_epochs = max_epochs

        # Initial epoch: all N UAVs idle, trigger first decision
        self._trigger_epoch(decision_fn)

        while self.event_queue and not self.all_found:
            event = heapq.heappop(self.event_queue)

            if event.time > max_time:
                break

            self.current_time = event.time

            if event.type == "UAV_COMPLETE":
                self._handle_uav_complete(event, decision_fn)

        return self._compile_results(max_time, time_budget=time_budget)

    def _trigger_epoch(self, decision_fn):
        """Execute one decision epoch: perceive → decide → advance → act.

        UAVs that were already active continue executing while the policy is
        computing.  UAVs selected in this epoch are dispatched only when the
        policy call finishes.
        """
        if (self._max_epochs is not None
                and self.epoch_count >= self._max_epochs):
            return

        self.epoch_count += 1
        decision_start_time = self.current_time

        # Perceive
        state = self.perception.compute_state(
            self.uavs, self.search_stats, self.current_time
        )

        # Count unfound targets
        num_unfound = sum(1 for t in self.targets if not t.found)
        if num_unfound == 0:
            self.all_found = True
            return

        # Decide — get assignments from the decision function.
        # Exclude retired UAVs: they're permanently grounded and can't
        # be assigned, so passing them to the LLM wastes tokens / API
        # calls and muddles the priority output.
        idle_uavs = [u for u in self.uavs
                      if u.status == "idle" and not u.retired]
        if not idle_uavs:
            return
        # Only dispatch up to B UAVs per epoch
        batch = idle_uavs[:BATCH_SIZE]

        llm_calls_before = len(_decision.get_stats()["llm_calls"])
        decision_started = time.perf_counter()
        assignments = decision_fn(
            self.grid, self.memory, state, batch, num_unfound
        )
        decision_wall_time = time.perf_counter() - decision_started

        # Fresh API latency is already part of decision_wall_time.  Cache
        # replay is nearly instantaneous, so add back only the stored latency
        # of newly consumed cached calls.  The resulting value is one total
        # computation overhead for the cycle; it intentionally does not split
        # LLM and optimizer time.
        new_llm_calls = _decision.get_stats()["llm_calls"][llm_calls_before:]
        replayed_api_time = sum(
            float(call.get("latency", 0.0) or 0.0)
            for call in new_llm_calls
            if call.get("from_cache")
        )
        measured_agent_time = decision_wall_time + replayed_api_time
        agent_computation_time = _decision.resolve_epoch_computation_time(
            self.epoch_count - 1, measured_agent_time, new_llm_calls)
        self.agent_computation_times.append(agent_computation_time)

        # Advance the event clock through the policy-computation interval.
        # Completion events from already-active UAVs remain physically
        # concurrent with the computation, but they cannot start a second
        # decision while this one is still in progress.
        decision_ready_time = decision_start_time + agent_computation_time
        completions_during_decision = self._advance_during_decision(
            decision_ready_time)
        if self.all_found:
            return
        if (self._max_time is not None
                and decision_ready_time > self._max_time):
            return

        # Act — dispatch assigned UAVs (only decrement idle_count for
        # UAVs that are actually dispatched)
        for uav_id, cell_rc in assignments.items():
            uav = next(u for u in self.uavs if u.id == uav_id)
            cell_centre = self.grid.cell_centre(*cell_rc)
            terrain_type = self.grid.terrain(*cell_rc)
            completion_time = uav.dispatch(cell_rc, cell_centre,
                                           self.current_time,
                                           terrain_type=terrain_type)
            self.idle_count -= 1

            # Record dispatch for case-study visualisation
            self.dispatch_log.append({
                "t": float(self.current_time),
                "uav_id": int(uav_id),
                "cell": [int(cell_rc[0]), int(cell_rc[1])],
                "completion_time": float(completion_time),
            })

            # Schedule completion event
            heapq.heappush(
                self.event_queue,
                Event(completion_time, "UAV_COMPLETE",
                      {"uav_id": uav_id, "cell": cell_rc})
            )

        # Recall UAVs with no feasible cells.  If retiring one of them makes
        # the smaller active fleet ready for dispatch, immediately recompute
        # the batch; otherwise an empty assignment can leave the event queue
        # empty even though the remaining UAVs still have feasible cells.
        recalled_count = 0
        for uav in batch:
            if uav.id not in assignments and uav.status == "idle" \
                    and not uav.retired:
                # Check if UAV has enough energy for ANY cell
                can_do_anything = False
                for rc in self.grid.get_traversable_cells():
                    d_to = float(np.linalg.norm(
                        uav.position - self.grid.cell_centre(*rc)))
                    d_back = float(np.linalg.norm(
                        self.grid.cell_centre(*rc) - self.base_pos))
                    if uav.can_reach_and_return(d_to, d_back,
                                                self.grid.terrain(*rc)):
                        can_do_anything = True
                        break
                if not can_do_anything:
                    uav.recall_to_base(self.base_pos, self.current_time)
                    # Retired UAV no longer counts as an active idle UAV
                    # for the batch-dispatch trigger.
                    self.idle_count -= 1
                    recalled_count += 1

        # If a complete trigger batch finished while this decision was being
        # computed, the newly idle UAVs may immediately start the next epoch
        # after the current batch has been dispatched.  Requiring at least one
        # completion in the interval avoids spinning when a policy returns no
        # assignments and no new physical event occurs.
        if ((completions_during_decision > 0 or recalled_count > 0)
                and self._epoch_trigger_ready()
                and (self._max_epochs is None
                     or self.epoch_count < self._max_epochs)):
            self._trigger_epoch(decision_fn)

    def _advance_during_decision(self, decision_ready_time: float) -> int:
        """Process active-UAV completions up to a decision's ready time.

        Returns the number of completion events processed.  No nested decision
        epoch is triggered here; scheduling resumes after the pending policy
        output has been applied.
        """
        advance_limit = decision_ready_time
        if self._max_time is not None:
            advance_limit = min(advance_limit, self._max_time)

        completed = 0
        while (self.event_queue
               and self.event_queue[0].time <= advance_limit):
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            if event.type == "UAV_COMPLETE":
                self._handle_uav_complete(event, decision_fn=None,
                                          allow_epoch_trigger=False)
                completed += 1
            if self.all_found:
                return completed

        self.current_time = advance_limit
        return completed

    def _handle_uav_complete(self, event, decision_fn,
                             allow_epoch_trigger: bool = True):
        """Handle UAV completing its search pass."""
        uav_id = event.data["uav_id"]
        cell_rc = event.data["cell"]
        uav = self.uavs[uav_id]

        # Perform detection — target positions are looked up at current_time
        detection = self._detect(cell_rc)

        # Update search stats
        if cell_rc not in self.search_stats:
            self.search_stats[cell_rc] = {"count": 0, "last_time": 0.0}
        self.search_stats[cell_rc]["count"] += 1
        self.search_stats[cell_rc]["last_time"] = self.current_time
        self.total_cells_searched += 1

        # Record observation
        self.memory.add_observation(uav_id, cell_rc, detection,
                                    self.current_time)

        # If detection, register every target currently in the cell as
        # found. This matches the behavioural principle that sub-units
        # stay together: seeing one member of a sub-unit on the ground
        # locates all members in the same cell. If no target is actually
        # in the cell the detection was a false alarm (nothing is
        # recorded against any target).
        if detection == 1:
            for t in self.targets:
                if not t.found and t.cell_at(self.current_time) == cell_rc:
                    t.found = True
                    t.discovery_time = self.current_time
                    self.discovery_times[t.id] = self.current_time
                    self.memory.record_find(t.id, cell_rc, self.current_time)

        # Mark UAV as idle
        uav.complete_task()
        self.idle_count += 1

        # Check if all found
        if all(t.found for t in self.targets):
            self.all_found = True
            return

        # Trigger new epoch when a full batch of UAVs is idle. Threshold
        # is min(BATCH_SIZE, active_fleet) where active_fleet excludes
        # UAVs permanently retired at base (energy exhausted). Without
        # this, small-N runs stall after partial retirements because
        # NUM_UAVS idle UAVs is impossible once some have retired.
        if allow_epoch_trigger and self._epoch_trigger_ready():
            self._trigger_epoch(decision_fn)

    def _epoch_trigger_ready(self) -> bool:
        """Whether the active fleet currently has a full idle batch."""
        active_fleet = sum(1 for u in self.uavs if not u.retired)
        if active_fleet == 0:
            return False
        trigger_threshold = min(BATCH_SIZE, active_fleet)
        return self.idle_count >= trigger_threshold

    def _detect(self, cell_rc) -> int:
        """Simulate sensor detection at cell_rc.

        Checks each unfound target's continuous (x,y) position at current_time
        to see if it falls within the searched cell.

        Returns 1 if detection (true positive or false alarm), 0 otherwise.
        """
        p_d, p_f = self.grid.sensor_params(*cell_rc)

        # Check if any unfound target is in this cell at current time
        target_present = any(
            t.cell_at(self.current_time) == cell_rc and not t.found
            for t in self.targets
        )

        if target_present:
            # True detection with probability p_d
            return 1 if self.rng.random() < p_d else 0
        else:
            # False alarm with probability p_f
            return 1 if self.rng.random() < p_f else 0

    def _compile_results(self, max_time: float,
                         time_budget: float = None) -> dict:
        """Compile simulation results.

        Adds (on top of the original fields) the following evaluation metrics:
          - time_to_first_target
          - time_to_50pct_targets
          - total_energy_consumed (J)
          - energy_per_found_target (J; inf if num_found == 0)
          - success_within_budget  (int) and success_rate_within_budget (float)
          - agent_computation_times, total_agent_computation_time,
            llm_latencies, llm_cache_hits, llm_api_calls,
            solver_latencies, priority_snapshots  (from the decision module)
        """
        found_targets = [t for t in self.targets if t.found]
        unfound_targets = [t for t in self.targets if not t.found]

        discovery_list = list(self.discovery_times.values())
        num_found = len(found_targets)

        max_discovery = max(discovery_list) if discovery_list else max_time
        mean_discovery = (float(np.mean(discovery_list))
                          if discovery_list else max_time)
        time_to_first = min(discovery_list) if discovery_list else max_time

        # Penalised mission completion time — the metric the paper
        # formulation ``max_m T_m`` requires (T_m = max_time for
        # unfound targets). Discovery times already come from the event clock,
        # which includes policy-induced dispatch waits while allowing active
        # UAV execution to overlap with policy computation.
        if num_found < NUM_TARGETS:
            mission_completion_time = float(max_time)
        else:
            mission_completion_time = float(max_discovery)

        # Agent computation is reported independently.  It must not be added
        # again here because its scheduling effect is already represented in
        # the event-clock discovery times above.
        total_agent_computation_time = float(
            sum(self.agent_computation_times))

        # time-to-50% — take the ceil(M/2)-th smallest discovery time;
        # if we haven't reached 50% yet, fall back to max_time.
        target_idx_50pct = int(np.ceil(NUM_TARGETS / 2))
        if num_found >= target_idx_50pct:
            time_to_50pct = float(sorted(discovery_list)[target_idx_50pct - 1])
        else:
            time_to_50pct = float(max_time)

        total_remaining_energy = sum(u.energy for u in self.uavs)
        total_energy_budget = NUM_UAVS * E_INIT
        total_energy_consumed = total_energy_budget - total_remaining_energy
        energy_per_found = (total_energy_consumed / num_found
                            if num_found > 0 else float("inf"))

        # Success within budget: how many targets had t <= budget
        budget = time_budget if time_budget is not None else max_time
        num_found_within = sum(1 for t in discovery_list if t <= budget)
        success_rate_within = num_found_within / float(NUM_TARGETS)

        # Pull decision-module telemetry (LLM/solver latencies, cache stats,
        # priority snapshots). Kept as lists so analyze.py can compute
        # mean/p50/p95 however it likes.
        dec_stats = _decision.get_stats()

        return {
            "all_found": self.all_found,
            "num_found": num_found,
            "num_unfound": len(unfound_targets),
            "max_discovery_time": max_discovery,
            "mission_completion_time": mission_completion_time,
            "mean_discovery_time": mean_discovery,
            "discovery_times": dict(self.discovery_times),
            "total_epochs": self.epoch_count,
            "total_cells_searched": self.total_cells_searched,
            "energy_utilisation": 1.0 - (total_remaining_energy /
                                          total_energy_budget),
            # ── Extended evaluation metrics ────────────────────────
            "time_to_first_target": time_to_first,
            "time_to_50pct_targets": time_to_50pct,
            "total_energy_consumed": total_energy_consumed,
            "energy_per_found_target": energy_per_found,
            "time_budget": budget,
            "num_found_within_budget": num_found_within,
            "success_rate_within_budget": success_rate_within,
            "agent_computation_times": list(self.agent_computation_times),
            "total_agent_computation_time": total_agent_computation_time,
            "llm_latencies": dec_stats["llm_latencies"],
            "llm_input_tokens": dec_stats["llm_input_tokens"],
            "llm_output_tokens": dec_stats["llm_output_tokens"],
            "llm_cache_hits": dec_stats["llm_cache_hits"],
            "llm_api_calls": dec_stats["llm_api_calls"],
            "llm_calls": dec_stats["llm_calls"],
            "llm_illegal_outputs": dec_stats["llm_illegal_outputs"],
            "llm_impassable_drops": dec_stats["llm_impassable_drops"],
            "llm_fallback_count": dec_stats["llm_fallback_count"],
            "llm_parse_retry_events": dec_stats["llm_parse_retry_events"],
            "solver_latencies": dec_stats["solver_latencies"],
            "priority_snapshots": dec_stats["priority_snapshots"],
            "dispatch_log": list(self.dispatch_log),
        }
