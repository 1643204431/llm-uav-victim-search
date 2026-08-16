import unittest
from unittest.mock import patch

import numpy as np

import agent.decision as decision
from simulation.grid_map import GridMap
from simulation.perception import PerceptionModule
from simulation.uav import UAV


class ParseRetryPolicyTests(unittest.TestCase):
    def setUp(self):
        decision.reset_stats()
        decision.set_cache_enabled(False)
        decision.set_current_run_id("test_parse_policy")
        self.grid_map = GridMap()

    def tearDown(self):
        decision.set_current_run_id(None)

    def test_priorities_retry_then_accept_valid_json(self):
        responses = [
            "not json",
            '{"priorities": {"1,1": 0.9}}',
        ]
        with patch("agent.decision._call_llm", side_effect=responses) as call:
            priorities, _ = decision._call_llm_and_parse_priorities(
                "system", "user", self.grid_map, "deepseek")

        self.assertEqual(call.call_count, 2)
        self.assertAlmostEqual(priorities[1, 1], 0.9)
        self.assertEqual(decision.get_stats()["llm_fallback_count"], 0)
        events = decision.get_stats()["llm_parse_retry_events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "tiny_response")

    def test_priorities_fall_back_after_one_retry(self):
        with patch("agent.decision._call_llm",
                   side_effect=["not json"] * 2) as call:
            priorities, _ = decision._call_llm_and_parse_priorities(
                "system", "user", self.grid_map, "deepseek")

        self.assertEqual(call.call_count, 2)
        self.assertEqual(decision.get_stats()["llm_fallback_count"], 1)
        self.assertEqual(
            len(decision.get_stats()["llm_parse_retry_events"]), 2)
        for row in range(priorities.shape[0]):
            for col in range(priorities.shape[1]):
                expected = 0.1 if self.grid_map.is_traversable(row, col) else 0.0
                self.assertAlmostEqual(priorities[row, col], expected)

    def test_invalid_priority_value_retries_without_partial_result(self):
        responses = [
            '{"priorities": {"1,1": 0.9, "1,2": "I"}}',
            '{"priorities": {"1,1": 0.7}}',
        ]
        with patch("agent.decision._call_llm", side_effect=responses) as call:
            priorities, _ = decision._call_llm_and_parse_priorities(
                "system", "user", self.grid_map, "qwen")

        self.assertEqual(call.call_count, 2)
        self.assertAlmostEqual(priorities[1, 1], 0.7)
        self.assertAlmostEqual(priorities[1, 2], 0.0)
        self.assertEqual(decision.get_stats()["llm_fallback_count"], 0)
        self.assertEqual(
            len(decision.get_stats()["llm_parse_retry_events"]), 1)

    def test_max_token_failure_is_retried_only_once(self):
        with patch("agent.decision._call_llm",
                   side_effect=["truncated", "truncated"]) as call, \
                patch("agent.decision._classify_parse_failure",
                      return_value="max_tokens"):
            decision._call_llm_and_parse_priorities(
                "system", "user", self.grid_map, "deepseek")

        self.assertEqual(call.call_count, 2)
        events = decision.get_stats()["llm_parse_retry_events"]
        self.assertEqual([event["kind"] for event in events],
                         ["max_tokens", "max_tokens"])
        self.assertEqual(decision.get_stats()["llm_fallback_count"], 1)

    def test_parseable_max_token_response_is_still_retried(self):
        responses = [
            '{"priorities": {"1,1": 0.9}}',
            '{"priorities": {"1,2": 0.8}}',
        ]
        with patch("agent.decision._call_llm", side_effect=responses) as call, \
                patch("agent.decision._last_llm_response_hit_output_limit",
                      side_effect=[True, False]):
            priorities, _ = decision._call_llm_and_parse_priorities(
                "system", "user", self.grid_map, "deepseek")

        self.assertEqual(call.call_count, 2)
        self.assertAlmostEqual(priorities[1, 1], 0.0)
        self.assertAlmostEqual(priorities[1, 2], 0.8)
        self.assertEqual(
            decision.get_stats()["llm_parse_retry_events"][0]["kind"],
            "max_tokens")

    def test_nonfinite_priority_value_is_parse_failure(self):
        priorities, ok = decision._parse_priorities(
            '{"priorities": {"1,1": NaN}}', self.grid_map)

        self.assertFalse(ok)
        self.assertTrue(np.all(priorities == 0.0))

    def _uavs_and_state(self, energy=None):
        base = self.grid_map.cell_centre(1, 13)
        uavs = [UAV(1, base), UAV(0, base)]
        if energy is not None:
            for uav in uavs:
                uav.energy = energy
        state = PerceptionModule(self.grid_map).compute_state(uavs, {}, 3600.0)
        state["occupied_cells"] = {(1, 13)}
        return uavs, state

    def test_direct_assignments_use_nearest_fallback_after_one_retry(self):
        uavs, state = self._uavs_and_state()
        expected = decision._nearest_direct_fallback(
            self.grid_map, state, uavs)
        with patch("agent.decision._build_llm_direct_prompt",
                   return_value=("system", "user")), \
                patch("agent.decision._call_llm",
                      side_effect=["not json"] * 2) as call:
            assignments = decision.make_decision_llm_direct(
                self.grid_map, None, state, uavs, 5, provider="deepseek")

        self.assertEqual(call.call_count, 2)
        self.assertEqual(assignments, expected)
        self.assertEqual(list(assignments), [0, 1])
        self.assertEqual(len(set(assignments.values())), len(assignments))
        self.assertEqual(decision.get_stats()["llm_fallback_count"], 1)

    def test_direct_zero_usable_assignments_are_retried(self):
        uavs, state = self._uavs_and_state()
        invalid = '{"assignments": {"0": "99,99"}}'
        valid = '{"assignments": {"0": "1,12"}}'
        with patch("agent.decision._build_llm_direct_prompt",
                   return_value=("system", "user")), \
                patch("agent.decision._call_llm",
                      side_effect=[invalid, valid]) as call:
            assignments = decision.make_decision_llm_direct(
                self.grid_map, None, state, uavs, 5, provider="deepseek")

        self.assertEqual(call.call_count, 2)
        self.assertEqual(assignments, {0: (1, 12)})
        self.assertEqual(decision.get_stats()["llm_fallback_count"], 0)

    def test_direct_fallback_leaves_energy_infeasible_uavs_unassigned(self):
        uavs, state = self._uavs_and_state(energy=0.0)
        with patch("agent.decision._build_llm_direct_prompt",
                   return_value=("system", "user")), \
                patch("agent.decision._call_llm",
                      side_effect=["not json"] * 2):
            assignments = decision.make_decision_llm_direct(
                self.grid_map, None, state, uavs, 5, provider="deepseek")

        self.assertEqual(assignments, {})
        self.assertEqual(decision.get_stats()["llm_fallback_count"], 1)


if __name__ == "__main__":
    unittest.main()
