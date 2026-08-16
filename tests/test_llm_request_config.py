"""Offline request-shape tests for the formal LLM configurations."""

import sys
import types
import unittest
from unittest import mock

import agent.decision as decision


class _Completions:
    def __init__(self, captured):
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        message = types.SimpleNamespace(content='{"priorities": []}')
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        return types.SimpleNamespace(choices=[choice], usage=usage)


class _Client:
    captured = None

    def __init__(self, **kwargs):
        self.chat = types.SimpleNamespace(
            completions=_Completions(self.captured))


def _fake_openai_module(captured):
    _Client.captured = captured
    module = types.ModuleType("openai")
    module.OpenAI = _Client
    return module


class LLMRequestConfigTests(unittest.TestCase):
    def setUp(self):
        decision.reset_stats()

    def test_formal_models_share_one_output_budget(self):
        configs = [
            ("deepseek", "deepseek-v4-flash", False),
            ("deepseek", "deepseek-v4-pro", False),
            ("qwen", "qwen-flash", False),
            ("qwen", "qwen-plus-2025-12-01", False),
            ("qwen", "qwen3-max-2026-01-23", False),
        ]
        for provider, model, thinking in configs:
            with self.subTest(model=model):
                self.assertEqual(
                    decision._max_tokens_for(provider, model, thinking),
                    16384,
                )

    def test_deepseek_non_thinking_explicitly_disables_thinking(self):
        captured = {}
        with mock.patch.dict(sys.modules, {
                "openai": _fake_openai_module(captured)}), \
                mock.patch.object(decision, "_use_cache", False), \
                mock.patch.object(decision, "LLM_MODEL_DEEPSEEK",
                                  "deepseek-v4-pro"), \
                mock.patch.object(decision, "LLM_THINKING_DEEPSEEK", False):
            decision._call_llm("system", "user", provider="deepseek",
                               max_retries=1)

        self.assertEqual(captured["model"], "deepseek-v4-pro")
        self.assertEqual(captured["max_tokens"], 16384)
        self.assertEqual(captured["extra_body"], {
            "thinking": {"type": "disabled"}
        })
        self.assertNotIn("reasoning_effort", captured)

    def test_qwen_non_thinking_request_is_unchanged(self):
        captured = {}
        with mock.patch.dict(sys.modules, {
                "openai": _fake_openai_module(captured)}), \
                mock.patch.object(decision, "_use_cache", False), \
                mock.patch.object(decision, "LLM_MODEL_QWEN",
                                  "qwen-plus-2025-12-01"), \
                mock.patch.object(decision, "LLM_THINKING_QWEN", False):
            decision._call_llm("system", "user", provider="qwen",
                               max_retries=1)

        self.assertEqual(captured["model"], "qwen-plus-2025-12-01")
        self.assertEqual(captured["max_tokens"], 16384)
        self.assertEqual(captured["extra_body"], {
            "enable_thinking": False
        })


if __name__ == "__main__":
    unittest.main()
