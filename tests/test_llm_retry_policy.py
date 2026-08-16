import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

import agent.decision as decision


def _response(content="{\"priorities\": {\"1,1\": 1.0}}",
              finish_reason="stop"):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20)
    return SimpleNamespace(choices=[choice], usage=usage)


class LLMRetryPolicyTests(unittest.TestCase):
    def setUp(self):
        decision.reset_stats()
        decision.set_cache_enabled(False)

    def test_default_transient_attempt_limit_is_300(self):
        default = inspect.signature(decision._call_llm).parameters[
            "max_retries"].default
        self.assertEqual(default, 300)

    def test_timeout_is_retried_until_success(self):
        create = Mock(side_effect=[
            httpx.ReadTimeout("temporary timeout"),
            _response(),
        ])
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)))

        with patch("openai.OpenAI", return_value=fake_client), \
                patch("agent.decision.time.sleep") as sleep:
            text = decision._call_llm("system", "user", max_retries=300)

        self.assertIn("priorities", text)
        self.assertEqual(create.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_timeout_exhausts_exact_requested_attempt_count(self):
        create = Mock(side_effect=httpx.ReadTimeout("persistent timeout"))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)))

        with patch("openai.OpenAI", return_value=fake_client), \
                patch("agent.decision.time.sleep"), \
                patch("builtins.print"):
            with self.assertRaises(httpx.ReadTimeout):
                decision._call_llm("system", "user", max_retries=300)

        self.assertEqual(create.call_count, 300)

    def test_non_transient_error_is_not_retried(self):
        create = Mock(side_effect=ValueError("invalid request"))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)))

        with patch("openai.OpenAI", return_value=fake_client), \
                patch("agent.decision.time.sleep"):
            with self.assertRaises(ValueError):
                decision._call_llm("system", "user", max_retries=300)

        self.assertEqual(create.call_count, 1)

    def test_length_response_is_returned_to_parse_retry_layer(self):
        create = Mock(return_value=_response(
            content='{\"priorities\": {\"1,1\":',
            finish_reason="length"))
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)))

        with patch("openai.OpenAI", return_value=fake_client):
            text = decision._call_llm("system", "user", max_retries=300)

        self.assertEqual(text, '{\"priorities\": {\"1,1\":')
        self.assertEqual(create.call_count, 1)


if __name__ == "__main__":
    unittest.main()
