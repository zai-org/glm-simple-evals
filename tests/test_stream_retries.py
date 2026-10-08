import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from samplers.openai_sampler import OpenAISampler
from samplers.zai_sampler import ZaiSampler


def chunk(reasoning=None, content=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        reasoning_content=reasoning, content=content))])


def interrupted_stream():
    yield chunk(reasoning="FAILED_ATTEMPT " * 100)
    yield chunk(content="partial answer")
    raise ConnectionError("interrupted stream")


class StreamRetryTests(unittest.TestCase):
    def make_sampler(self, sampler_type, create):
        sampler = sampler_type(api_key="test-key", stream=True)
        sampler.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        return sampler

    def test_successful_retry_discards_previous_reasoning(self):
        for sampler_type in (OpenAISampler, ZaiSampler):
            with self.subTest(sampler=sampler_type.__name__):
                reasoning = "successful reasoning " * 100
                create = Mock(side_effect=[interrupted_stream(),
                                           iter([chunk(reasoning=reasoning)])])
                sampler = self.make_sampler(sampler_type, create)
                with patch("time.sleep"), contextlib.redirect_stdout(io.StringIO()):
                    output = sampler([])
                self.assertEqual(output, reasoning[:-512].strip())
                self.assertEqual(create.call_count, 2)

    def test_exhausted_retries_do_not_return_failed_reasoning(self):
        for sampler_type, attempts in ((OpenAISampler, 50), (ZaiSampler, 200)):
            with self.subTest(sampler=sampler_type.__name__):
                create = Mock(side_effect=lambda **kwargs: interrupted_stream())
                sampler = self.make_sampler(sampler_type, create)
                with patch("time.sleep"), contextlib.redirect_stdout(io.StringIO()):
                    output = sampler([])
                self.assertEqual(output, "")
                self.assertEqual(create.call_count, attempts)


if __name__ == "__main__":
    unittest.main()
