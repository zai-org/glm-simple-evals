import contextlib
import io
import json
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from samplers.openai_sampler import OpenAISampler


def completion_chunk(delta=None, *, usage=False):
    return {
        "id": "completion-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": (
            [] if usage else [{"index": 0, "delta": delta, "finish_reason": None}]
        ),
        **(
            {"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}
            if usage
            else {}
        ),
    }


class OpenAIStreamTests(unittest.TestCase):
    def sample(self, chunks):
        requests = []
        events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        events += "data: [DONE]\n\n"

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=events
            )

        sampler = OpenAISampler(api_key="test-key", stream=True)
        sampler.client.close()
        with OpenAI(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            sampler.client = client
            with patch("samplers.openai_sampler.time.sleep") as sleep:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = sampler([{"role": "user", "content": "test"}])
            self.assertEqual(len(requests), 1)
            self.assertTrue(requests[0]["stream"])
            sleep.assert_not_called()
        return result

    def test_standard_content_stream_without_reasoning_extension(self):
        result = self.sample(
            [
                completion_chunk({"role": "assistant", "content": ""}),
                completion_chunk({"content": "Hello"}),
                completion_chunk({"content": " world"}),
                completion_chunk({}),
            ]
        )
        self.assertEqual(result, "Hello world")

    def test_empty_choices_usage_chunk_preserves_answer(self):
        result = self.sample(
            [
                completion_chunk({"content": "answer", "reasoning_content": None}),
                completion_chunk(usage=True),
            ]
        )
        self.assertEqual(result, "answer")

    def test_provider_reasoning_extension_keeps_answer_precedence(self):
        result = self.sample(
            [
                completion_chunk({"reasoning_content": "reasoning"}),
                completion_chunk({"content": "answer", "reasoning_content": None}),
            ]
        )
        self.assertEqual(result, "answer")

    def test_reasoning_only_stream_keeps_existing_fallback(self):
        reasoning = "reasoning " * 100
        result = self.sample([completion_chunk({"reasoning_content": reasoning})])
        self.assertEqual(result, reasoning[:-512].strip())


if __name__ == "__main__":
    unittest.main()
