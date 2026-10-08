import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from samplers.openai_sampler import OpenAISampler
from samplers.zai_sampler import ZaiSampler


class SamplerParameterTests(unittest.TestCase):
    def make_sampler(self, sampler_type, stream=False):
        sampler = sampler_type(api_key="test-key", temperature=0.7, top_p=0.9,
                               stream=stream)
        create = Mock()
        sampler.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        if stream:
            create.return_value = iter([SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="answer", reasoning_content=None))])])
        else:
            create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="answer"))])
        return sampler, create

    def test_call_parameters_reach_both_clients(self):
        for sampler_type in (OpenAISampler, ZaiSampler):
            for stream in (False, True):
                for overrides, expected in (
                    ({}, (0.7, 0.9)),
                    ({"temperature": 0.2, "top_p": 0.4}, (0.2, 0.4)),
                    ({"temperature": 0.0}, (0.0, 0.9)),
                ):
                    with self.subTest(sampler=sampler_type.__name__, stream=stream,
                                      overrides=overrides):
                        sampler, create = self.make_sampler(sampler_type, stream)
                        self.assertEqual(sampler([{"role": "user", "content": "hi"}],
                                                 **overrides), "answer")
                        self.assertEqual(create.call_args.kwargs["temperature"],
                                         expected[0])
                        self.assertEqual(create.call_args.kwargs["top_p"], expected[1])


if __name__ == "__main__":
    unittest.main()
