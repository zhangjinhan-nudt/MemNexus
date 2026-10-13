import asyncio
import json

import httpx
import pytest

from agent_memory.providers import BudgetExceededError, OpenAICompatibleModel


@pytest.mark.parametrize("second_reason", ["stop", "length"])
def test_one_length_recovery_counts_both_attempts(monkeypatch, second_reason):
    monkeypatch.setenv("EXPERIMENT_TRUNCATION_MAX_TOKENS", "8192")
    limits = []

    def handler(request):
        limits.append(json.loads(request.content)["max_tokens"])
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "valid"},
                         "finish_reason": "length" if len(limits) == 1 else second_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAICompatibleModel(base_url="https://example.invalid", model="test", client=client)
            if second_reason == "length":
                with pytest.raises(BudgetExceededError, match="truncated"):
                    await model.answer("test")
            else:
                assert await model.answer("test") == "valid"
            assert model.usage["provider_attempts"] == 2
            assert model.usage["length_recovery_attempts"] == 1
            assert model.usage["input_tokens"] == 20
            assert model.usage["output_tokens"] == 10
            assert model.usage["reserved_rmb"] == 0
    asyncio.run(run())
    assert limits == [4096, 8192]
