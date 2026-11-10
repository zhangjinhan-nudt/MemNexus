"""Prefix-cache invariant: variable payloads must come after every fixed part.

A provider only bills a cached prefix when the request begins with bytes it has
seen before. Anything that changes per call — conversation turns, the query, the
planning context with its query_time timestamp — therefore has to sit at the very
end, after the instructions, the output-language directive and the JSON Schema.

Before this invariant was enforced, ``extract`` and ``plan`` inlined their data
into the prompt and the schema was appended afterwards, so the cacheable prefix
stopped where the data began (24% for extract, 18% for plan).
"""

import json
import unittest

import httpx

from agent_memory.providers import OpenAICompatibleModel

from test_providers import sample_extraction, sample_window


class PromptPrefixCacheTests(unittest.IsolatedAsyncioTestCase):
    def capture(self, *replies: str):
        seen: list[str] = []

        def handler(request):
            seen.append(json.loads(request.content)["messages"][0]["content"])
            reply = replies[0] if len(replies) == 1 else replies[len(seen) - 1]
            return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

        return seen, handler

    async def build(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        return OpenAICompatibleModel(
            base_url="https://api.deepseek.com", model="m", api_key="k", client=client
        )

    async def test_plan_puts_schema_before_the_query_payload(self):
        seen, handler = self.capture('{"route":"short","semantic_queries":["q"],"depth":3}')
        model = await self.build(handler)
        await model.plan("who adopted the beagle", {"query_time": "2026-10-20T20:00:00+00:00"})

        prompt = seen[0]
        self.assertIn("JSON Schema", prompt)
        self.assertIn("输入数据：", prompt)
        # 固定部分（含 Schema）在前，变化部分在最后。
        self.assertLess(prompt.index("JSON Schema"), prompt.index("输入数据："))
        # 查询本身属于变化部分，必须落在数据块里。
        self.assertLess(prompt.index("输入数据："), prompt.index("who adopted the beagle"))

    async def test_extract_puts_schema_before_the_turn_payload(self):
        seen, handler = self.capture(json.dumps(sample_extraction(), ensure_ascii=False))
        model = await self.build(handler)
        await model.extract(sample_window())

        prompt = seen[0]
        self.assertIn("JSON Schema", prompt)
        self.assertIn("输入数据：", prompt)
        self.assertLess(prompt.index("JSON Schema"), prompt.index("输入数据："))

    async def test_consecutive_extractions_share_a_prefix_past_the_schema(self):
        """Two different windows must agree on a prefix that already covers the schema."""
        reply = json.dumps(sample_extraction(), ensure_ascii=False)
        prompts: list[str] = []

        def handler(request):
            # 重复返回同一个回复：万一触发 repair 也不会耗尽队列，
            # repair 只在初始 prompt 之后追加，不影响公共前缀。
            prompts.append(json.loads(request.content)["messages"][0]["content"])
            return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

        model = await self.build(handler)
        await model.extract(sample_window())
        second_window = sample_window()
        second_window.session.session_id = "session2"
        await model.extract(second_window)

        first, second = prompts[0], prompts[1]
        schema_at = first.index("JSON Schema")
        common = 0
        while common < min(len(first), len(second)) and first[common] == second[common]:
            common += 1
        # 公共前缀必须越过 Schema 起点，证明固定段整体可缓存。
        self.assertGreater(common, schema_at, f"shared prefix {common} did not reach schema at {schema_at}")
