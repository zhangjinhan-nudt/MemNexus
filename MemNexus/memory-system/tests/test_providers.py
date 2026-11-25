"""Provider contracts exercise mocks only; tests never need keys or a model server."""

import asyncio
import json
import os
import threading
import unittest
from unittest.mock import patch

import httpx

from agent_memory.models import Event, ExtractionWindow, Session, Turn
from agent_memory.ports import ModelNotConfigured, ModelRequestError
from agent_memory.providers import (
    CallableModel,
    ModelOutputError,
    OpenAICompatibleModel,
    UnconfiguredModel,
    build_model,
)
from agent_memory.settings import Settings


def sample_window() -> ExtractionWindow:
    return ExtractionWindow(
        session=Session(session_id="session1"),
        new_turns=[
            Turn(
                turn_id="turn1",
                session_id="session1",
                sequence=1,
                request_id="request1",
                events=[Event(role="user", content="这次请用中文回答。")],
            )
        ],
        context_turns=[],
    )


def sample_extraction() -> dict:
    return {
        "candidates": [
            {
                "content": "当前任务使用中文回答",
                "subject": "当前任务",
                "predicate": "回答语言",
                "value": "中文",
                "scope_type": "session",
                "durable": False,
                "evidence": [{"turn_id": "turn1", "event_index": 0, "quote": "这次请用中文回答。"}],
            }
        ],
        "summary": "当前任务使用中文",
        "state_patch": {},
    }


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_repaired_extraction_keeps_whole_valid_candidates_under_both_policies(self):
        data = sample_extraction()
        valid = data["candidates"][0]
        invalid = {**valid, "predicate": "x" * 101}
        data["candidates"] = [invalid, valid]
        for flag in ("0", "1"):
            with self.subTest(flag=flag), patch.dict(os.environ, {"MEMORY_EXPERIMENT_DROP_QUESTION_CANDIDATES": flag}):
                prompts = []

                async def chat(prompt):
                    prompts.append(prompt)
                    return json.dumps(data)

                model = CallableModel(chat)
                result = await model.extract(sample_window())
                self.assertEqual(len(prompts), 2)
                self.assertEqual(result.summary, data["summary"])
                self.assertEqual(len(result.candidates), 1)
                [retained] = result.candidates
                self.assertEqual(retained.predicate, valid["predicate"])
                self.assertEqual(retained.evidence[0].model_dump(exclude_none=True), valid["evidence"][0])
                self.assertEqual(model.validation_rejections["Invalid candidate schema after one complete repair"], 1)

    async def test_repaired_extraction_requires_complete_structure_and_authorized_scope(self):
        data = sample_extraction()
        valid = data["candidates"][0]
        invalid = {**valid, "predicate": "x" * 101}
        examples = {
            "invalid_json": '{"candidates":[',
            "top_level_error": json.dumps({"summary": [], "candidates": [valid, invalid]}),
            "no_valid_candidate": json.dumps({"candidates": [invalid]}),
            "scope_violation": json.dumps({"candidates": [{**valid, "scope_type": "project"}, invalid]}),
        }
        for flag in ("0", "1"):
            for case, output in examples.items():
                with self.subTest(flag=flag, case=case), patch.dict(os.environ, {"MEMORY_EXPERIMENT_DROP_QUESTION_CANDIDATES": flag}):
                    prompts = []

                    async def chat(prompt):
                        prompts.append(prompt)
                        return output

                    with self.assertRaises(ModelOutputError):
                        await CallableModel(chat).extract(sample_window())
                    self.assertEqual(len(prompts), 2)

    async def test_candidate_schema_recovery_keeps_existing_quote_policy(self):
        valid = sample_extraction()["candidates"][0]
        invalid_schema = {**valid, "predicate": "x" * 101}
        false_quote = {**valid, "evidence": [{**valid["evidence"][0], "quote": "伪造的引文"}]}
        output = json.dumps({"candidates": [invalid_schema, false_quote, valid]})
        for flag in ("0", "1"):
            with self.subTest(flag=flag), patch.dict(os.environ, {"MEMORY_EXPERIMENT_DROP_QUESTION_CANDIDATES": flag}):
                model = CallableModel(lambda _: output)
                if flag == "0":
                    with self.assertRaisesRegex(ModelOutputError, "exact substring"):
                        await model.extract(sample_window())
                else:
                    result = await model.extract(sample_window())
                    self.assertEqual([c.evidence[0].quote for c in result.candidates], [valid["evidence"][0]["quote"]])

    async def test_extraction_output_limit_repairs_from_original_input_once(self):
        calls = []
        partial = "partial output must not be replayed"

        def reply(request):
            body = json.loads(request.content)
            calls.append(body)
            first = len(calls) == 1
            payload = {
                "choices": [{"finish_reason": "length" if first else "stop",
                             "message": {"content": partial if first else json.dumps(sample_extraction())}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50},
            }
            return httpx.Response(200, json=payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            model = OpenAICompatibleModel(base_url="https://test.invalid/v1", model="test", client=client)
            result = await model.extract(sample_window())
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["max_tokens"], calls[1]["max_tokens"])
        repair = calls[1]["messages"][0]["content"]
        self.assertTrue(repair.startswith(calls[0]["messages"][0]["content"]))
        self.assertNotIn(partial, repair)

    async def test_unconfigured_has_no_factual_fallback(self):
        model = UnconfiguredModel()
        for operation in (
            lambda: model.extract(sample_window()),
            lambda: model.plan("query", {}),
            lambda: model.answer("prompt"),
        ):
            with self.assertRaises(ModelNotConfigured):
                await operation()

    async def test_valid_extraction_and_fenced_json(self):
        prompts = []

        async def chat(prompt):
            prompts.append(prompt)
            return "```json\n" + json.dumps(sample_extraction(), ensure_ascii=False) + "\n```"

        result = await CallableModel(chat).extract(sample_window())
        self.assertEqual(result.candidates[0].value, "中文")
        self.assertEqual(len(prompts), 1)
        self.assertIn("Event.occurred_at", prompts[0])
        self.assertIn("context_turns", prompts[0])

    async def test_format_repair_is_bounded_and_strict(self):
        calls = []

        async def chat(prompt):
            calls.append(prompt)
            return '{"route":"short","depth":"3"}' if len(calls) == 1 else '{"route":"short","depth":3}'

        result = await CallableModel(chat).plan("刚才要求什么", {})
        self.assertEqual(result.depth, 3)
        self.assertEqual(len(calls), 2)
        self.assertIn("previous_response", calls[1])

    async def test_invalid_format_stops_after_two_calls(self):
        calls = 0

        async def chat(prompt):
            nonlocal calls
            calls += 1
            return '{"route":"short","owner_id":"other-owner"}'

        with self.assertRaises(ModelOutputError):
            await CallableModel(chat).plan("query", {})
        self.assertEqual(calls, 2)

    async def test_schema_repair_keeps_original_deadline(self):
        calls = 0
        deadlines = []
        real_timeout = asyncio.timeout

        def deadline(delay):
            timer = real_timeout(delay)
            deadlines.append(timer)
            return timer

        async def chat(prompt):
            nonlocal calls
            calls += 1
            if calls == 2:
                await asyncio.sleep(1)
            return "bad JSON" if calls == 1 else '{"route":"short","depth":3}'

        with (
            patch("agent_memory.providers.asyncio.timeout", side_effect=deadline),
            self.assertRaises(TimeoutError),
        ):
            await CallableModel(chat, timeout=0.2).plan("query", {})
        self.assertEqual(calls, 2)
        self.assertEqual(len(deadlines), 1, "Schema repair must not restart the request timeout")

    async def test_network_or_host_error_is_not_repaired(self):
        calls = 0

        async def chat(prompt):
            nonlocal calls
            calls += 1
            raise ConnectionError("host unavailable")

        with self.assertRaises(ConnectionError):
            await CallableModel(chat).plan("query", {})
        self.assertEqual(calls, 1)

    async def test_sync_host_callable_runs_in_worker_thread(self):
        main_thread = threading.get_ident()
        seen_threads = []

        def chat(prompt):
            seen_threads.append(threading.get_ident())
            return "answer"

        self.assertEqual(await CallableModel(chat).answer("prompt"), "answer")
        self.assertNotEqual(seen_threads[0], main_thread)

    async def test_answer_has_no_json_repair(self):
        calls = 0

        async def chat(prompt):
            nonlocal calls
            calls += 1
            return "依据 [1]，当前使用中文。"

        self.assertEqual(await CallableModel(chat).answer("prompt"), "依据 [1]，当前使用中文。")
        self.assertEqual(calls, 1)

    async def test_new_only_evidence_cannot_reference_context(self):
        data = sample_extraction()
        data["candidates"][0]["evidence"][0]["turn_id"] = "old-turn"

        async def chat(prompt):
            return json.dumps(data)

        with self.assertRaisesRegex(ModelOutputError, "new turn"):
            await CallableModel(chat).extract(sample_window())

    async def test_evidence_must_be_exact_quote(self):
        data = sample_extraction()
        data["candidates"][0]["evidence"][0]["quote"] = "以后所有任务都用中文"

        async def chat(prompt):
            return json.dumps(data)

        with self.assertRaisesRegex(ModelOutputError, "exact substring"):
            await CallableModel(chat).extract(sample_window())

    async def test_assistant_alone_cannot_be_stated_fact_source(self):
        window = sample_window()
        window.new_turns[0].events[0].role = "assistant"

        async def chat(prompt):
            return json.dumps(sample_extraction())

        with self.assertRaisesRegex(ModelOutputError, "Assistant"):
            await CallableModel(chat).extract(window)

    async def test_assistant_plan_is_marked_and_kept_provisional(self):
        window = sample_window()
        window.new_turns[0].events[0].role = "assistant"
        data = sample_extraction()
        data["candidates"][0]["assertion"] = "planned"

        async def chat(prompt):
            return json.dumps(data)

        result = await CallableModel(chat).extract(window)
        self.assertEqual(result.candidates[0].assertion, "planned")
        self.assertFalse(result.candidates[0].durable)

    async def test_context_can_support_new_reference_resolution(self):
        window = sample_window()
        window.context_turns.append(
            Turn(
                turn_id="context1",
                session_id="session1",
                sequence=0,
                request_id="context-request",
                events=[Event(role="user", content="这个任务是项目 P 的汇报。")],
            )
        )
        data = sample_extraction()
        data["candidates"][0]["evidence"].append(
            {
                "turn_id": "context1",
                "event_index": 0,
                "quote": "项目 P 的汇报",
            }
        )

        async def chat(prompt):
            return json.dumps(data)

        result = await CallableModel(chat).extract(window)
        self.assertEqual(len(result.candidates[0].evidence), 2)

    async def test_planner_does_not_receive_whole_store_gold_or_identity(self):
        seen = []

        async def chat(prompt):
            seen.append(prompt)
            return '{"route":"none","depth":1}'

        await CallableModel(chat).plan(
            "1+1",
            {
                "summary": "当前任务",
                "owner_id": "SECRET_OWNER",
                "gold_answer": "SECRET_GOLD",
                "memories": ["SECRET_MEMORY"],
            },
        )
        self.assertIn("当前任务", seen[0])
        for marker in ("SECRET_OWNER", "SECRET_GOLD", "SECRET_MEMORY"):
            self.assertNotIn(marker, seen[0])

    async def test_mock_http_endpoint_preserves_v1_and_model_configuration(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAICompatibleModel(
                base_url="https://example.invalid/v1/",
                model="configured-model",
                api_key="mock-key",
                client=client,
            )
            self.assertEqual(await model.answer("prompt"), "hello")
            await model.aclose()
            self.assertFalse(client.is_closed, "Injected clients remain owned by the caller")
        self.assertEqual(str(requests[0].url), "https://example.invalid/v1/chat/completions")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer mock-key")
        self.assertEqual(json.loads(requests[0].content)["model"], "configured-model")
        self.assertEqual(json.loads(requests[0].content)["max_tokens"], 4096)

    async def test_mock_http_status_error_propagates_without_retry(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(503, text="unavailable")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAICompatibleModel(base_url="https://example.invalid", model="test", client=client)
            with self.assertRaisesRegex(ModelRequestError, "HTTP 503"):
                await model.plan("query", {})
        self.assertEqual(len(requests), 1)

    async def test_deepseek_answer_plan_and_extract_request_contract(self):
        requests = []
        replies = [
            '{"route":"short","semantic_queries":["报告语言"],"depth":1}',
            json.dumps(sample_extraction(), ensure_ascii=False),
            "这次使用中文回答。",
        ]

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": replies.pop(0)}}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAICompatibleModel(
                base_url="https://api.deepseek.com",
                model="deepseek-flash",
                api_key="mock-key",
                client=client,
            )
            self.assertEqual(model.provider, "deepseek")
            self.assertEqual((await model.plan("报告使用什么语言", {})).route, "short")
            self.assertEqual((await model.extract(sample_window())).candidates[0].value, "中文")
            self.assertEqual(await model.answer("请回答"), "这次使用中文回答。")

        self.assertEqual(len(requests), 3)
        for index, request in enumerate(requests):
            self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer mock-key")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "deepseek-flash")
            self.assertEqual(body["thinking"], {"type": "disabled"})
            self.assertEqual(body["temperature"], 0)
            if index < 2:
                self.assertEqual(body["response_format"], {"type": "json_object"})
            else:
                self.assertNotIn("response_format", body)

    async def test_empty_model_answer_is_rejected(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
            )
        ) as client:
            model = OpenAICompatibleModel(
                base_url="https://api.deepseek.com", model="deepseek-flash", client=client
            )
            with self.assertRaisesRegex(ModelOutputError, "empty content"):
                await model.answer("prompt")

    async def test_proxy_host_still_disables_reasoning(self):
        """A proxy gateway must not lose the directive just because of its hostname."""
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAICompatibleModel(
                base_url="https://www.tokenav.ai/v1",
                model="deepseek-v4.1-flash",
                api_key="mock-key",
                client=client,
            )
            self.assertEqual(model.provider, "openai_compatible")
            self.assertTrue(model.disable_thinking)
            await model.answer("prompt")
        self.assertEqual(json.loads(requests[0].content)["thinking"], {"type": "disabled"})

    async def test_reasoning_directive_can_be_switched_off(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

        with patch.dict(os.environ, {"MEMORY_MODEL_DISABLE_THINKING": "false"}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                model = OpenAICompatibleModel(
                    base_url="https://www.tokenav.ai/v1",
                    model="deepseek-v4.1-flash",
                    api_key="mock-key",
                    client=client,
                )
                self.assertFalse(model.disable_thinking)
                await model.answer("prompt")
        self.assertNotIn("thinking", json.loads(requests[0].content))

    async def test_rejected_reasoning_directive_is_dropped_once(self):
        """A gateway that 400s on unknown fields must not fail the whole run."""
        requests = []

        def handler(request):
            requests.append(request)
            if "thinking" in json.loads(request.content):
                return httpx.Response(400, json={"error": {"message": "unknown field thinking"}})
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAICompatibleModel(
                base_url="https://www.tokenav.ai/v1",
                model="deepseek-v4.1-flash",
                api_key="mock-key",
                client=client,
            )
            self.assertEqual(await model.answer("prompt"), "hello")
        self.assertEqual(len(requests), 2)
        self.assertIn("thinking", json.loads(requests[0].content))
        self.assertNotIn("thinking", json.loads(requests[1].content))

    async def test_empty_answer_reports_the_reasoning_trace(self):
        """An empty content caused by an unbounded trace must say so."""
        payload = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "thinking very hard"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4096},
        }
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ) as client:
            model = OpenAICompatibleModel(
                base_url="https://www.tokenav.ai/v1", model="deepseek-v4.1-flash", client=client
            )
            with self.assertRaisesRegex(ModelOutputError, "reasoning trace"):
                await model.answer("prompt")

    async def test_factory_unconfigured_partial_and_local_endpoint(self):
        empty = {"MEMORY_MODEL_BASE_URL": "", "MEMORY_MODEL_NAME": "", "MEMORY_MODEL_API_KEY": ""}
        with patch.dict(os.environ, empty):
            self.assertIsNone(build_model(Settings()))
        with (
            patch.dict(os.environ, empty | {"MEMORY_MODEL_API_KEY": "mock-key"}),
            self.assertRaises(ValueError),
        ):
            build_model(Settings())
        with patch.dict(
            os.environ,
            empty
            | {
                "MEMORY_MODEL_BASE_URL": "https://api.deepseek.com",
                "MEMORY_MODEL_NAME": "deepseek-flash",
            },
        ):
            self.assertIsNone(build_model(Settings()))
        with patch.dict(
            os.environ,
            empty
            | {
                "MEMORY_MODEL_BASE_URL": "http://localhost:1234/v1",
                "MEMORY_MODEL_NAME": "local",
            },
        ):
            model = build_model(Settings())
            self.assertIsInstance(model, OpenAICompatibleModel)
            await model.aclose()

    async def test_process_environment_overrides_dotenv_configuration(self):
        configured = {
            "MEMORY_MODEL_BASE_URL": "https://example.invalid/v4",
            "MEMORY_MODEL_NAME": "configured-model",
            "MEMORY_MODEL_API_KEY": "local-secret",
        }
        with patch.dict(os.environ, configured):
            model = build_model(Settings())
            self.assertEqual(model.endpoint, "https://example.invalid/v4/chat/completions")
            self.assertEqual(model.model, "configured-model")
            await model.aclose()


if __name__ == "__main__":
    unittest.main()
