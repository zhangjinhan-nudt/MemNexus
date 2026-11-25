"""Answer candidates retain recall diversity before public top-k selection."""

import unittest
from datetime import timedelta

from agent_memory.models import Evidence, Hit, Memory, QueryPlan, Session, utcnow
from agent_memory.reasoning import indexed_memory_hits
from agent_memory.retrieval import Retriever, _Match, _RecallResult
from agent_memory.settings import Settings


def fact(identifier, **changes):
    fields = dict(
        memory_id=identifier, session_id="s", scope_type="user", scope_id="owner",
        tier="long", subject=identifier, predicate="detail", value="known",
        content=f"A recorded detail for {identifier}.",
        evidence=[Evidence(turn_id="t", event_index=0, quote="known")],
    )
    fields.update(changes)
    return Memory(**fields)


class RankedChannels(Retriever):
    """Local scores model independent channel scales; no embedding/model calls."""

    def __init__(self, lexical, semantic, **settings):
        super().__init__(Settings(planner_mode="rule", **settings), embedder=object())
        self.lexical, self.semantic = lexical, semantic

    def _lexical_recall(self, terms, eligible, limit, namespace, generation):
        return self._scores(eligible, self.lexical, "lexical")

    async def _semantic_recall(self, query, plan, eligible, namespace, generation):
        return self._scores(eligible, self.semantic, "semantic"), 0

    @staticmethod
    def _scores(eligible, scores, view):
        matches = [_Match(m, scores[m.memory_id], {view})
                   for m in eligible if scores.get(m.memory_id, 0) > 0]
        return _RecallResult(eligible, matches,
                             {(m.memory_id, m.version): scores.get(m.memory_id, 0)
                              for m in eligible})


class CandidateUnionTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_head_survives_stronger_lexical_scores_and_public_top_k(self):
        decoys = [fact(f"lex{i:02}") for i in range(20)]
        target = fact("semantic", content="Paddling on a river is a regular pastime.")
        retriever = RankedChannels(
            {m.memory_id: .99 - i * .001 for i, m in enumerate(decoys)},
            {target.memory_id: .7}, max_candidates=6,
        )
        self.addCleanup(retriever.close)
        result = await retriever.search("water hobby", Session(session_id="s"), [], decoys + [target], top_k=1)
        self.assertEqual([h.metadata["memory_id"] for h in result.results], ["lex00"])
        ids = [h.metadata["memory_id"] for h in result.candidates]
        self.assertEqual(ids[:3], ["lex00", "semantic", "lex01"])
        self.assertEqual(len(ids), 6)
        self.assertEqual([h.metadata["retrieval_rank"] for h in result.candidates], list(range(1, 7)))
        self.assertEqual(result.candidates[1].metadata["retrieval_channel_ranks"], {"long.semantic": 1})
        self.assertEqual(result.candidates[1].metadata["retrieval_rank_stage"],
                         "candidate_union_before_public_selection")
        self.assertNotIn("candidates", result.model_dump())
        self.assertNotIn('"candidates":', result.model_dump_json(exclude={"channels"}))

    async def test_candidate_union_does_not_depend_on_cross_channel_score_scale(self):
        items = [fact("first"), fact("second"), fact("semantic")]
        outputs = []
        for scale in (1, .01):
            retriever = RankedChannels({"first": .98 * scale, "second": .97 * scale},
                                      {"semantic": .7}, max_candidates=2)
            self.addCleanup(retriever.close)
            result = await retriever.search("hobby", Session(session_id="s"), [], items)
            outputs.append([h.metadata["memory_id"] for h in result.candidates])
        self.assertEqual(outputs, [["first", "semantic"], ["first", "semantic"]])

    async def test_short_and_long_ranks_interleave_before_selection(self):
        short = [fact(f"short{i}", tier="short") for i in range(3)]
        long = [fact(f"long{i}") for i in range(6)]
        retriever = RankedChannels({m.memory_id: .99 - i * .01 for i, m in enumerate(short + long)},
                                  {}, max_candidates=9)
        self.addCleanup(retriever.close)
        result = await retriever.search("hobby", Session(session_id="s"), short, long, top_k=1)
        self.assertEqual([h.metadata["memory_id"] for h in result.candidates[:6]],
                         ["short0", "long0", "short1", "long1", "short2", "long2"])
        self.assertEqual(result.candidates[1].metadata["retrieval_channel_ranks"], {"long.lexical": 1})

    async def test_internal_candidates_enforce_scope_lifecycle_and_validity(self):
        now = utcnow()
        items = [fact("valid"), fact("withdrawn", status="retracted"),
                 fact("pending", status="pending"), fact("archived", status="archived"),
                 fact("old", status="superseded"), fact("future", valid_from=now + timedelta(days=1)),
                 fact("expired", valid_to=now - timedelta(days=1)),
                 fact("private", scope_type="session", scope_id="other", session_id="other"),
                 fact("project", scope_type="project", scope_id="other")]
        retriever = RankedChannels({m.memory_id: .9 for m in items}, {m.memory_id: .8 for m in items})
        self.addCleanup(retriever.close)
        result = await retriever.search("hobby", Session(session_id="s"), [], items)
        self.assertEqual([h.metadata["memory_id"] for h in result.candidates], ["valid"])
        historical = await retriever.search("previous hobby", Session(session_id="s"), [], items)
        self.assertEqual({h.metadata["memory_id"] for h in historical.candidates},
                         {"valid", "expired", "future", "old"})

    async def test_local_override_is_preserved_without_age_ranking_internal_candidates(self):
        local = fact("local", tier="short", subject="user", predicate="drink", value="tea",
                     scope_type="session", scope_id="s")
        durable = fact("durable", subject="user", predicate="drink", value="coffee")
        retriever = RankedChannels({"local": .1, "durable": .99}, {})
        self.addCleanup(retriever.close)
        result = await retriever.search("drink", Session(session_id="s"), [local], [durable])
        self.assertEqual(result.candidates[0].metadata["memory_id"], "local")
        self.assertTrue(result.candidates[0].metadata["local_override"])
        self.assertFalse(result.candidates[1].metadata["local_override"])

    async def test_pending_conflict_follows_candidate_outside_public_top_k(self):
        items = [fact("top"), fact("candidate", subject="user", predicate="drink", value="coffee"),
                 fact("pending", subject="user", predicate="drink", value="tea", status="pending")]
        retriever = RankedChannels({"top": .99, "candidate": .5}, {})
        self.addCleanup(retriever.close)
        result = await retriever.search("hobby", Session(session_id="s"), [], items, top_k=1)
        candidate = next(h for h in result.candidates if h.metadata["memory_id"] == "candidate")
        self.assertEqual(candidate.metadata["conflict_pending"][0]["memory_id"], "pending")
        self.assertNotIn("pending", [h.metadata["memory_id"] for h in result.candidates])

    async def test_age_reranking_remains_public_without_reordering_candidate_ranks(self):
        items = [fact("old", kind="event", recorded_at=utcnow() - timedelta(days=3650)),
                 fact("new", kind="event")]
        retriever = RankedChannels({"old": .99, "new": .98}, {})
        self.addCleanup(retriever.close)
        result = await retriever.search("hobby", Session(session_id="s"), [], items, top_k=1)
        self.assertEqual([h.metadata["memory_id"] for h in result.results], ["new"])
        self.assertEqual([h.metadata["memory_id"] for h in result.candidates], ["old", "new"])
        self.assertTrue(result.candidates[0].metadata["age_decay_applied"])

    def test_duplicate_channels_keep_all_ranks_without_spending_another_slot(self):
        a, b, c = fact("a"), fact("b"), fact("c")
        lexical = [_Match(a, .99, {"lexical"}, channel_ranks={"long.lexical": 1}),
                   _Match(b, .98, {"lexical"}, channel_ranks={"long.lexical": 2})]
        semantic = [_Match(c, .8, {"semantic"}, channel_ranks={"long.semantic": 1}),
                    _Match(a, .7, {"semantic"}, channel_ranks={"long.semantic": 2})]
        merged = Retriever._interleave([lexical, semantic], 2)
        self.assertEqual([h.memory.memory_id for h in merged], ["a", "c"])
        self.assertEqual(merged[0].channel_ranks, {"long.lexical": 1, "long.semantic": 2})
        self.assertEqual(merged[0].views, {"lexical", "semantic"})

    def test_eighty_unselected_candidates_are_exclusions_not_expansion_seeds(self):
        target = fact("target", subject="assistant", predicate="project objectives",
                      content="The project objectives are mapping stars and measuring spectra.")
        noise = fact("noise", subject="assistant", predicate="furniture",
                     content="Orchard lantern velvet cushion tapestry cabinet marble timber parquet wicker.")
        existing = [Hit(content=f"{noise.content} Inventory item {index}.", score=.9,
                        source_file="[memory]", chunk_id=f"candidate{index}:1",
                        metadata={"memory_id": f"candidate{index}", "subject": "assistant"})
                    for index in range(80)]
        duplicate = fact("candidate0", predicate="project objectives",
                         content="The project objectives are already represented in the candidate pool.")
        plan = QueryPlan(query_entities=["stellar research project"])
        baseline = indexed_memory_hits("What are the project objectives?", plan, [target, noise], [], limit=1)
        candidates = indexed_memory_hits("What are the project objectives?", plan,
            [target, noise, duplicate], [], limit=1, existing_candidates=existing)
        self.assertEqual([h.metadata["memory_id"] for h in baseline], ["target"])
        self.assertEqual([h.metadata["memory_id"] for h in candidates], ["target"])
        self.assertEqual(candidates[0].metadata["retrieval_rank"], 1)
        self.assertEqual(candidates[0].metadata["retrieval_rank_stage"], "indexed_sweep_before_grounding")
        self.assertEqual(candidates[0].metadata["retrieval_channel_ranks"], {"indexed_sweep": 1})

    def test_bridge_only_candidate_survives_without_overwhelming_direct_evidence(self):
        phrase = "Orchard lantern velvet cushion tapestry cabinet marble timber parquet wicker"
        seed = Hit(content=phrase, score=.9, source_file="[memory:seed]", chunk_id="seed:1",
                   metadata={"memory_id": "seed", "subject": "Rowan"})
        bridge = fact("bridge", subject="Mira", predicate="discussion", content=phrase)
        direct = fact("direct", subject="assistant", predicate="cobalt status",
                      content="The cobalt status is confirmed.")
        hits = indexed_memory_hits("cobalt status", QueryPlan(), [bridge, direct], [seed], limit=2)
        self.assertEqual([h.metadata["memory_id"] for h in hits], ["direct", "bridge"])
        # A complete phrase bridge contributes 2 before the existing bounded
        # score conversion, irrespective of how many rare pairs it contains.
        self.assertAlmostEqual(hits[1].score, 2 / 12)

    def test_entity_expansion_preserves_semantic_head_and_atomic_support_under_cap(self):
        semantic = fact("semantic", content="A semantically relevant statement.")
        noise = [fact("noise1"), fact("noise2")]
        alias = fact("alias", subject="Rowan", predicate="nickname", value="bluebird")
        location = fact("location", subject="Rowan", predicate="location", value="R210",
                        content="Rowan location R210")
        pool = [_Match(semantic, .7, {"semantic"}, channel_ranks={"long.semantic": 1}),
                _Match(noise[0], .99, {"lexical"}), _Match(noise[1], .98, {"lexical"})]
        expanded = Retriever._expand_entities(
            "bluebird location", QueryPlan(), Session(session_id="s"),
            [semantic, *noise, alias, location], pool, 3, utcnow(),
        )
        ids = {h.memory.memory_id for h in expanded}
        self.assertEqual(ids, {"semantic", "alias", "location"})
        leaf = next(h for h in expanded if h.memory.memory_id == "location")
        self.assertEqual(leaf.support, ("alias",))
        self.assertTrue(set(leaf.support) <= ids)
        self.assertEqual(pool[0].views, {"semantic"})
        self.assertLessEqual(len(expanded), 3)

