"""Retrieval must not use a process hash seed as an implicit tie breaker."""

import json
import os
import subprocess
import sys
from pathlib import Path

import agent_memory

PROBE = """
import json, random, string, sys
sys.path.insert(0, sys.argv[1])
from agent_memory.retrieval import Retriever
rng = random.Random(20261117)
words = ['term' + a for a in string.ascii_lowercase[:20]]
bag = rng.choices(words, k=60)
docs = []
for i in range(12):
    rng.shuffle(bag)
    docs.append(' '.join(bag))
docs += [' '.join(rng.choices(words, k=rng.randrange(4, 100))) for _ in range(15)]
scores = Retriever._bm25(set(words), docs)
print(json.dumps(scores))
"""


def test_bm25_equal_bags_and_process_hash_seeds_have_identical_scores():
    """Bag-of-words evidence cannot change rank due to word insertion order."""
    package_root = str(Path(agent_memory.__file__).resolve().parent.parent)
    results = []
    for seed in (0, 17, 101):
        completed = subprocess.run(
            [sys.executable, "-c", PROBE, package_root],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            check=True, capture_output=True, text=True,
        )
        scores = json.loads(completed.stdout)
        assert all(score > 0 for score in scores)
        assert len(set(scores[:12])) == 1
        results.append(scores)
    assert results[0] == results[1] == results[2]
