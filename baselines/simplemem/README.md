# Frozen SimpleMem adapter

This directory preserves the SimpleMem baseline used by the MemNexus benchmark without pushing local changes to the upstream repository.

- Upstream: <https://github.com/aiming-lab/SimpleMem>
- Pinned commit: `db80b6a7c591e0ea730a058e9f5fc4eb06572299`
- Adapter: `adapter.patch` (9 changed files; generated databases excluded)
- Verification: `manifest.json` records normalized hashes for 464 tracked files.
- Local adapter validation: `validation.json`; no paid API calls were made.

## Restore after cloning MemNexus

From the repository root:

```shell
git submodule update --init -- SimpleMem
python baselines/simplemem/restore.py --apply
python baselines/simplemem/restore.py --check
```

The script checks the upstream commit and patch hash, refuses to overwrite an incompatible working tree, and verifies every recorded file after applying. Running it on an already restored tree only verifies the files. It does not install dependencies or invoke a model.

## What the adapter changes

- Injects the shared embedding implementation into the baseline.
- Separates embedding query encoding from document encoding.
- Records model usage, failed attempts, and extraction-window failures.
- Provides benchmark configuration defaults and regression tests for accounting and embedding roles.

The memory-extraction, retrieval-planning, and answer-generation prompt text remains the pinned upstream text. The patch does not introduce MemNexus reasoning or evidence-selection logic into SimpleMem.

Most apparent local modifications were CRLF line-ending changes. Hash verification normalizes CRLF to LF for UTF-8 files. The package also blanks `OPENAI_API_KEY` and `JUDGE_API_KEY` in example configurations. Real local `config.py`, API keys, databases, logs, and generated `tests/test_lancedb` data are not packaged. Hash equivalence is consequently source equivalence under this explicitly documented normalization, not raw-byte identity of private local configuration.

## Experiment configuration

`config.paper.py.example` is a preserved historical example, not the current experiment protocol. Its model name and token ceiling are historical. The recent paired run used `deepseek-flash`, temperature `0`, maximum output tokens `32768`, thinking disabled, shared Qwen3 embeddings, window/overlap `20/15`, planning enabled, and up to two SimpleMem reflection rounds. Its judge used the SimpleMem Appendix A.4 prompt with an explicitly declared DeepSeek substitute.

For a new run, record and pass the selected model, decoding limits, embedding settings, dataset identity, judge identity, pricing, and budget through the benchmark runner. Do not silently fall back to this historical example or infer a spending authorization from it. Keep credentials in ignored local configuration or environment variables. This package does not start or resume experiments.
