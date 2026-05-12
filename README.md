# Korean Dialogue KG Reasoning

Rebuilt, reproducible, and security-safe version of the original Korean dialogue KG-aware reasoning prototype.

The original repository had a strong idea but weak reproducibility:

- exposed API keys in source files
- hard-coded local paths
- no recoverable dataset bootstrap after artifacts were lost
- isolated scripts instead of one runnable pipeline

This rebuild keeps the core idea alive:

1. extract dialogue events
2. expand them into ATOMIC-style commonsense relations
3. build a Korean event knowledge graph
4. learn a lightweight relational encoder
5. retrieve relevant graph context for KG-aware reasoning

## What Changed

- Removed all hard-coded secrets and machine-specific paths.
- Replaced one-off scripts with a single Python package and CLI.
- Added curated bootstrap dialogue data so the project still runs even after losing the original artifacts.
- Added rule-based fallbacks so the full demo works without external APIs.
- Added optional OpenAI-compatible hooks for stronger event and relation generation.
- Added tests and a clean end-to-end demo command.

## How This Maps To The Previous Idea

The previous workflow in your slide can be mapped like this:

- `Kiwi + Bllossom` style event extraction:
  the rebuild uses a clean event extraction interface with a dependency-free rule baseline and an optional LLM backend. If you later install Kiwi or connect a local OpenAI-compatible model server, the interface is already ready for that upgrade.
- `KLUE/roberta-large -> 원인 / 관계 추출`:
  the rebuild exposes a separate relation extraction stage. Right now it ships with curated seed triples plus a rule-based fallback, and it can be swapped for a classifier or LLM extractor later without changing the rest of the pipeline.
- `KG + reasoning`:
  graph building, relational encoding, retrieval, and KG-grounded answer generation are now stitched together into one reproducible flow.

## Project Layout

```text
data/
  bootstrap/demo_dialogues.jsonl
kg_reasoning/
  bootstrap.py
  cli.py
  event_extraction.py
  graph.py
  heuristics.py
  inference.py
  io.py
  llm.py
  relation_extraction.py
  retrieval.py
  schema.py
  text.py
  training.py
tests/
  test_pipeline.py
```

## Requirements

- Python 3.11+
- no external dependency is required for the offline demo

If you want stronger extraction/generation later, configure an OpenAI-compatible endpoint:

```powershell
$env:KG_REASONING_API_BASE="http://localhost:8000"
$env:KG_REASONING_CHAT_MODEL="your-model-name"
$env:KG_REASONING_API_KEY="optional"
```

The client also understands `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, and `OPENAI_MODEL`.

## Quick Start

Run the full rebuilt demo:

```powershell
python -m kg_reasoning run-demo --query "배가 너무 아파서 조퇴하고 싶어."
```

Or, in this Codex workspace, use the bundled Python runtime:

```powershell
& "C:\Users\jaesa\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m kg_reasoning run-demo --query "배가 너무 아파서 조퇴하고 싶어."
```

## CLI Commands

Prepare curated demo assets:

```powershell
python -m kg_reasoning prepare-demo --output-dir artifacts/demo
```

Extract events from dialogues:

```powershell
python -m kg_reasoning extract-events --dialogues artifacts/demo/dialogues.jsonl --output artifacts/demo/events.jsonl --provider hybrid
```

Expand events into ATOMIC-style triples:

```powershell
python -m kg_reasoning extract-relations --dialogues artifacts/demo/dialogues.jsonl --events artifacts/demo/events.jsonl --output artifacts/demo/triples.jsonl --provider hybrid
```

Build a graph bundle:

```powershell
python -m kg_reasoning build-graph --triples artifacts/demo/triples.jsonl --output artifacts/demo/graph.json
```

Train the lightweight relational encoder:

```powershell
python -m kg_reasoning train-encoder --graph artifacts/demo/graph.json --output artifacts/demo/encoder.json
```

Compare baseline vs KG-aware reasoning:

```powershell
python -m kg_reasoning infer --graph artifacts/demo/graph.json --encoder artifacts/demo/encoder.json --triples artifacts/demo/triples.jsonl --query "시험 망친 것 같아." --top-k 5
```

## Notes On The Rebuilt Encoder

The original repo mentioned R-GCN, but the old public code was not packaged in a way that could be reproduced safely in this workspace.

This rebuild therefore ships with a dependency-light relational encoder:

- text hashing for node initialization
- relation-aware link training
- lightweight message passing for contextualization

That keeps the graph-aware reasoning idea functional right now. If you want, we can later add:

- a full PyTorch backend
- Kiwi-based anchor extraction
- a KLUE or sentence-transformer retrieval backend
- a local Bllossom or Qwen event extraction backend

## Security

The old repository exposed API keys directly in source code. Those keys should be treated as compromised and rotated immediately in any upstream environment, even though they are no longer present in this rebuilt working tree.

## Test

```powershell
python -m unittest discover -s tests -v
```
