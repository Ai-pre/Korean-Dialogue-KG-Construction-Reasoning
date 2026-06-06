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

## Original-Code Preservation

This branch now supports both:

- rebuilt mode: a dependency-light path that runs in a clean environment
- legacy-preserving mode: keeps the original prompt style and the original `event_sentence -> event_cause -> triples` flow as closely as possible

In practice that means the repository is no longer only a redesign. It also carries forward the original event extraction prompt, relation generation prompt, and QA-sample generation logic in a safer modular form.

## What Changed

- Removed all hard-coded secrets and machine-specific paths.
- Replaced one-off scripts with a single Python package and CLI.
- Added curated bootstrap dialogue data so the project still runs even after losing the original artifacts.
- Added rule-based fallbacks so the full demo works without external APIs.
- Added optional OpenAI-compatible hooks for stronger event and relation generation.
- Added Korean pragmatic KG layers for speech acts, stance, politeness, indirectness, emotion cues, and speaker relations.
- Added multi-hop KG-response training data generation from pragmatic reasoning paths.
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
  multihop.py
  pragmatic_extraction.py
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

Convert raw AIHub dialogue JSON into the project format:

```powershell
python -m kg_reasoning import-aihub --input data/raw/aihub --output artifacts/aihub/dialogues.jsonl
```

The importer is designed for AIHub-style dialogue JSON with fields such as:

- `data[]`
- `header.dialogueInfo.dialogueID`
- `header.dialogueInfo.single_topic`
- `header.participantsInfo[]`
- `body[].participantID`
- `body[].utterance`

By default it drops system placeholders like `#@시스템#사진#` and keeps only dialogues with at least 2 utterances.

Extract events from dialogues:

```powershell
python -m kg_reasoning extract-events --dialogues artifacts/demo/dialogues.jsonl --output artifacts/demo/events.jsonl --provider legacy-hybrid
```

Expand events into ATOMIC-style triples:

```powershell
python -m kg_reasoning extract-relations --dialogues artifacts/demo/dialogues.jsonl --events artifacts/demo/events.jsonl --output artifacts/demo/triples.jsonl --provider legacy-hybrid
```

Extract Korean pragmatic signals and speaker relations:

```powershell
python -m kg_reasoning extract-pragmatics --dialogues artifacts/demo/dialogues.jsonl --signals-output artifacts/demo/pragmatic_signals.jsonl --relations-output artifacts/demo/speaker_relations.jsonl
```

Build pragmatic multi-hop reasoning queries:

```powershell
python -m kg_reasoning build-multihop-queries --dialogues artifacts/demo/dialogues.jsonl --events artifacts/demo/events.jsonl --pragmatics artifacts/demo/pragmatic_signals.jsonl --speaker-relations artifacts/demo/speaker_relations.jsonl --hops-output artifacts/demo/reasoning_hops.jsonl --queries-output artifacts/demo/multihop_queries.jsonl
```

Convert those paths into KG-response chat training data:

```powershell
python -m kg_reasoning prepare-multihop-training-data --queries artifacts/demo/multihop_queries.jsonl --hops artifacts/demo/reasoning_hops.jsonl --output artifacts/demo/multihop_kg_response_train.jsonl
```

Build a graph bundle:

```powershell
python -m kg_reasoning build-graph --triples artifacts/demo/triples.jsonl --output artifacts/demo/graph.json
```

Train the lightweight relational encoder:

```powershell
python -m kg_reasoning train-encoder --graph artifacts/demo/graph.json --output artifacts/demo/encoder.json
```

Train with the original-style R-GCN backend if `torch` and `torch_geometric` are installed:

```powershell
python -m kg_reasoning train-encoder --graph artifacts/demo/graph.json --output artifacts/demo/encoder.json --backend legacy-rgcn --hidden-dim 128 --epochs 50
```

Compare baseline vs KG-aware reasoning:

```powershell
python -m kg_reasoning infer --graph artifacts/demo/graph.json --encoder artifacts/demo/encoder.json --triples artifacts/demo/triples.jsonl --query "시험 망친 것 같아." --top-k 5
```

## AIHub Workflow

If you already have AIHub's `주제별 텍스트 일상 대화 데이터`, the practical flow is:

```powershell
python -m kg_reasoning import-aihub --input data/raw/aihub --output artifacts/aihub/dialogues.jsonl
python -m kg_reasoning extract-events --dialogues artifacts/aihub/dialogues.jsonl --output artifacts/aihub/events.jsonl --provider legacy-hybrid
python -m kg_reasoning extract-relations --dialogues artifacts/aihub/dialogues.jsonl --events artifacts/aihub/events.jsonl --output artifacts/aihub/triples.jsonl --provider legacy-hybrid
python -m kg_reasoning extract-pragmatics --dialogues artifacts/aihub/dialogues.jsonl --signals-output artifacts/aihub/pragmatic_signals.jsonl --relations-output artifacts/aihub/speaker_relations.jsonl
python -m kg_reasoning build-multihop-queries --dialogues artifacts/aihub/dialogues.jsonl --events artifacts/aihub/events.jsonl --pragmatics artifacts/aihub/pragmatic_signals.jsonl --speaker-relations artifacts/aihub/speaker_relations.jsonl --hops-output artifacts/aihub/reasoning_hops.jsonl --queries-output artifacts/aihub/multihop_queries.jsonl
python -m kg_reasoning prepare-multihop-training-data --queries artifacts/aihub/multihop_queries.jsonl --hops artifacts/aihub/reasoning_hops.jsonl --output artifacts/aihub/multihop_kg_response_train.jsonl
python -m kg_reasoning filter-multihop-training-data --input artifacts/aihub/multihop_kg_response_train.jsonl --output artifacts/aihub/multihop_kg_response_train_strategy_strict_clean.jsonl --drop-event-hops --drop-weak-indirectness --rewrite-answers
python -m kg_reasoning build-graph --triples artifacts/aihub/triples.jsonl --output artifacts/aihub/graph.json
python -m kg_reasoning train-encoder --graph artifacts/aihub/graph.json --output artifacts/aihub/encoder.json
```

That means the project no longer depends on the lost original artifacts. Once the raw AIHub files are present, the preprocessing path can be rebuilt from scratch, including event-only KG and pragmatic multi-hop KG-response supervision. For the first SFT run, prefer the strict clean file: it removes noisy event-grounded hops, generic rule events, and weak indirect-speech-act labels, then rewrites template answers into pragmatic response-strategy supervision.

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

## Legacy-Compatible Commands

Generate the original ATOMIC-style QA supervision set from extracted triples:

```powershell
python -m kg_reasoning generate-qa --triples artifacts/aihub/triples.jsonl --output artifacts/aihub/kg_qa_samples.jsonl
```

That command ports the old `generate_kg_qa.py` behavior into the new package without keeping secrets in source code.

## Security

The old repository exposed API keys directly in source code. Those keys should be treated as compromised and rotated immediately in any upstream environment, even though they are no longer present in this rebuilt working tree.

## Test

```powershell
python -m unittest discover -s tests -v
```
