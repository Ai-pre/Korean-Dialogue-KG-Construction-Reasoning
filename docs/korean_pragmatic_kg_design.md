# Korean Pragmatic KG Design

## Goal

This project should not stop at "Korean ATOMIC-style triples."

The stronger goal is:

> Build a Korean dialogue-native knowledge graph that captures events, pragmatics, and speaker relations, then use that graph to augment LLM reasoning and response generation.

That means the graph must model not only:

- what happened

but also:

- how it was said
- what was implied
- who was speaking to whom
- how social distance and politeness changed the meaning

## Why This Is Korean-Specific

Korean dialogue contains systematic phenomena that are weakly covered by generic commonsense graphs:

- frequent subject/object omission
- indirect refusals and softened disagreement
- politeness/register changes
- sentence endings that encode stance and certainty
- family/seniority/service relations that shape interpretation
- humor, banter, and face-saving behavior in casual conversation

Therefore, Korean KG augmentation should target:

- event understanding
- pragmatic interpretation
- relation-aware response generation

instead of only event-to-ATOMIC expansion.

## Three-Layer Graph

### 1. Event Layer

This is the existing project core.

- event_sentence
- event_cause
- ATOMIC-style relations

Purpose:

- capture what happened
- support standard commonsense reasoning

### 2. Pragmatics Layer

This is the Korean-specialized layer.

Core signal families:

- `speechAct`
- `stance`
- `politeness`
- `indirectness`
- `emotionCue`
- `listenerPressure`
- `facework`
- `humor`
- `ellipsisSubject`
- `ellipsisTarget`
- `agreementShift`
- `certainty`

Purpose:

- capture how meaning is realized in dialogue
- recover hidden intent from softened or incomplete utterances

### 3. Social Relation Layer

This layer encodes speaker pair structure.

- `relation_label`
- `social_distance`
- `power_relation`

Purpose:

- explain why the same utterance can mean different things across relations
- generate relation-consistent responses

## Node and Edge Philosophy

The graph should support both:

- symbolic retrieval
- multi-hop reasoning

Recommended node families:

- utterance nodes
- event nodes
- pragmatic nodes
- speaker-relation nodes

Recommended edge families:

- utterance -> pragmatic signal
- pragmatic signal -> event
- pragmatic signal -> speaker relation
- event -> ATOMIC tail
- event -> event (temporal or causal links later)

## Suggested Korean Relation Taxonomy

### Event Commonsense

- `xIntent`
- `xNeed`
- `xEffect`
- `xReact`
- `xWant`
- `oEffect`
- `oReact`
- `oWant`
- `xAttr`

### Pragmatic Signals

- `speechAct`
- `stance`
- `politeness`
- `indirectness`
- `emotionCue`
- `listenerPressure`
- `facework`
- `humor`
- `ellipsisSubject`
- `ellipsisTarget`
- `agreementShift`
- `certainty`

### Social Relations

- `relation_label`
- `social_distance`
- `power_relation`

## Multi-Hop Reasoning Targets

The most important upgrade over event-only KG is to build Korean-native multi-hop reasoning queries.

### Query Type 1: Intent From Context

Path:

- utterance -> stance
- stance -> event

Question:

- What is the hidden intent of this utterance when the event context is considered?

### Query Type 2: Indirect Speech Act

Path:

- utterance -> indirectness
- indirectness -> speechAct

Question:

- Is this really advice, complaint, refusal, or soft disagreement?

### Query Type 3: Ellipsis Recovery

Path:

- utterance -> ellipsisSubject
- utterance -> ellipsisTarget

Question:

- Who or what is omitted in the utterance?

### Query Type 4: Relation-Aware Response

Path:

- utterance -> politeness
- politeness -> speaker relation

Question:

- What response preserves the current relation and register?

### Query Type 5: Emotion Cascade

Path:

- event -> effect
- utterance -> emotionCue

Question:

- How is emotion shifting across the dialogue?

## Evaluation And Training Views

This is not a separate benchmark detour. The goal is to make the KG-response method measurable by training and evaluating several views over the same Korean pragmatic KG.

Recommended tasks:

1. implicit intent inference
2. indirect speech act classification
3. ellipsis recovery
4. relation-aware response generation
5. KG-grounded explanation generation

## Implemented Pipeline

The current rebuild now supports this pragmatic KG-response path:

```powershell
python -m kg_reasoning import-aihub --input data/raw/aihub --output artifacts/aihub/dialogues.jsonl
python -m kg_reasoning extract-events --dialogues artifacts/aihub/dialogues.jsonl --output artifacts/aihub/events.jsonl
python -m kg_reasoning extract-pragmatics --dialogues artifacts/aihub/dialogues.jsonl --signals-output artifacts/aihub/pragmatic_signals.jsonl --relations-output artifacts/aihub/speaker_relations.jsonl
python -m kg_reasoning build-multihop-queries --dialogues artifacts/aihub/dialogues.jsonl --events artifacts/aihub/events.jsonl --pragmatics artifacts/aihub/pragmatic_signals.jsonl --speaker-relations artifacts/aihub/speaker_relations.jsonl --hops-output artifacts/aihub/reasoning_hops.jsonl --queries-output artifacts/aihub/multihop_queries.jsonl
python -m kg_reasoning prepare-multihop-training-data --queries artifacts/aihub/multihop_queries.jsonl --hops artifacts/aihub/reasoning_hops.jsonl --output artifacts/aihub/multihop_kg_response_train.jsonl
```

The key upgrade is that training examples are grounded in paths such as:

- utterance -> pragmatic signal -> event
- utterance -> pragmatic signal -> speaker relation
- utterance -> emotion/stance/politeness cues -> response direction

## Ablation Strategy

The project should report at least:

- Base LLM
- Base LLM + text retrieval
- Base LLM + event-only KG
- Base LLM + event + pragmatics KG
- Base LLM + event + pragmatics + relation KG
- Gold KG
- Blank-KG ablation

This is the cleanest way to prove that Korean KG augmentation adds value beyond ordinary instruction tuning.

## Project Claim

The strongest project claim is not:

> We built a Korean knowledge graph.

It is:

> We built a Korean dialogue-native pragmatic commonsense graph and showed that it improves LLM grounding, hidden-intent reasoning, and relation-consistent response generation.
