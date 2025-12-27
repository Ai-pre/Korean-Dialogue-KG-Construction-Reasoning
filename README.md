# Korean Dialogue-based KG-aware Reasoning System

**R-GCN + ATOMIC-style Knowledge Graph for Commonsense Inference**

## 📌 Overview

This project proposes a **KG-aware reasoning framework** for Korean daily dialogue, addressing a critical limitation of large language models (LLMs):

> **LLMs often fail to maintain consistent commonsense causal reasoning (intent–effect–emotion) in Korean conversational contexts.**

Instead of relying solely on parametric knowledge inside LLMs, this work introduces an **external structured commonsense Knowledge Graph (KG)** and integrates it into the inference pipeline using:

* **ATOMIC-style relational structure**
* **R-GCN (Relational Graph Convolutional Network)**
* **Event-level retrieval + prompt injection**

The system demonstrates that **KG-augmented inference produces more stable, causal-consistent, and hallucination-resistant outputs** than KG-free baselines.

---

## 🔍 Problem Motivation

While modern LLMs generate fluent Korean text, we observed repeated failures in:

* Mixing causal directions (cause ↔ effect)
* Confusing intent, emotion, and reaction types
* Making unjustified inference jumps (hallucination)
* Producing overly abstract or generic explanations

These issues are especially severe in **short, ambiguous daily dialogue**, where commonsense grounding is required.

---

## 🧠 Core Idea

LLMs should **not infer everything alone**.

Instead:

1. **Extract events from dialogue**
2. **Ground them in a structured commonsense KG**
3. **Use KG as a latent reasoning space**, not a direct answer source
4. **Guide LLM reasoning paths via retrieved KG context**

---

## 🏗 System Architecture

```
User Dialogue
      ↓
Event Extraction (GPT-4o mini)
      ↓
ATOMIC-style Triple Generation (Korean)
      ↓
Knowledge Graph Construction
      ↓
R-GCN Training (Node Contextualization)
      ↓
EventMatcher (Semantic Retrieval)
      ↓
KG-aware Prompt Injection
      ↓
LLM Inference (Qwen / LLaMA)
```

---

## 🧱 Knowledge Graph Construction

### 🔹 Why Not Translate English ATOMIC?

We initially attempted to translate the English ATOMIC dataset, but encountered major issues:

* PersonX / PersonY templates break Korean naturalness
* Subject ambiguity and relation direction confusion
* Unstable node semantics for graph learning

➡️ **Decision**:
❌ Drop English ATOMIC
✅ Build **Korean-native ATOMIC-style KG from scratch**

---

### 🔹 Event & Relation Generation

Using **GPT-4o mini**, we automatically generate:

* **Event** (Korean natural sentence)
* **ATOMIC relations (9 types)**:

  * `xIntent`, `xNeed`, `xEffect`, `xReact`, `xWant`
  * `oEffect`, `oReact`, `oWant`
  * `xAttr`

Strict constraints were enforced:

* One relation = one sentence
* No moralizing / over-generalization
* No abstract norms
* Causality must be explicit

---

## 🧪 KG Quality Control

Randomly sampled **500 triples** were evaluated via GPT-4o mini on:

* **Consistency**
* **Commonsense Plausibility**
* **Factuality**

| Metric      | Score          |
| ----------- | -------------- |
| Consistency | 4.00           |
| Commonsense | 4.40           |
| Factuality  | 4.00           |
| **Average** | **4.13 / 5.0** |

Low-quality triples were aggressively discarded.
➡️ **Quality > Quantity**

---

## 🔗 R-GCN Reasoning Module

### Purpose

R-GCN is **not** used to output answers.

Instead, it learns:

> *How events are contextually connected via relational structure*

### Design

* **Node**: Korean event sentence
* **Edge**: ATOMIC relation type
* **Output**: Context-aware event embeddings

This forms a **latent reasoning space** that helps guide LLM inference.

---

## 🔎 EventMatcher (Retrieval Layer)

At inference time:

1. User input is embedded
2. Most semantically similar event nodes are retrieved
3. Their neighboring relations (`xIntent`, `xReact`, etc.) are collected
4. Retrieved knowledge is summarized and injected into the prompt

This prevents blind generation and anchors reasoning.

---

## 🤖 KG-aware Inference Pipeline

1. Encode user input
2. Retrieve relevant KG events via EventMatcher
3. Extract relational context (subgraph)
4. Inject KG facts into LLM prompt
5. Generate grounded response

> **KG is used as contextual guidance, not as explicit facts to parrot**

---

## 🧪 Experimental Strategy (KG ON / OFF)

To verify KG effectiveness, we conducted **controlled A/B testing**.

### Settings

* **Baseline (KG-Free)**
  LLM inference using only pretrained knowledge
* **Proposed (KG-Augmented)**
  Same LLM + retrieved KG context injected into prompt

### Model

* `llama-3-Korean-Bllossom-8B`

---

## 📊 Evaluation Criteria

### Qualitative Focus

* **Causal Consistency**
* **Emotion–Intent–Effect alignment**
* **Hallucination frequency**
* **Specificity vs abstraction**
* **Inference stability**

---

## 📈 Results & Analysis

### Case Study

**Input**:

> “배가 너무 아파서 조퇴하고 싶어.”

#### ❌ KG-Free (Baseline)

* Misinterpreted “배” metaphorically
* Injected unrelated school stress narratives
* Produced abstract, incoherent reasoning

➡️ **Hallucination + causal failure**

#### ✅ KG-Augmented

* Retrieved KG facts:

  * Overeating → stomach pain
  * Pain → desire to rest
  * Honest expression of discomfort
* Built clear causal chain:

  ```
  과식 → 복통 → 휴식 필요 → 조퇴 의도
  ```

➡️ **Grounded, specific, causally coherent response**

---

## ✅ Conclusion

This project demonstrates that:

* LLMs alone are insufficient for stable commonsense reasoning
* **Structured KG acts as a causal anchor**
* KG-aware prompting significantly reduces hallucination
* Reasoning quality improves without modifying LLM weights

> **The key is not bigger models, but better knowledge structuring and retrieval.**

---

## 🔮 Future Work

* Multi-hop reasoning over KG
* Retrieval-aware selective knowledge injection
* Dynamic relation-path reasoning (`xIntent → xEffect → oReact`)
* Alignment-aware KG filtering

---

## 📁 Repository Structure (Recommended)

```
kg_project/
├── data/
│   └── korean_atomic_triples.json
├── graph/
│   └── graph_data.pkl
├── rgcn/
│   ├── rgcn_model.py
│   └── train_rgcn.py
├── inference/
│   ├── event_matcher.py
│   ├── kg_on_off.py
│   └── prompt_builder.py
└── README.md
```
