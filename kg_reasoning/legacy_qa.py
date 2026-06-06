from __future__ import annotations

import random
import time

from kg_reasoning.legacy_prompts import (
    RELATION_DEFINITION,
    RELATION_QUESTION,
    build_legacy_qa_prompt,
    is_low_quality_qa_answer,
)
from kg_reasoning.llm import OpenAICompatibleClient
from kg_reasoning.schema import TripleRecord


def generate_qa_samples(
    triples: list[TripleRecord],
    client: OpenAICompatibleClient,
    *,
    shuffle: bool = True,
    sleep_seconds: float = 0.0,
) -> list[dict]:
    filtered = [triple for triple in triples if triple.relation in RELATION_DEFINITION]
    if shuffle:
        random.shuffle(filtered)

    samples: list[dict] = []
    for triple in filtered:
        prompt = build_legacy_qa_prompt(head=triple.head, relation=triple.relation)
        try:
            answer = client.chat_messages(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            ).strip()
        except Exception:
            continue
        if is_low_quality_qa_answer(triple.head, answer, triple.relation):
            continue
        samples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{triple.head}\n\n{RELATION_QUESTION[triple.relation]}",
                    },
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ],
                "source_triple": {
                    "head": triple.head,
                    "relation": triple.relation,
                    "tail": triple.tail,
                },
            }
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return samples
