from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kg_reasoning.bootstrap import prepare_demo_assets
from kg_reasoning.graph import build_graph
from kg_reasoning.inference import compare_reasoning
from kg_reasoning.io import load_triples
from kg_reasoning.training import train_relational_encoder


class PipelineSmokeTest(unittest.TestCase):
    def test_demo_pipeline_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = prepare_demo_assets("data/bootstrap/demo_dialogues.jsonl", tmpdir)
            triples = load_triples(output["triples"])
            graph = build_graph(triples)
            encoder = train_relational_encoder(graph)
            result = compare_reasoning(
                query="배가 너무 아파서 조퇴하고 싶어.",
                graph=graph,
                encoder=encoder,
                triples=triples,
                top_k=3,
            )

        self.assertIn("KG-Aware", f"[KG-Aware]\n{result['kg_aware']}")
        self.assertIn("근거", result["kg_aware"])
        self.assertGreater(len(result["grounded_triples"]), 0)


if __name__ == "__main__":
    unittest.main()
