from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_reasoning.bootstrap import prepare_demo_assets
from kg_reasoning.event_extraction import extract_events
from kg_reasoning.graph import build_graph, load_graph, save_graph
from kg_reasoning.inference import compare_reasoning
from kg_reasoning.io import load_dialogues, load_events, load_triples, write_jsonl
from kg_reasoning.llm import OpenAICompatibleClient
from kg_reasoning.relation_extraction import extract_relations
from kg_reasoning.training import TrainingConfig, load_encoder, save_encoder, train_relational_encoder


DEFAULT_BOOTSTRAP = Path("data/bootstrap/demo_dialogues.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args) or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuilt Korean dialogue KG reasoning toolkit")
    subparsers = parser.add_subparsers(dest="command")

    prepare_demo_parser = subparsers.add_parser("prepare-demo", help="materialize the curated bootstrap dataset")
    prepare_demo_parser.add_argument("--input", default=str(DEFAULT_BOOTSTRAP))
    prepare_demo_parser.add_argument("--output-dir", default="artifacts/demo")
    prepare_demo_parser.set_defaults(handler=handle_prepare_demo)

    extract_events_parser = subparsers.add_parser("extract-events", help="extract events from dialogues")
    extract_events_parser.add_argument("--dialogues", required=True)
    extract_events_parser.add_argument("--output", required=True)
    extract_events_parser.add_argument("--provider", choices=("seed", "rule", "hybrid", "llm"), default="hybrid")
    extract_events_parser.set_defaults(handler=handle_extract_events)

    extract_relations_parser = subparsers.add_parser("extract-relations", help="expand events into ATOMIC-style triples")
    extract_relations_parser.add_argument("--dialogues", required=True)
    extract_relations_parser.add_argument("--events", required=True)
    extract_relations_parser.add_argument("--output", required=True)
    extract_relations_parser.add_argument("--provider", choices=("seed", "rule", "hybrid", "llm"), default="hybrid")
    extract_relations_parser.set_defaults(handler=handle_extract_relations)

    build_graph_parser = subparsers.add_parser("build-graph", help="build a graph bundle from triples")
    build_graph_parser.add_argument("--triples", required=True)
    build_graph_parser.add_argument("--output", required=True)
    build_graph_parser.set_defaults(handler=handle_build_graph)

    train_parser = subparsers.add_parser("train-encoder", help="train the lightweight relational encoder")
    train_parser.add_argument("--graph", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--dim", type=int, default=96)
    train_parser.add_argument("--epochs", type=int, default=120)
    train_parser.add_argument("--layers", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=0.06)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.set_defaults(handler=handle_train_encoder)

    infer_parser = subparsers.add_parser("infer", help="compare baseline and KG-aware reasoning")
    infer_parser.add_argument("--graph", required=True)
    infer_parser.add_argument("--encoder", required=True)
    infer_parser.add_argument("--triples", required=True)
    infer_parser.add_argument("--query", required=True)
    infer_parser.add_argument("--top-k", type=int, default=5)
    infer_parser.add_argument("--mode", choices=("template", "llm"), default="template")
    infer_parser.set_defaults(handler=handle_infer)

    run_demo_parser = subparsers.add_parser("run-demo", help="run the complete bootstrap pipeline")
    run_demo_parser.add_argument("--query", required=True)
    run_demo_parser.add_argument("--workspace", default="artifacts/demo_run")
    run_demo_parser.add_argument("--mode", choices=("template", "llm"), default="template")
    run_demo_parser.set_defaults(handler=handle_run_demo)
    return parser


def handle_prepare_demo(args: argparse.Namespace) -> int:
    outputs = prepare_demo_assets(dialogues_path=args.input, output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))
    return 0


def handle_extract_events(args: argparse.Namespace) -> int:
    dialogues = load_dialogues(args.dialogues)
    client = OpenAICompatibleClient.from_environment()
    events = extract_events(dialogues=dialogues, provider=args.provider, client=client)
    write_jsonl(args.output, [event.to_dict() for event in events])
    print(json.dumps({"events": len(events), "output": args.output}, ensure_ascii=False, indent=2))
    return 0


def handle_extract_relations(args: argparse.Namespace) -> int:
    dialogues = load_dialogues(args.dialogues)
    events = load_events(args.events)
    client = OpenAICompatibleClient.from_environment()
    triples = extract_relations(dialogues=dialogues, events=events, provider=args.provider, client=client)
    write_jsonl(args.output, [triple.to_dict() for triple in triples])
    print(json.dumps({"triples": len(triples), "output": args.output}, ensure_ascii=False, indent=2))
    return 0


def handle_build_graph(args: argparse.Namespace) -> int:
    triples = load_triples(args.triples)
    graph = build_graph(triples)
    save_graph(args.output, graph)
    print(
        json.dumps(
            {
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def handle_train_encoder(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph)
    config = TrainingConfig(
        dim=args.dim,
        epochs=args.epochs,
        layers=args.layers,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    encoder = train_relational_encoder(graph=graph, config=config)
    save_encoder(args.output, encoder)
    print(json.dumps({"output": args.output, "metrics": encoder["metrics"]}, ensure_ascii=False, indent=2))
    return 0


def handle_infer(args: argparse.Namespace) -> int:
    graph = load_graph(args.graph)
    encoder = load_encoder(args.encoder)
    triples = load_triples(args.triples)
    client = OpenAICompatibleClient.from_environment()
    result = compare_reasoning(
        query=args.query,
        graph=graph,
        encoder=encoder,
        triples=triples,
        top_k=args.top_k,
        client=client,
        mode=args.mode,
    )
    print(render_comparison(result))
    return 0


def handle_run_demo(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    outputs = prepare_demo_assets(dialogues_path=DEFAULT_BOOTSTRAP, output_dir=workspace)
    graph_path = workspace / "graph.json"
    encoder_path = workspace / "encoder.json"

    triples = load_triples(outputs["triples"])
    graph = build_graph(triples)
    save_graph(graph_path, graph)
    encoder = train_relational_encoder(graph=graph)
    save_encoder(encoder_path, encoder)

    client = OpenAICompatibleClient.from_environment()
    result = compare_reasoning(
        query=args.query,
        graph=graph,
        encoder=encoder,
        triples=triples,
        top_k=5,
        client=client,
        mode=args.mode,
    )
    print(render_comparison(result))
    return 0


def render_comparison(result: dict) -> str:
    parts = [
        "[Baseline]",
        result["baseline"],
        "",
        "[KG-Aware]",
        result["kg_aware"],
    ]
    return "\n".join(parts)
