from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_reasoning.aihub import import_aihub_dataset
from kg_reasoning.augmentation import augment_dialogues
from kg_reasoning.bootstrap import prepare_demo_assets
from kg_reasoning.event_extraction import extract_events
from kg_reasoning.graph import build_graph, load_graph, save_graph
from kg_reasoning.inference import compare_reasoning
from kg_reasoning.io import (
    ensure_parent,
    iter_jsonl,
    load_dialogues,
    load_events,
    load_multihop_queries,
    load_reasoning_hops,
    load_speaker_relations,
    load_triples,
    write_jsonl,
)
from kg_reasoning.legacy_formats import (
    load_legacy_event_blocks,
    load_legacy_triple_blocks,
    write_legacy_events_jsonl,
    write_legacy_triples_jsonl,
)
from kg_reasoning.legacy_pipeline import export_dialogues_to_txt, process_legacy_dialogue_files
from kg_reasoning.legacy_qa import generate_qa_samples
from kg_reasoning.llm import OpenAICompatibleClient
from kg_reasoning.multihop_filter import parse_type_caps, select_balanced_examples
from kg_reasoning.multihop import build_multihop_queries
from kg_reasoning.multihop_answering import rewrite_multihop_training_answer
from kg_reasoning.pragmatic_extraction import attach_event_links_to_pragmatics, extract_pragmatics
from kg_reasoning.relation_extraction import extract_relations
from kg_reasoning.schema import PragmaticSignalRecord
from kg_reasoning.training import TrainingConfig, load_encoder, save_encoder, train_relational_encoder
from kg_reasoning.training_data import build_multihop_training_rows, build_training_examples


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

    import_aihub_parser = subparsers.add_parser("import-aihub", help="convert AIHub dialogue JSON into project dialogue JSONL")
    import_aihub_parser.add_argument("--input", required=True, help="AIHub JSON file or directory")
    import_aihub_parser.add_argument("--output", required=True, help="output dialogue JSONL path")
    import_aihub_parser.add_argument("--limit", type=int, default=None)
    import_aihub_parser.add_argument("--min-utterances", type=int, default=2)
    import_aihub_parser.add_argument("--include-system-utterances", action="store_true")
    import_aihub_parser.set_defaults(handler=handle_import_aihub)

    export_txt_parser = subparsers.add_parser("export-dialogues-txt", help="export dialogue JSONL into legacy-style .txt files")
    export_txt_parser.add_argument("--dialogues", required=True)
    export_txt_parser.add_argument("--output-dir", required=True)
    export_txt_parser.add_argument("--limit", type=int, default=None)
    export_txt_parser.set_defaults(handler=handle_export_dialogues_txt)

    legacy_process_parser = subparsers.add_parser(
        "legacy-openai-process",
        help="run the original file-by-file OpenAI pipeline and write dialog_events.json / dialog_triples.json",
    )
    legacy_process_parser.add_argument("--src-dir", default=None, help="directory containing legacy .txt dialogue files")
    legacy_process_parser.add_argument("--dialogues", default=None, help="optional dialogue JSONL to export before processing")
    legacy_process_parser.add_argument("--txt-dir", default="artifacts/legacy_txt", help="txt export dir when --dialogues is used")
    legacy_process_parser.add_argument("--events-output", required=True)
    legacy_process_parser.add_argument("--triples-output", required=True)
    legacy_process_parser.add_argument("--events-jsonl-output", default=None)
    legacy_process_parser.add_argument("--triples-jsonl-output", default=None)
    legacy_process_parser.add_argument("--limit", type=int, default=None)
    legacy_process_parser.set_defaults(handler=handle_legacy_openai_process)

    convert_legacy_events_parser = subparsers.add_parser(
        "convert-legacy-events",
        help="convert legacy dialog_events.json array into event JSONL",
    )
    convert_legacy_events_parser.add_argument("--input", required=True)
    convert_legacy_events_parser.add_argument("--output", required=True)
    convert_legacy_events_parser.set_defaults(handler=handle_convert_legacy_events)

    convert_legacy_triples_parser = subparsers.add_parser(
        "convert-legacy-triples",
        help="convert legacy dialog_triples.json array into triple JSONL",
    )
    convert_legacy_triples_parser.add_argument("--input", required=True)
    convert_legacy_triples_parser.add_argument("--output", required=True)
    convert_legacy_triples_parser.set_defaults(handler=handle_convert_legacy_triples)

    extract_events_parser = subparsers.add_parser("extract-events", help="extract events from dialogues")
    extract_events_parser.add_argument("--dialogues", required=True)
    extract_events_parser.add_argument("--output", required=True)
    extract_events_parser.add_argument(
        "--provider",
        choices=("seed", "rule", "hybrid", "llm", "legacy-llm", "legacy-hybrid"),
        default="legacy-hybrid",
    )
    extract_events_parser.set_defaults(handler=handle_extract_events)

    extract_relations_parser = subparsers.add_parser("extract-relations", help="expand events into ATOMIC-style triples")
    extract_relations_parser.add_argument("--dialogues", required=True)
    extract_relations_parser.add_argument("--events", required=True)
    extract_relations_parser.add_argument("--output", required=True)
    extract_relations_parser.add_argument(
        "--provider",
        choices=("seed", "rule", "hybrid", "llm", "legacy-llm", "legacy-hybrid"),
        default="legacy-hybrid",
    )
    extract_relations_parser.set_defaults(handler=handle_extract_relations)

    extract_pragmatics_parser = subparsers.add_parser(
        "extract-pragmatics",
        help="extract Korean pragmatic signals and speaker relations from dialogues",
    )
    extract_pragmatics_parser.add_argument("--dialogues", required=True)
    extract_pragmatics_parser.add_argument("--signals-output", "--output-signals", required=True)
    extract_pragmatics_parser.add_argument("--relations-output", "--output-relations", required=True)
    extract_pragmatics_parser.add_argument("--provider", choices=("rule",), default="rule")
    extract_pragmatics_parser.set_defaults(handler=handle_extract_pragmatics)

    multihop_parser = subparsers.add_parser(
        "build-multihop-queries",
        help="build pragmatic KG reasoning hops and multi-hop query records",
    )
    multihop_parser.add_argument("--dialogues", required=True)
    multihop_parser.add_argument("--events", required=True)
    multihop_parser.add_argument("--pragmatics", required=True)
    multihop_parser.add_argument("--speaker-relations", required=True)
    multihop_parser.add_argument("--hops-output", "--output-hops", required=True)
    multihop_parser.add_argument("--queries-output", "--output-queries", required=True)
    multihop_parser.add_argument("--max-queries-per-dialogue", type=int, default=8)
    multihop_parser.set_defaults(handler=handle_build_multihop_queries)

    build_graph_parser = subparsers.add_parser("build-graph", help="build a graph bundle from triples")
    build_graph_parser.add_argument("--triples", required=True)
    build_graph_parser.add_argument("--output", required=True)
    build_graph_parser.set_defaults(handler=handle_build_graph)

    train_parser = subparsers.add_parser("train-encoder", help="train the lightweight relational encoder")
    train_parser.add_argument("--graph", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--backend", choices=("lightweight", "legacy-rgcn"), default="lightweight")
    train_parser.add_argument("--dim", type=int, default=96)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
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

    augment_parser = subparsers.add_parser("augment-dialogues", help="attach KG evidence and KG-aware summaries to dialogues")
    augment_parser.add_argument("--dialogues", required=True)
    augment_parser.add_argument("--graph", required=True)
    augment_parser.add_argument("--encoder", required=True)
    augment_parser.add_argument("--triples", required=True)
    augment_parser.add_argument("--output", required=True)
    augment_parser.add_argument("--limit", type=int, default=None)
    augment_parser.add_argument("--top-k", type=int, default=5)
    augment_parser.add_argument("--mode", choices=("template", "llm"), default="template")
    augment_parser.add_argument("--query-source", choices=("last-utterance", "history", "full-dialogue"), default="last-utterance")
    augment_parser.add_argument("--evidence-source", choices=("dialogue-linked", "history-linked", "retrieval"), default="dialogue-linked")
    augment_parser.add_argument("--history-window", type=int, default=6)
    augment_parser.set_defaults(handler=handle_augment_dialogues)

    prepare_training_parser = subparsers.add_parser(
        "prepare-training-data",
        help="convert augmented dialogue records into chat-training JSONL",
    )
    prepare_training_parser.add_argument("--input", required=True)
    prepare_training_parser.add_argument("--output", required=True)
    prepare_training_parser.add_argument(
        "--task",
        choices=("next-turn", "kg-response", "multi-hop-kg-response"),
        default="next-turn",
    )
    prepare_training_parser.set_defaults(handler=handle_prepare_training_data)

    prepare_multihop_parser = subparsers.add_parser(
        "prepare-multihop-training-data",
        help="convert multi-hop query/hop JSONL into chat-training JSONL",
    )
    prepare_multihop_parser.add_argument("--queries", required=True)
    prepare_multihop_parser.add_argument("--hops", required=True)
    prepare_multihop_parser.add_argument("--output", required=True)
    prepare_multihop_parser.set_defaults(handler=handle_prepare_multihop_training_data)

    filter_multihop_parser = subparsers.add_parser(
        "filter-multihop-training-data",
        help="create a balanced clean subset from multi-hop KG-response chat-training JSONL",
    )
    filter_multihop_parser.add_argument("--input", required=True)
    filter_multihop_parser.add_argument("--output", required=True)
    filter_multihop_parser.add_argument("--type-caps", default=None, help="comma list like relation_aware_response=6000,indirect_speech_act=6000")
    filter_multihop_parser.add_argument("--min-hops", type=int, default=3)
    filter_multihop_parser.add_argument("--keep-generic-events", action="store_true")
    filter_multihop_parser.add_argument(
        "--drop-event-hops",
        action="store_true",
        help="drop examples that depend on an event grounding hop",
    )
    filter_multihop_parser.add_argument(
        "--drop-weak-indirectness",
        action="store_true",
        help="drop weak softened indirectness examples triggered only by ellipsis or emotive surface markers",
    )
    filter_multihop_parser.add_argument("--drop-unknown-relation", action="store_true")
    filter_multihop_parser.add_argument(
        "--rewrite-answers",
        action="store_true",
        help="replace template assistant answers with richer pragmatic response-strategy answers",
    )
    filter_multihop_parser.add_argument("--seed", type=int, default=42)
    filter_multihop_parser.set_defaults(handler=handle_filter_multihop_training_data)

    qa_parser = subparsers.add_parser("generate-qa", help="generate the original ATOMIC-style QA samples from triples")
    qa_parser.add_argument("--triples", required=True)
    qa_parser.add_argument("--output", required=True)
    qa_parser.add_argument("--sleep-seconds", type=float, default=0.0)
    qa_parser.set_defaults(handler=handle_generate_qa)

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


def handle_import_aihub(args: argparse.Namespace) -> int:
    stats = import_aihub_dataset(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        include_system_utterances=args.include_system_utterances,
        min_utterances=args.min_utterances,
    )
    payload = stats.to_dict()
    payload["output"] = args.output
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def handle_export_dialogues_txt(args: argparse.Namespace) -> int:
    dialogues = load_dialogues(args.dialogues)
    stats = export_dialogues_to_txt(dialogues=dialogues, output_dir=args.output_dir, limit=args.limit)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0


def handle_legacy_openai_process(args: argparse.Namespace) -> int:
    client = OpenAICompatibleClient.from_environment()
    if client is None:
        raise RuntimeError("legacy-openai-process requires an OpenAI-compatible endpoint in the environment.")

    src_dir = args.src_dir
    export_payload: dict[str, int | str] | None = None
    if args.dialogues:
        dialogues = load_dialogues(args.dialogues)
        export_stats = export_dialogues_to_txt(dialogues=dialogues, output_dir=args.txt_dir, limit=args.limit)
        export_payload = export_stats.to_dict()
        src_dir = args.txt_dir
    if not src_dir:
        raise RuntimeError("legacy-openai-process requires either --src-dir or --dialogues.")

    stats = process_legacy_dialogue_files(
        src_dir=src_dir,
        out_event_json=args.events_output,
        out_triple_json=args.triples_output,
        client=client,
        limit=args.limit if not args.dialogues else None,
        out_event_jsonl=args.events_jsonl_output,
        out_triple_jsonl=args.triples_jsonl_output,
    )
    payload = stats.to_dict()
    if export_payload is not None:
        payload["export"] = export_payload
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def handle_convert_legacy_events(args: argparse.Namespace) -> int:
    blocks = load_legacy_event_blocks(args.input)
    write_legacy_events_jsonl(blocks, args.output)
    total_events = sum(len(block.get("events", [])) for block in blocks)
    print(json.dumps({"blocks": len(blocks), "events": total_events, "output": args.output}, ensure_ascii=False, indent=2))
    return 0


def handle_convert_legacy_triples(args: argparse.Namespace) -> int:
    blocks = load_legacy_triple_blocks(args.input)
    write_legacy_triples_jsonl(blocks, args.output)
    total_triples = sum(len(block.get("triples", [])) for block in blocks)
    print(json.dumps({"blocks": len(blocks), "triples": total_triples, "output": args.output}, ensure_ascii=False, indent=2))
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


def handle_extract_pragmatics(args: argparse.Namespace) -> int:
    dialogues = load_dialogues(args.dialogues)
    signals, relations = extract_pragmatics(dialogues=dialogues, provider=args.provider)
    write_jsonl(args.signals_output, [signal.to_dict() for signal in signals])
    write_jsonl(args.relations_output, [relation.to_dict() for relation in relations])
    print(
        json.dumps(
            {
                "signals": len(signals),
                "relations": len(relations),
                "signals_output": args.signals_output,
                "relations_output": args.relations_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def handle_build_multihop_queries(args: argparse.Namespace) -> int:
    dialogues = load_dialogues(args.dialogues)
    events = load_events(args.events)
    relations = load_speaker_relations(args.speaker_relations)
    dialogues_by_id = {dialogue.dialogue_id: dialogue for dialogue in dialogues}
    events_by_dialogue = group_records_by_dialogue(events)
    relations_by_dialogue = group_records_by_dialogue(relations)

    hops_output = ensure_parent(args.hops_output)
    queries_output = ensure_parent(args.queries_output)
    total_hops = 0
    total_queries = 0
    signal_groups = 0
    skipped_signal_groups = 0
    with hops_output.open("w", encoding="utf-8") as hops_handle, queries_output.open("w", encoding="utf-8") as queries_handle:
        for dialogue_id, signals in iter_pragmatic_signal_groups(args.pragmatics):
            signal_groups += 1
            dialogue = dialogues_by_id.get(dialogue_id)
            if dialogue is None:
                skipped_signal_groups += 1
                continue
            dialogue_events = events_by_dialogue.get(dialogue_id, [])
            dialogue_relations = relations_by_dialogue.get(dialogue_id, [])
            linked_signals = attach_event_links_to_pragmatics(signals, dialogue_events)
            hops, queries = build_multihop_queries(
                dialogue=dialogue,
                events=dialogue_events,
                pragmatic_signals=linked_signals,
                speaker_relations=dialogue_relations,
                max_queries=args.max_queries_per_dialogue,
            )
            for hop in hops:
                hops_handle.write(json.dumps(hop.to_dict(), ensure_ascii=False) + "\n")
            for query in queries:
                queries_handle.write(json.dumps(query.to_dict(), ensure_ascii=False) + "\n")
            total_hops += len(hops)
            total_queries += len(queries)

    print(
        json.dumps(
            {
                "hops": total_hops,
                "queries": total_queries,
                "signal_groups": signal_groups,
                "skipped_signal_groups": skipped_signal_groups,
                "hops_output": args.hops_output,
                "queries_output": args.queries_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def group_records_by_dialogue(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record.dialogue_id, []).append(record)
    return grouped


def iter_pragmatic_signal_groups(path: str):
    current_dialogue_id = None
    current_signals = []
    for row in iter_jsonl(path):
        signal = PragmaticSignalRecord.from_dict(row)
        if current_dialogue_id is None:
            current_dialogue_id = signal.dialogue_id
        if signal.dialogue_id != current_dialogue_id:
            yield current_dialogue_id, current_signals
            current_dialogue_id = signal.dialogue_id
            current_signals = []
        current_signals.append(signal)
    if current_dialogue_id is not None:
        yield current_dialogue_id, current_signals


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
        backend=args.backend,
        dim=args.dim,
        hidden_dim=args.hidden_dim,
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


def handle_augment_dialogues(args: argparse.Namespace) -> int:
    dialogues = load_dialogues(args.dialogues)
    graph = load_graph(args.graph)
    encoder = load_encoder(args.encoder)
    triples = load_triples(args.triples)
    client = OpenAICompatibleClient.from_environment()
    records = augment_dialogues(
        dialogues=dialogues,
        graph=graph,
        encoder=encoder,
        triples=triples,
        top_k=args.top_k,
        mode=args.mode,
        client=client,
        query_source=args.query_source,
        evidence_source=args.evidence_source,
        history_window=args.history_window,
        limit=args.limit,
    )
    write_jsonl(args.output, records)
    print(json.dumps({"dialogues": len(records), "output": args.output}, ensure_ascii=False, indent=2))
    return 0


def handle_prepare_training_data(args: argparse.Namespace) -> int:
    rows = load_jsonl_rows(args.input)
    examples = build_training_examples(rows, task=args.task)
    write_jsonl(args.output, examples)
    print(json.dumps({"examples": len(examples), "task": args.task, "output": args.output}, ensure_ascii=False, indent=2))
    return 0


def handle_prepare_multihop_training_data(args: argparse.Namespace) -> int:
    queries = load_multihop_queries(args.queries)
    needed_hop_ids = {
        hop_id
        for query in queries
        for hop_id in query.supporting_hop_ids
    }
    hops_by_id = {}
    scanned_hops = 0
    for row in iter_jsonl(args.hops):
        scanned_hops += 1
        hop_id = str(row.get("hop_id", ""))
        if hop_id in needed_hop_ids:
            hops_by_id[hop_id] = row

    output = ensure_parent(args.output)
    examples_written = 0
    with output.open("w", encoding="utf-8") as handle:
        for query in queries:
            row = {
                "dialogue_id": query.dialogue_id,
                "query_id": query.query_id,
                "query_type": query.query_type,
                "question": query.question,
                "answer": query.answer,
                "difficulty": query.difficulty,
                "reasoning_focus": list(query.reasoning_focus),
                "supporting_hop_ids": list(query.supporting_hop_ids),
                "supporting_hops": [
                    hops_by_id[hop_id]
                    for hop_id in query.supporting_hop_ids
                    if hop_id in hops_by_id
                ],
                "target_utterance_id": query.target_utterance_id,
                "metadata": dict(query.metadata),
            }
            example = build_training_examples([row], task="multi-hop-kg-response")[0]
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            examples_written += 1

    print(
        json.dumps(
            {
                "examples": examples_written,
                "task": "multi-hop-kg-response",
                "queries": len(queries),
                "scanned_hops": scanned_hops,
                "matched_hops": len(hops_by_id),
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def handle_filter_multihop_training_data(args: argparse.Namespace) -> int:
    type_caps = parse_type_caps(args.type_caps)
    selected, stats = select_balanced_examples(
        iter_jsonl(args.input),
        type_caps=type_caps,
        min_hops=args.min_hops,
        drop_generic_event=not args.keep_generic_events,
        drop_event_hops=args.drop_event_hops,
        drop_weak_indirectness=args.drop_weak_indirectness,
        drop_unknown_relation=args.drop_unknown_relation,
        seed=args.seed,
    )
    output = ensure_parent(args.output)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            if args.rewrite_answers:
                row = rewrite_multihop_training_answer(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    payload = stats.to_dict()
    payload["output"] = args.output
    payload["type_caps"] = type_caps
    payload["drop_generic_event"] = not args.keep_generic_events
    payload["drop_event_hops"] = args.drop_event_hops
    payload["drop_weak_indirectness"] = args.drop_weak_indirectness
    payload["drop_unknown_relation"] = args.drop_unknown_relation
    payload["rewrite_answers"] = args.rewrite_answers
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def handle_generate_qa(args: argparse.Namespace) -> int:
    triples = load_triples(args.triples)
    client = OpenAICompatibleClient.from_environment()
    if client is None:
        raise RuntimeError("generate-qa requires an OpenAI-compatible endpoint in the environment.")
    samples = generate_qa_samples(triples=triples, client=client, sleep_seconds=args.sleep_seconds)
    write_jsonl(args.output, samples)
    print(json.dumps({"samples": len(samples), "output": args.output}, ensure_ascii=False, indent=2))
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


def load_jsonl_rows(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        rows: list[dict] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
