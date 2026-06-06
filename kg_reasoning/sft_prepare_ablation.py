from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_reasoning.sft_eval import apply_kg_ablation, load_message_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an ablated SFT dataset by modifying prompt messages.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--kg-ablation", choices=("none", "blank-evidence"), default="blank-evidence")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_message_rows(args.input_jsonl)
    output_rows: list[dict] = []
    for row in rows:
        messages = list(row["messages"])
        if not messages:
            continue
        prompt_messages = apply_kg_ablation(messages[:-1], args.kg_ablation)
        output_rows.append(
            {
                **row,
                "messages": prompt_messages + [messages[-1]],
            }
        )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "kg_ablation": args.kg_ablation,
                "output_jsonl": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
