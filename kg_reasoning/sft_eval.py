from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def load_message_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def split_rows(
    rows: list[dict[str, Any]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if eval_ratio <= 0:
        return shuffled, []
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    if eval_size >= len(shuffled):
        eval_size = max(0, len(shuffled) - 1)
    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    return train_rows, eval_rows


def normalize_response(text: str) -> str:
    return " ".join(text.strip().split())


def blank_kg_sections(text: str) -> str:
    text = re.sub(
        r"(\[Retrieved Events\]\n)(.*?)(\n\n\[Grounded Triples\])",
        r"\1\3",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"(\[Grounded Triples\]\n)(.*?)(\n\n\[Instruction\])",
        r"\1\3",
        text,
        flags=re.S,
    )
    return text


def apply_kg_ablation(messages: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "none":
        return [dict(message) for message in messages]
    if mode != "blank-evidence":
        raise ValueError(f"Unsupported KG ablation mode: {mode}")
    ablated: list[dict[str, Any]] = []
    for message in messages:
        cloned = dict(message)
        if cloned.get("role") == "user":
            cloned["content"] = blank_kg_sections(str(cloned.get("content", "")))
        ablated.append(cloned)
    return ablated


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_response(prediction).split()
    ref_tokens = normalize_response(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA SFT adapter on held-out chat examples")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--max-eval-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--kg-ablation", choices=("none", "blank-evidence"), default="none")
    return parser.parse_args(argv)


def resolve_dtype(name: str) -> Any:
    import torch

    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_generation_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
    eos_token_id: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if temperature > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    else:
        kwargs["do_sample"] = False
    return kwargs


def normalize_generation_config(
    config: Any,
    *,
    temperature: float,
    top_p: float,
) -> Any:
    if config is None:
        return None
    config.do_sample = temperature > 0
    if temperature > 0:
        config.temperature = temperature
        config.top_p = top_p
        return config
    for attribute in ("temperature", "top_p", "top_k", "min_p", "typical_p", "epsilon_cutoff", "eta_cutoff"):
        if hasattr(config, attribute):
            setattr(config, attribute, None)
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_message_rows(args.train_file)
    indexed_rows = []
    for index, row in enumerate(rows):
        enriched = dict(row)
        enriched["_source_index"] = index
        indexed_rows.append(enriched)
    _, eval_rows = split_rows(indexed_rows, eval_ratio=args.eval_ratio, seed=args.seed)
    eval_rows = eval_rows[: args.max_eval_samples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=resolve_dtype(args.dtype),
        local_files_only=args.local_files_only,
        trust_remote_code=True,
        device_map="auto",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.generation_config = normalize_generation_config(
        getattr(model, "generation_config", None),
        temperature=args.temperature,
        top_p=args.top_p,
    )
    model.eval()

    output_rows: list[dict[str, Any]] = []
    exact_matches = 0
    token_f1_sum = 0.0
    generation_kwargs = build_generation_kwargs(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for row in eval_rows:
        prompt_messages = apply_kg_ablation(row["messages"][:-1], args.kg_ablation)
        gold = str(row["messages"][-1]["content"])
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, **generation_kwargs)
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        prediction = normalize_response(prediction)
        gold_normalized = normalize_response(gold)
        is_exact = prediction == gold_normalized
        exact_matches += int(is_exact)
        sample_token_f1 = token_f1(prediction, gold_normalized)
        token_f1_sum += sample_token_f1
        output_rows.append(
            {
                "source_index": row.get("_source_index"),
                "metadata": row.get("metadata", {}),
                "prompt": prompt_text,
                "prediction": prediction,
                "reference": gold_normalized,
                "exact_match": is_exact,
                "token_f1": sample_token_f1,
            }
        )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "adapter_path": args.adapter_path,
        "eval_examples": len(output_rows),
        "exact_match": (exact_matches / len(output_rows)) if output_rows else 0.0,
        "avg_token_f1": (token_f1_sum / len(output_rows)) if output_rows else 0.0,
        "kg_ablation": args.kg_ablation,
        "output_jsonl": str(output_path),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
