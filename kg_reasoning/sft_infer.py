from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kg_reasoning.sft_eval import (
    apply_kg_ablation,
    build_generation_kwargs,
    load_message_rows,
    normalize_generation_config,
    normalize_response,
    resolve_dtype,
)


def load_inference_row(
    *,
    input_jsonl: str | None,
    row_index: int,
    messages_file: str | None,
) -> dict[str, Any]:
    if messages_file:
        payload = json.loads(Path(messages_file).read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict) and "messages" in payload:
            return payload
        if isinstance(payload, list):
            return {"messages": payload}
        raise ValueError("messages-file must contain either a message list or an object with a 'messages' field.")
    if not input_jsonl:
        raise ValueError("Either --input-jsonl or --messages-file is required.")
    rows = load_message_rows(input_jsonl)
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row-index {row_index} is out of range for {len(rows)} rows.")
    return rows[row_index]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample inference with the SFT adapter")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--messages-file", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--kg-ablation", choices=("none", "blank-evidence"), default="none")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    row = load_inference_row(
        input_jsonl=args.input_jsonl,
        row_index=args.row_index,
        messages_file=args.messages_file,
    )
    messages = list(row["messages"])
    prompt_messages = apply_kg_ablation(messages[:-1], args.kg_ablation)
    reference = str(messages[-1]["content"]) if messages and messages[-1].get("role") == "assistant" else None

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

    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    generation_kwargs = build_generation_kwargs(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    prediction = normalize_response(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    result = {
        "metadata": row.get("metadata", {}),
        "adapter_path": args.adapter_path,
        "kg_ablation": args.kg_ablation,
        "prediction": prediction,
        "reference": normalize_response(reference) if reference is not None else None,
        "prompt": prompt_text,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
