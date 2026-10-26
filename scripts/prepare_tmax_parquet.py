#!/usr/bin/env python3
"""Build train/val parquets for the ECHO pilot from TMax-15K tasks (cs795).

Uses the seed-0 sample order that cs795's Phase 0 uses everywhere: tasks
[0, 500) are the evaluation rollouts (runs/phase0/r500), so training takes
[--start, --end) = [500, 1500) by default, all of which have converted images.
Tasks whose image is not converted are dropped. Prompts use ECHO's format for the
chosen parser; ``extra_info`` carries what the tmax_remote backend needs.

    python scripts/prepare_tmax_parquet.py --project-root /scratch/$USER/cs795-course \\
        --tokenizer models/hamishivi--Qwen3.5-2B/main --val 100 --out-dir tmax-data/pilot
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import datasets

from echo_rl.terminal_agent.prompts import format_task_prompt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--index", default="datasets/tmax-15k-open-instruct/tasks_index.jsonl")
    p.add_argument("--sif-dir", default="images/tasks")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--parser", default="qwen35", choices=["xml", "hermes", "qwen35"])
    p.add_argument("--start", type=int, default=500)
    p.add_argument("--end", type=int, default=1500)
    p.add_argument("--val", type=int, default=100)
    p.add_argument("--seed", type=int, default=0, help="the Phase 0 sample seed")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prefix", default=None, help="instruction prefix variant, e.g. qwen35_single (default: the parser's)")
    p.add_argument("--train-list", default=None, help="task ids for train (overrides --start/--end/--val)")
    p.add_argument("--val-list", default=None, help="task ids for val, with --train-list")
    p.add_argument("--task-list", default=None,
                   help="all task ids: drop long prompts, shuffle, then --val for val, --max-train for train, "
                        "the rest listed in heldout_tasks.txt")
    p.add_argument("--max-train", type=int, default=None, help="with --task-list: cap on train tasks")
    p.add_argument("--max-prompt-tokens", type=int, default=None,
                   help="drop tasks whose templated prompt is longer (match trainer.max_prompt_length), "
                        "before splitting, so the trainer's own filter drops nothing")
    args = p.parse_args()

    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from analysis.phase0.sandbox import image_sif_path
    from analysis.phase0.tasks import load_index, load_instruction

    sif_dir = str(root / args.sif_dir)
    tokenizer = None
    if args.parser != "xml":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def prompt_of(e: dict) -> list:
        prompt = format_task_prompt(load_instruction(e), args.parser, tokenizer,
                                    add_instruction_prefix=args.parser != "xml", prefix_name=args.prefix)
        return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt

    def short_enough(entries: list[dict]) -> list[dict]:
        if not args.max_prompt_tokens or tokenizer is None:
            return entries
        # the same count as the trainer's filter (echo_rl.terminal_agent.dataset)
        kept = [e for e in entries if len(tokenizer.apply_chat_template(
            prompt_of(e), tokenize=True, add_generation_prompt=True, return_dict=False)) <= args.max_prompt_tokens]
        print(f"prompts <= {args.max_prompt_tokens} tokens: {len(kept)} of {len(entries)}")
        return kept

    heldout = None
    if args.task_list:
        entries = load_index(str(root / args.index), task_list=args.task_list)
        usable = short_enough([e for e in entries if os.path.isfile(image_sif_path(e["image"], sif_dir))])
        print(f"listed: {len(entries)}, usable: {len(usable)}")
        random.Random(args.split_seed).shuffle(usable)
        end = args.val + (args.max_train if args.max_train is not None else len(usable))
        splits = {"val": usable[: args.val], "train": usable[args.val : end]}
        heldout = usable[end:]
    elif args.train_list:
        splits = {}
        for split, lst in (("train", args.train_list), ("val", args.val_list)):
            if not lst:
                continue
            entries = load_index(str(root / args.index), task_list=lst)
            splits[split] = short_enough([e for e in entries if os.path.isfile(image_sif_path(e["image"], sif_dir))])
            print(f"{split}: {len(entries)} listed, with a converted image: {len(splits[split])}")
        random.Random(args.split_seed).shuffle(splits["train"])
    else:
        entries = load_index(str(root / args.index), limit=args.end, seed=args.seed)[args.start : args.end]
        usable = short_enough([e for e in entries if os.path.isfile(image_sif_path(e["image"], sif_dir))])
        print(f"tasks [{args.start}, {args.end}): {len(entries)}, usable: {len(usable)}")
        random.Random(args.split_seed).shuffle(usable)
        splits = {"val": usable[: args.val], "train": usable[args.val :]}

    out = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for split, rows_in in splits.items():
        rows = []
        for e in rows_in:
            rows.append({"prompt": prompt_of(e), "extra_info": {"task_dir": e["task_dir"], "image": e["image"],
                                                          "task_id": e["task_id"], "max_time": 120}})
        datasets.Dataset.from_list(rows).to_parquet(str(out / f"{split}.parquet"))
        (out / f"{split}_tasks.txt").write_text("\n".join(e["task_id"] for e in rows_in) + "\n")
        print(f"{split}: {len(rows)} -> {out / (split + '.parquet')}")
    if heldout is not None:
        (out / "heldout_tasks.txt").write_text("".join(e["task_id"] + "\n" for e in heldout))
        print(f"held out (not trained on): {len(heldout)} -> {out / 'heldout_tasks.txt'}")


if __name__ == "__main__":
    main()
