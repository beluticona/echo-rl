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
    args = p.parse_args()

    root = args.project_root.resolve()
    sys.path.insert(0, str(root))
    from analysis.phase0.sandbox import image_sif_path
    from analysis.phase0.tasks import load_index, load_instruction

    entries = load_index(str(root / args.index), limit=args.end, seed=args.seed)[args.start : args.end]
    sif_dir = str(root / args.sif_dir)
    usable = [e for e in entries if os.path.isfile(image_sif_path(e["image"], sif_dir))]
    print(f"tasks [{args.start}, {args.end}): {len(entries)}, with a converted image: {len(usable)}")
    random.Random(args.split_seed).shuffle(usable)
    splits = {"val": usable[: args.val], "train": usable[args.val :]}

    tokenizer = None
    if args.parser != "xml":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    out = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for split, rows_in in splits.items():
        rows = []
        for e in rows_in:
            prompt = format_task_prompt(load_instruction(e), args.parser, tokenizer,
                                        add_instruction_prefix=args.parser != "xml")
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            rows.append({"prompt": prompt, "extra_info": {"task_dir": e["task_dir"], "image": e["image"],
                                                          "task_id": e["task_id"], "max_time": 120}})
        datasets.Dataset.from_list(rows).to_parquet(str(out / f"{split}.parquet"))
        (out / f"{split}_tasks.txt").write_text("\n".join(e["task_id"] for e in rows_in) + "\n")
        print(f"{split}: {len(rows)} -> {out / (split + '.parquet')}")


if __name__ == "__main__":
    main()
