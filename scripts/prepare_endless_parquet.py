#!/usr/bin/env python3
"""Build train/val parquets from Endless Terminals tasks whose images built.

Takes the tasks the Hopper image builder (cs795 ``slurm/echo_build_tasks.sbatch``)
reported as built *and* passing their own initial-state tests, shuffles them with a
fixed seed, holds out ``--val`` for validation, and writes ECHO's prompt format for
the chosen parser (for ``qwen35``: the tokenizer-derived system message plus the
instruction prefix, as ``format_task_prompt`` builds it).

    python scripts/prepare_endless_parquet.py --tasks-root endless-tasks \\
        --build-log runs/echo/build --tokenizer models/hamishivi--Qwen3.5-2B/main \\
        --parser qwen35 --val 100 --out-dir endless-data/pilot
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import datasets

from echo_rl.terminal_agent.prompts import format_task_prompt


def built_tasks(build_log: Path) -> list[str]:
    ok = []
    for f in sorted(build_log.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            init = r.get("initial_tests", "")
            if r.get("status") == "ok" and "passed" in init and "failed" not in init and "error" not in init:
                ok.append(r["task"])
    return sorted(set(ok))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks-root", type=Path, required=True)
    p.add_argument("--build-log", type=Path, required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--parser", default="qwen35", choices=["xml", "hermes", "qwen35"])
    p.add_argument("--val", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-time", type=int, default=120)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    tokenizer = None
    if args.parser != "xml":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    names = built_tasks(args.build_log)
    random.Random(args.seed).shuffle(names)
    if len(names) <= args.val:
        raise SystemExit(f"only {len(names)} usable tasks; need more than --val {args.val}")
    splits = {"val": names[: args.val], "train": names[args.val :]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        rows = []
        for name in ids:
            task_dir = args.tasks_root / name
            instruction = (task_dir / "instruction.md").read_text().strip()
            prompt = format_task_prompt(instruction, args.parser, tokenizer, add_instruction_prefix=args.parser != "xml")
            if isinstance(prompt, str):
                prompt = [{"role": "user", "content": prompt}]
            rows.append({"prompt": prompt, "extra_info": {"task_dir": str(task_dir.resolve()), "max_time": args.max_time,
                                                          "task_id": name}})
        datasets.Dataset.from_list(rows).to_parquet(str(args.out_dir / f"{split}.parquet"))
        (args.out_dir / f"{split}_tasks.txt").write_text("\n".join(ids) + "\n")
        print(f"{split}: {len(rows)} tasks -> {args.out_dir / (split + '.parquet')}")


if __name__ == "__main__":
    main()
