#!/usr/bin/env python3
"""Create a one-row ECHO parquet from an Endless Terminals task directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import datasets

from echo_rl.terminal_agent.prompts import XML_SYSTEM_PROMPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    candidates = [args.tasks_root / args.task] if args.task else sorted(args.tasks_root.glob("task_*"))
    def runnable(path: Path) -> bool:
        return all(
            candidate.is_file()
            for candidate in (
                path / "instruction.md",
                path / "environment" / "test_initial_state.py",
                path / "tests" / "test_final_state.py",
                path / "environment" / "container.sif",
            )
        )

    task_dir = next((path for path in candidates if runnable(path)), None)
    if task_dir is None:
        raise SystemExit(f"No runnable converted Endless task found below {args.tasks_root}")

    instruction = (task_dir / "instruction.md").read_text().strip()

    row = {
        "prompt": [
            {"role": "system", "content": XML_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        "extra_info": {"task_dir": str(task_dir), "max_time": 120},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list([row]).to_parquet(str(args.output))
    print(f"Wrote {args.output} from {task_dir}")


if __name__ == "__main__":
    main()
