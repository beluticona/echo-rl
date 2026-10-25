"""Apptainer environment adapter for Endless Terminals tasks."""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int


# Hopper's apptainer module exports APPTAINER_BIND="/home, /scratch, /projects,
# /groups", which Apptainer applies to every container whatever its flags. It is
# the "/groups doesn't exist in container" failure when building a task image,
# and it would mount the cluster's /home over the task's /home/user. Endless
# Terminals starts its containers with the inherited environment, so remove
# these before it does (the same fix as analysis/phase0/sandbox.py in cs795).
_BIND_ENV = ("APPTAINER_BIND", "APPTAINER_BINDPATH", "SINGULARITY_BIND", "SINGULARITY_BINDPATH")


def _drop_cluster_binds() -> None:
    for key in _BIND_ENV:
        os.environ.pop(key, None)


class EndlessApptainerEnvironment:
    def __init__(self, task_dir: Path, source_root: Path) -> None:
        self.task_dir = task_dir.expanduser().resolve()
        self.source_root = source_root.expanduser().resolve()
        self._environment = None

    async def setup(self) -> None:
        _drop_cluster_binds()
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        from generator.env import InteractiveContainerEnvironment

        environment_dir = self.task_dir / "environment"
        if not environment_dir.is_dir():
            environment_dir = self.task_dir
        tests_dir = self.task_dir / "tests"
        self._environment = InteractiveContainerEnvironment(
            container_sif_path=environment_dir / "container.sif",
            initial_test_path=environment_dir / "test_initial_state.py",
            final_test_path=(tests_dir / "test_final_state.py") if tests_dir.is_dir() else self.task_dir / "test_final_state.py",
            def_path=environment_dir / "container.def",
            verbose=False,
        )
        ready = await asyncio.to_thread(self._environment.initialize, False)
        if not ready:
            raise RuntimeError(f"Failed to initialize Endless task {self.task_dir}")

    async def exec(self, command: str, timeout: float | None = None) -> ExecResult:
        if self._environment is None:
            raise RuntimeError("Environment not set up. Call setup() first.")
        success, output = await asyncio.to_thread(self._environment.exec, command, timeout)
        return ExecResult(stdout=output, stderr="", return_code=0 if success else 1)

    async def run_verifier(self, timeout: float | None = None) -> tuple[float, str | None]:
        if self._environment is None:
            raise RuntimeError("Environment not set up. Call setup() first.")
        try:
            success, _ = await asyncio.wait_for(
                asyncio.to_thread(self._environment.run_final_tests), timeout=timeout
            )
            return float(success), None
        except asyncio.TimeoutError:
            return 0.0, "verifier_timeout"
        except Exception:
            return 0.0, "verifier_error"

    async def cleanup(self) -> None:
        if self._environment is not None:
            await asyncio.to_thread(self._environment.cleanup)
            self._environment = None


class EndlessApptainerEnvironmentProvider:
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root

    async def prepare_batch(self, batch_environment_data: list[dict[str, Any]], num_generations: int) -> None:
        for prompt_idx, env_data in enumerate(batch_environment_data):
            if not env_data.get("task_dir"):
                raise ValueError(f"prompt {prompt_idx}: Endless environment requires 'task_dir'.")
            env_data["_prompt_idx"] = prompt_idx

    async def create(self, environment_data: dict[str, Any]) -> EndlessApptainerEnvironment:
        return EndlessApptainerEnvironment(Path(environment_data["task_dir"]), self.source_root)

    async def cleanup_batch(self) -> None:
        return None
