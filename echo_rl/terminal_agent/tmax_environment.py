"""TMax-15K task environments served by cs795's host-side sandbox server.

The trainer runs inside the SkyRL image; the task containers run on the host, in
the same Slurm job, behind ``analysis.phase0.sandbox_server`` (cs795). This adapter
is the generator's view of one episode's sandbox, over HTTP on localhost, so no
container is ever started from inside another. Exit codes are the commands' real
ones, so the generator's observations end in the true ``(exit_code=N)``.

Selected with ``generator.environment_backend=tmax_remote`` and
``generator.tmax_sandbox_url``; each dataset row's ``extra_info`` carries
``task_dir`` and ``image``.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int


def _post(url: str, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"sandbox server {path}: {e.read().decode(errors='replace')[:500]}") from e


class TMaxRemoteEnvironment:
    def __init__(self, url: str, task_dir: str, image: str) -> None:
        self.url, self.task_dir, self.image = url, task_dir, image
        self.sid: str | None = None

    async def setup(self) -> None:
        # Starting a container converts nothing (images are pre-converted) but can
        # take tens of seconds without squashfuse; allow for a queue of starts.
        out = await asyncio.to_thread(_post, self.url, "/start",
                                      {"image": self.image, "task_dir": self.task_dir}, 900)
        self.sid = out["id"]

    async def exec(self, command: str, timeout: float | None = None) -> ExecResult:
        if self.sid is None:
            raise RuntimeError("Environment not set up. Call setup() first.")
        out = await asyncio.to_thread(_post, self.url, "/exec",
                                      {"id": self.sid, "command": command, "timeout": timeout},
                                      (timeout or 120) + 120)
        # The generator shows stdout only; stderr is part of what a terminal shows.
        text = out["stdout"] + (("\n" if out["stdout"] and out["stderr"] else "") + out["stderr"])
        return ExecResult(stdout=text, stderr="", return_code=int(out["exit_code"]))

    async def run_verifier(self, timeout: float | None = None) -> tuple[float, str | None]:
        if self.sid is None:
            raise RuntimeError("Environment not set up. Call setup() first.")
        try:
            out = await asyncio.to_thread(_post, self.url, "/verify", {"id": self.sid}, (timeout or 600) + 120)
            return float(out["reward"]), None
        except Exception:
            return 0.0, "verifier_error"

    async def cleanup(self) -> None:
        if self.sid is not None:
            try:
                await asyncio.to_thread(_post, self.url, "/close", {"id": self.sid}, 120)
            finally:
                self.sid = None


class TMaxRemoteEnvironmentProvider:
    def __init__(self, url: str) -> None:
        self.url = url

    async def prepare_batch(self, batch_environment_data: list[dict[str, Any]], num_generations: int) -> None:
        for prompt_idx, env_data in enumerate(batch_environment_data):
            if not env_data.get("task_dir") or not env_data.get("image"):
                raise ValueError(f"prompt {prompt_idx}: tmax_remote needs 'task_dir' and 'image' in extra_info.")
            env_data["_prompt_idx"] = prompt_idx

    async def create(self, environment_data: dict[str, Any]) -> TMaxRemoteEnvironment:
        return TMaxRemoteEnvironment(self.url, environment_data["task_dir"], environment_data["image"])

    async def cleanup_batch(self) -> None:
        return None
