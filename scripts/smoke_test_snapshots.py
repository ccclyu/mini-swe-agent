#!/usr/bin/env python3
"""Like smoke_test_local_llm.py but snapshots self.messages at every step boundary.

Writes ``snapshots.jsonl`` with one line per event:
    {event: "before_step"|"after_compact"|"after_step"|"exit",
     n_calls, n_compactions, messages: [...] }

Also still saves the final trajectory JSON (same contract as the smoke test).
The point is to let you see the trajectory *evolve* across compactions —
the default saved trajectory only has the end state.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

from minisweagent.agents.compacting import CompactingAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel as LitellmModel


COMPACT_SYSTEM = """You are a helpful assistant that can interact with a computer.

Your response MUST contain exactly one bash command in a fenced code block
with the language ``mswea_bash_command``. Put a short THOUGHT line before
it. When the task is complete, run:

    ```mswea_bash_command
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
    ```

(alone, no other command on the same turn)."""


INSTANCE_TEMPLATE = """Please solve this task: {{ task }}

When done, submit with:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""


class SnapshottingAgent(CompactingAgent):
    """Extends CompactingAgent with per-step + per-compaction snapshots.

    We hook:
    - ``step``         → snapshot *before* and *after* each step
    - ``_compact_slice`` → snapshot *after* each compaction (captures the
      transition the default trajectory JSON hides)
    """

    def __init__(self, *args, snapshot_path: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot_path = Path(snapshot_path)
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any prior file — smoke tests are one-shot.
        self.snapshot_path.write_text("")
        self._snapshot("init")

    def _snapshot(self, event: str, extra: dict | None = None) -> None:
        record = {
            "event": event,
            "n_calls": self.n_calls,
            "n_compactions": len(self.compactions),
            "cost": self.cost,
            "ts": time.time(),
            "messages": copy.deepcopy(self.messages),
        }
        if extra:
            record.update(extra)
        with self.snapshot_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def step(self) -> list[dict]:
        n_compactions_before = len(self.compactions)
        self._snapshot("before_step")
        result = super().step()
        # If the pre-query compaction fired inside _maybe_compact_on_budget,
        # the `step` callback below records it; the after_step snapshot then
        # captures the final state of the step.
        self._snapshot("after_step", {"n_new_compactions": len(self.compactions) - n_compactions_before})
        return result

    def _compact_slice(self, lo, hi) -> None:
        # Snapshot pre-compaction so readers can diff.
        self._snapshot("before_compact", {"slice": [lo, hi]})
        super()._compact_slice(lo, hi)
        # Post-compaction snapshot shows the replacement user message.
        self._snapshot(
            "after_compact",
            {"compaction_id": self.compactions[-1]["compaction_id"],
             "replaced_range": self.compactions[-1]["replaced_range"],
             "replaced_token_count": self.compactions[-1]["replaced_token_count"],
             "summary_token_count": self.compactions[-1]["summary_token_count"]},
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/compact_snap"))
    p.add_argument("--token-budget", type=int, default=800)
    p.add_argument("--min-compact-tokens", type=int, default=200)
    p.add_argument("--step-limit", type=int, default=20)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = args.out_dir / "final.traj.json"
    snap_path = args.out_dir / "snapshots.jsonl"

    model = LitellmModel(
        model_name=args.model,
        cost_tracking="ignore_errors",
        model_kwargs={"temperature": 0.1, "max_tokens": 512},
    )
    env = LocalEnvironment()
    agent = SnapshottingAgent(
        model=model, env=env,
        snapshot_path=snap_path,
        system_template=COMPACT_SYSTEM,
        instance_template=INSTANCE_TEMPLATE,
        cost_limit=0,
        step_limit=args.step_limit,
        output_path=traj_path,
        token_budget=args.token_budget,
        min_compact_tokens=args.min_compact_tokens,
        min_preserve_head=2,
        min_preserve_tail=2,
        min_compact_size=2,
    )
    print(f"OPENAI_BASE_URL = {os.environ.get('OPENAI_BASE_URL', '<unset>')}")
    print(f"model           = {args.model}")
    print(f"token_budget    = {args.token_budget}")
    print(f"task            = {args.task}")
    print(f"snapshots       = {snap_path}")
    print(f"final traj      = {traj_path}")

    info = agent.run(args.task)

    # One final snapshot so the exit state is easy to diff against the last step.
    agent._snapshot("exit", {"exit_status": info.get("exit_status", "?")})

    print("\n=== final summary ===")
    print(json.dumps({
        "exit_status": info.get("exit_status"),
        "n_api_calls": agent.n_calls,
        "n_compactions": len(agent.compactions),
        "snapshot_lines": snap_path.read_text().count("\n"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
