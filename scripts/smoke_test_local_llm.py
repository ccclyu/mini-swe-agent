#!/usr/bin/env python3
"""Smoke test for CompactingAgent / FoldingAgent against a locally hosted OpenAI-compatible model.

Usage:
    # start the server first, e.g. in a separate tmux pane:
    #   bash scripts/launch_vllm.sh
    #
    # then:
    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    OPENAI_API_KEY=dummy \
    PYTHONPATH=src \
    python scripts/smoke_test_local_llm.py \
        --model openai/qwen3-4b-instruct-2507 \
        --agent both \
        --out-dir /tmp/ctxmgmt_smoke

The script drives a real, tiny task (``ls`` + submit) through one or both
agents, validates that a trajectory was saved with the expected shape, and
prints a summary. No API cost — local inference only — but it uses the full
mini-swe-agent stack (litellm → LitellmModel → /v1/chat/completions → vLLM),
so any wiring bug surfaces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from minisweagent.agents.compacting import CompactingAgent
from minisweagent.agents.folding import FoldingAgent
from minisweagent.environments.local import LocalEnvironment
# Text-based model: parses bash blocks from content. Avoids vLLM's requirement
# of ``--enable-auto-tool-choice --tool-call-parser hermes`` that the tool-call
# variant would trigger. Both agents work equivalently with either.
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel as LitellmModel


COMPACT_SYSTEM = """You are a helpful assistant that can interact with a computer.

Your response MUST contain exactly one bash command in a fenced code block
with the language ``mswea_bash_command``. Put a short THOUGHT line before
it. When the task is complete, run:

    ```mswea_bash_command
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
    ```

(alone, no other command on the same turn)."""


FOLD_SYSTEM = """You are a helpful assistant that can interact with a computer.

You have access to bash (one command per turn, in a ```mswea_bash_command
block) and to these structured functions:

    <function=branch>
    <parameter=description>short 3-5 word name</parameter>
    <parameter=prompt>sub-task description</parameter>
    </function>

    <function=return>
    <parameter=message>summary for MAIN</parameter>
    </function>

Branch to delegate a scoped sub-task; return to finish the branch. You
cannot submit/finish from inside a branch. Submit the overall task with:

    ```mswea_bash_command
    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
    ```

Emit exactly ONE action (either a <function=...> call or a bash block) per turn.
"""


INSTANCE_TEMPLATE = """Please solve this task: {{ task }}

When done, submit with:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```
"""


def build_agent(
    *,
    model_name: str,
    agent_kind: str,
    out_path: Path,
    token_budget: int = 10**9,
    step_limit: int = 12,
    min_compact_tokens: int = 1024,
):
    """Construct the requested agent wired to a local OpenAI-shaped model.

    ``token_budget`` defaults to "effectively infinite" so healthy compact
    rollouts just run end-to-end. Pass a small value (e.g. 500-2000) plus
    ``min_compact_tokens`` low enough that the interior exceeds it to
    deliberately trigger a real compaction event.
    """
    # LitellmModel reads OPENAI_BASE_URL / OPENAI_API_KEY from the env. We
    # pass cost_tracking=ignore_errors because our local served model isn't
    # in litellm's pricing table.
    model = LitellmModel(
        model_name=model_name,
        cost_tracking="ignore_errors",
        model_kwargs={"temperature": 0.1, "max_tokens": 512},
    )
    env = LocalEnvironment()

    if agent_kind == "compact":
        return CompactingAgent(
            model=model, env=env,
            system_template=COMPACT_SYSTEM,
            instance_template=INSTANCE_TEMPLATE,
            cost_limit=0,   # local model → no cost, disable trip-wire
            step_limit=step_limit,
            output_path=out_path,
            token_budget=token_budget,
            min_compact_tokens=min_compact_tokens,
            # Keep head/tail preservation tight so small trajectories still compact.
            min_preserve_tail=2,
            min_compact_size=2,
        )
    return FoldingAgent(
        model=model, env=env,
        system_template=FOLD_SYSTEM,
        instance_template=INSTANCE_TEMPLATE,
        cost_limit=0,
        step_limit=step_limit,
        output_path=out_path,
        # Keep branch turn budget short — this is a smoke test.
        max_branch_turns=6,
        max_branches=2,
    )


def run_agent_kind(
    *,
    agent_kind: str,
    model_name: str,
    task: str,
    out_dir: Path,
    token_budget: int = 10**9,
    min_compact_tokens: int = 1024,
    step_limit: int = 12,
) -> dict:
    """Run one episode; return a status dict the caller can assert on."""
    out_path = out_dir / f"{agent_kind}.traj.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agent = build_agent(
        model_name=model_name,
        agent_kind=agent_kind,
        out_path=out_path,
        token_budget=token_budget,
        step_limit=step_limit,
        min_compact_tokens=min_compact_tokens,
    )

    t0 = time.time()
    print(f"\n=== [{agent_kind}] running episode ===", flush=True)
    info = agent.run(task)
    elapsed = time.time() - t0

    # Reload what was saved — the e2e guarantee is that the on-disk traj
    # matches the live agent state.
    saved = json.loads(out_path.read_text())

    # Minimal structural validation — details are covered by the pytest
    # suite; here we just want to prove the local-model path works.
    expected_format = f"mini-swe-agent-{agent_kind}-1"
    assert saved["trajectory_format"] == expected_format, saved["trajectory_format"]
    assert saved["info"]["context_management"]["strategy"] == agent_kind

    result = {
        "agent": agent_kind,
        "elapsed_s": round(elapsed, 2),
        "exit_status": info.get("exit_status", "<unknown>"),
        "submission": info.get("submission", ""),
        "n_api_calls": saved["info"]["model_stats"]["api_calls"],
        "traj_path": str(out_path),
    }
    if agent_kind == "compact":
        result["n_compactions"] = saved["info"]["context_management"]["n_compactions"]
    else:
        result["n_agents"] = len(saved.get("agents", {}))
        result["n_branches"] = saved["info"]["context_management"]["n_branches"]
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="litellm model string, e.g. openai/qwen3-4b-instruct-2507")
    p.add_argument("--agent", choices=["compact", "fold", "both"], default="both")
    p.add_argument(
        "--task",
        default="List the files in the current directory, then submit.",
        help="Task statement passed to the agent.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/ctxmgmt_smoke"))
    p.add_argument(
        "--token-budget", type=int, default=10**9,
        help="Compact when context exceeds this many tokens. Default = effectively off.",
    )
    p.add_argument(
        "--min-compact-tokens", type=int, default=1024,
        help="Don't compact if the compactable slice is smaller than this; lower to force "
             "compaction on small trajectories.",
    )
    p.add_argument("--step-limit", type=int, default=12)
    args = p.parse_args()

    # Surface OPENAI_BASE_URL to make routing obvious in the log.
    print(f"OPENAI_BASE_URL = {os.environ.get('OPENAI_BASE_URL', '<unset>')}")
    print(f"model           = {args.model}")
    print(f"task            = {args.task}")
    print(f"out-dir         = {args.out_dir}")

    kinds = ["compact", "fold"] if args.agent == "both" else [args.agent]
    results = []
    for kind in kinds:
        try:
            results.append(run_agent_kind(
                agent_kind=kind, model_name=args.model, task=args.task, out_dir=args.out_dir,
                token_budget=args.token_budget,
                min_compact_tokens=args.min_compact_tokens,
                step_limit=args.step_limit,
            ))
        except Exception as e:
            # Don't let a fold flake hide a compact pass or vice versa —
            # record and carry on.
            print(f"[{kind}] FAILED: {type(e).__name__}: {e}")
            results.append({"agent": kind, "error": f"{type(e).__name__}: {e}"})

    print("\n=== results ===")
    for r in results:
        print(json.dumps(r, indent=2))

    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
