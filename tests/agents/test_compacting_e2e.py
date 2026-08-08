"""End-to-end rollout tests for CompactingAgent.

These exercise the full stack — the agent's run loop, the real
``LocalEnvironment`` (real ``subprocess.run``), trajectory save/reload,
and the agent factory wiring. The model is deterministic so each test
is hermetic, fast, and cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

from minisweagent.agents import get_agent
from minisweagent.agents.compacting import CompactingAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

BASE_CONFIG: dict = {
    "agent_class": "compacting",
    "system_template": (
        "You are a helpful assistant that can interact with a computer.\n"
        "Emit one bash command per turn in ```mswea_bash_command blocks."
    ),
    "instance_template": "Task: {{ task }}",
    # Give tests plenty of room on cost/step limits — the DeterministicModel
    # charges $1/call and our episodes make several calls including summary calls.
    "cost_limit": 0,
    "step_limit": 0,
    "summary_system_template": "Hand off this task. Keep <= {{ summary_max_tokens }} tokens.",
    "summary_user_template": "Produce the handoff summary.",
    "resumption_template": (
        "<context_compacted n=\"{{ n_compactions }}\" id=\"{{ compaction_id }}\">\n"
        "{{ summary }}\n</context_compacted>"
    ),
    "summary_max_tokens": 512,
}


def run_episode(
    *,
    outputs: list[dict],
    agent_config: dict,
    task: str,
    tmp_path: Path,
) -> tuple[CompactingAgent, dict]:
    """Drive one full end-to-end episode and reload the saved trajectory JSON.

    - Model: ``DeterministicModel`` with the scripted ``outputs``.
    - Env:   real ``LocalEnvironment`` (real bash).
    - Agent: resolved via :func:`minisweagent.agents.get_agent` — same code
      path the ``mini`` CLI takes.

    Returns the live agent plus the trajectory dict as parsed from the
    saved JSON file, so the test exercises both the in-memory state and
    the persisted format.
    """
    config = {**BASE_CONFIG, **agent_config}
    config["output_path"] = tmp_path / "traj.json"

    model = DeterministicModel(outputs=outputs)
    env = LocalEnvironment()
    agent = get_agent(model, env, config, default_type="")

    assert isinstance(agent, CompactingAgent), (
        "factory should have resolved 'compacting' to CompactingAgent"
    )

    info = agent.run(task)
    assert info["exit_status"] == "Submitted", info

    saved = json.loads(config["output_path"].read_text())
    return agent, saved


def _submit_output() -> dict:
    return make_output(
        "Submitting.",
        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'task done'"}],
    )


def test_e2e_episode_with_real_bash(tmp_path):
    """Full rollout: real bash, budget triggers mid-episode, trajectory saved and reloaded.

    Scripts a 5-turn episode where steps 1-3 produce long observations
    (real ``echo`` of 4000 chars) that push the interior over the token budget.
    The agent should fire exactly one compaction, then submit.
    """
    big_cmd = make_output("Exploring.", [{"command": "python3 -c \"print('x' * 4000)\""}])
    summary_response = make_output(
        "## Goal\nwrite the file\n## Files touched\nnone yet\n## Next action\nsubmit", []
    )

    outputs = [big_cmd, big_cmd, big_cmd, summary_response, _submit_output()]
    agent, saved = run_episode(
        outputs=outputs,
        agent_config={
            "token_budget": 200,
            "min_preserve_head": 2,
            "min_preserve_tail": 2,
            "min_compact_size": 2,
            "min_compact_tokens": 50,
        },
        task="compact your context when big outputs arrive",
        tmp_path=tmp_path,
    )

    assert saved["trajectory_format"] == "mini-swe-agent-compact-1"
    # Exactly one compaction event, and it carries the summary call metadata.
    assert len(saved["compactions"]) == 1
    c = saved["compactions"][0]
    assert c["kind"] == "compact"
    assert c["summary_call"] is not None
    assert c["summary_call"]["completion_tokens"] > 0

    # Context-management bookkeeping.
    assert saved["info"]["context_management"]["strategy"] == "compact"
    assert saved["info"]["context_management"]["n_compactions"] == 1

    # Real bash actually produced "x" * 4000 in some observations during the run,
    # but after compaction only the preserved head/tail keeps any raw output.
    all_contents = "\n".join(str(m.get("content", "")) for m in saved["messages"])
    assert "context_compacted" in all_contents
    # The exit message at the end is the submission.
    assert saved["messages"][-1]["role"] == "exit"
    assert "task done" in saved["info"]["submission"]


def test_e2e_no_compaction_path(tmp_path):
    """Short episode: budget is not hit, agent submits with 0 compactions."""
    short_cmd = make_output("ls it", [{"command": "echo 'tiny'"}])
    outputs = [short_cmd, _submit_output()]
    agent, saved = run_episode(
        outputs=outputs,
        agent_config={"token_budget": 10**9},
        task="small job",
        tmp_path=tmp_path,
    )
    assert saved["compactions"] == []
    # The real bash echo output survives unmodified on the trajectory.
    assert any("tiny" in str(m.get("content", "")) for m in saved["messages"])


def test_e2e_yaml_config_runs_real_episode(tmp_path):
    """Load the shipped YAML, instantiate via the real factory, run one episode.

    Proves the YAML is not only syntactically valid but actually wires up a
    working agent end-to-end.
    """
    import yaml

    cfg_path = Path("src/minisweagent/config") / "compact.yaml"
    with cfg_path.open() as f:
        agent_cfg = yaml.safe_load(f)["agent"]

    # Disable cost limit so the test doesn't trip on DeterministicModel's $1/call.
    agent_cfg["cost_limit"] = 0
    agent_cfg["output_path"] = tmp_path / "traj.json"
    # Raise budget so the short scripted episode doesn't fire compaction.
    agent_cfg["token_budget"] = 10**9

    outputs = [
        make_output("short", [{"command": "echo 'hi'"}]),
        _submit_output(),
    ]
    model = DeterministicModel(outputs=outputs)
    env = LocalEnvironment()
    agent = get_agent(model, env, agent_cfg, default_type="")
    assert isinstance(agent, CompactingAgent)
    info = agent.run("smoke test")
    assert info["exit_status"] == "Submitted"

    saved = json.loads(agent_cfg["output_path"].read_text())
    assert saved["info"]["context_management"]["strategy"] == "compact"
