"""End-to-end rollout tests for FoldingAgent.

These exercise the full stack — the agent's run loop, the real
``LocalEnvironment`` (real ``subprocess.run``), trajectory save/reload,
and the agent factory wiring. The model is deterministic so each test
is hermetic, fast, and cheap.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from minisweagent.agents import get_agent
from minisweagent.agents.folding import FoldingAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

BASE_CONFIG: dict = {
    "agent_class": "folding",
    "system_template": (
        "You are a helpful assistant that can interact with a computer.\n"
        "Emit one bash command per turn in ```mswea_bash_command blocks."
    ),
    "instance_template": "Task: {{ task }}",
    # DeterministicModel charges $1/call; disable the trip-wire.
    "cost_limit": 0,
    "step_limit": 0,
}


def run_episode(
    *,
    outputs: list[dict],
    agent_config: dict,
    task: str,
    tmp_path: Path,
) -> tuple[FoldingAgent, dict]:
    """Drive one full end-to-end episode and reload the saved trajectory JSON."""
    config = {**BASE_CONFIG, **agent_config}
    config["output_path"] = tmp_path / "traj.json"

    model = DeterministicModel(outputs=outputs)
    env = LocalEnvironment()
    agent = get_agent(model, env, config, default_type="")

    assert isinstance(agent, FoldingAgent), (
        "factory should have resolved 'folding' to FoldingAgent"
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


def _branch_open_output(description: str, prompt: str) -> dict:
    return make_output(
        "Let me delegate this.\n"
        f"<function=branch>\n"
        f"<parameter=description>{description}</parameter>\n"
        f"<parameter=prompt>{prompt}</parameter>\n"
        "</function>",
        [],
    )


def _branch_return_output(message: str) -> dict:
    return make_output(
        "Subtask done.\n"
        "<function=return>\n"
        f"<parameter=message>{message}</parameter>\n"
        "</function>",
        [],
    )


def test_e2e_branch_and_return_with_real_bash(tmp_path):
    """Full fold rollout: main spawns a branch, branch runs real bash, returns, main submits.

    Asserts that:
    - Branch is isolated (its bash output doesn't bleed into main's messages)
    - Return message lands on main as a user observation
    - Trajectory JSON exposes per-sub-agent message lists
    - fold_events log both branch_open and branch_return
    """
    outputs = [
        _branch_open_output("run reproducer", "run the failing test"),     # main turn 1
        make_output("Running.", [{"command": "echo 'SECRET_BRANCH_OUTPUT'"}]),  # branch turn 1 (real bash)
        _branch_return_output("Failed with ValueError on line 42"),         # branch turn 2 → return
        _submit_output(),                                                    # main turn 2 → submit
    ]
    agent, saved = run_episode(
        outputs=outputs,
        agent_config={},
        task="fix the bug",
        tmp_path=tmp_path,
    )

    # Isolation: the branch's bash stdout never reaches main.
    main_msgs_text = "\n".join(str(m.get("content", "")) for m in saved["agents"]["main"])
    assert "SECRET_BRANCH_OUTPUT" not in main_msgs_text

    # The branch actually ran real bash — the secret string shows up in the branch's history.
    branch_name = next(n for n in saved["agents"] if n.startswith("#"))
    branch_msgs_text = "\n".join(str(m.get("content", "")) for m in saved["agents"][branch_name])
    assert "SECRET_BRANCH_OUTPUT" in branch_msgs_text

    # Return observation landed on main verbatim through the branch_return_template.
    assert "Failed with ValueError on line 42" in main_msgs_text
    assert re.search(r"#0-run_reproducer.*finished its task", main_msgs_text)

    # fold_events log the branch lifecycle end-to-end.
    kinds = [e["kind"] for e in saved["fold_events"]]
    assert kinds == ["branch_open", "branch_return"]

    # Bookkeeping.
    assert saved["trajectory_format"] == "mini-swe-agent-fold-1"
    assert saved["info"]["context_management"]["strategy"] == "fold"
    assert saved["info"]["context_management"]["n_branches"] == 1
    assert saved["info"]["context_management"]["open_branches"] == []


def test_e2e_branch_cannot_submit(tmp_path):
    """Branch tries to submit via real bash; env raises Submitted, agent catches + corrects."""
    outputs = [
        _branch_open_output("probe", "look around"),
        # Branch attempts the submission sentinel via real bash. The env would
        # normally raise Submitted — the folding step must intercept and
        # redirect the branch to use `return`.
        make_output("finishing early", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'"}]),
        _branch_return_output("ok, returning properly this time"),
        _submit_output(),
    ]
    agent, saved = run_episode(
        outputs=outputs,
        agent_config={},
        task="verify branch cannot end episode",
        tmp_path=tmp_path,
    )

    # If the branch had ended the episode, submit_output would never have been
    # consumed — n_calls would be 3, not 4.
    assert saved["info"]["model_stats"]["api_calls"] == 4

    # The correction message landed in the branch's context.
    branch_name = next(n for n in saved["agents"] if n.startswith("#"))
    branch_text = "\n".join(str(m.get("content", "")) for m in saved["agents"][branch_name])
    assert "cannot branch further or submit" in branch_text

    # Main ends on the real submit.
    assert saved["messages"][-1]["role"] == "exit"
    assert "task done" in saved["info"]["submission"]


def test_e2e_trajectory_roundtrips_through_json(tmp_path):
    """Saved trajectory reloads byte-for-byte equivalent via json.loads."""
    outputs = [
        _branch_open_output("explore", "look around"),
        _branch_return_output("nothing interesting"),
        _submit_output(),
    ]
    agent, saved = run_episode(
        outputs=outputs,
        agent_config={},
        task="roundtrip",
        tmp_path=tmp_path,
    )

    # Serialize the live agent and compare shape with what was written to disk.
    live = agent.serialize()
    assert live["trajectory_format"] == saved["trajectory_format"]
    assert [e["event_id"] for e in live["fold_events"]] == [
        e["event_id"] for e in saved["fold_events"]
    ]
    assert set(live["agents"].keys()) == set(saved["agents"].keys())
    # Per-agent message counts match.
    for name in live["agents"]:
        assert len(live["agents"][name]) == len(saved["agents"][name]), name


def test_e2e_wall_time_limit_enforced_in_fold_step(tmp_path):
    """FoldingAgent.step goes through DefaultAgent.query, so wall-time limits apply.

    Regression guard: the pre-split agent reimplemented the limit checks in
    its fold step and silently dropped ``wall_time_limit_seconds``.
    """
    from minisweagent.exceptions import TimeExceeded

    config = {
        **BASE_CONFIG,
        "wall_time_limit_seconds": 1,
        "output_path": tmp_path / "traj.json",
    }
    config.pop("agent_class")
    model = DeterministicModel(outputs=[
        _branch_open_output("stall", "wait around"),
        make_output("waiting", [{"command": "sleep 2"}]),
        make_output("still here", [{"command": "echo hi"}]),
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **config)
    info = agent.run("trip the wall clock")
    assert info["exit_status"] == "TimeExceeded"


def test_e2e_yaml_config_runs_real_episode(tmp_path):
    """Load the shipped YAML, instantiate via the real factory, run one episode."""
    import yaml

    cfg_path = Path("src/minisweagent/config") / "fold.yaml"
    with cfg_path.open() as f:
        agent_cfg = yaml.safe_load(f)["agent"]

    agent_cfg["cost_limit"] = 0
    agent_cfg["output_path"] = tmp_path / "traj.json"

    outputs = [
        _branch_open_output("probe", "look"),
        _branch_return_output("found nothing"),
        _submit_output(),
    ]
    model = DeterministicModel(outputs=outputs)
    env = LocalEnvironment()
    agent = get_agent(model, env, agent_cfg, default_type="")
    assert isinstance(agent, FoldingAgent)
    info = agent.run("smoke test")
    assert info["exit_status"] == "Submitted"

    saved = json.loads(agent_cfg["output_path"].read_text())
    assert saved["info"]["context_management"]["strategy"] == "fold"
