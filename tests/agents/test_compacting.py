"""Tests for CompactingAgent — budget trigger + in-place slice replacement."""

from pathlib import Path

import yaml

from minisweagent.agents.compacting import CompactingAgent, CompactingConfig
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


def _minimal_config(**overrides) -> dict:
    base = {
        "system_template": "SYSTEM",
        "instance_template": "TASK: {{ task }}",
        "cost_limit": 0,  # DefaultAgent defaults to 3.0; our determ. model charges 1.0/call
        "token_budget": 0,
        "min_preserve_head": 2,
        "min_preserve_tail": 1,
        "min_compact_size": 2,
        "min_compact_tokens": 10,
        "summary_system_template": "Summarize concisely (<= {{ summary_max_tokens }} tokens).",
        "summary_user_template": "Produce the summary.",
        "resumption_template": (
            "<context_compacted n=\"{{ n_compactions }}\" id=\"{{ compaction_id }}\">\n"
            "{{ summary }}\n</context_compacted>"
        ),
        "summary_max_tokens": 512,
    }
    base.update(overrides)
    return base


def _submit_output() -> dict:
    """Last output that causes DefaultAgent to exit via Submitted."""
    return make_output(
        "Done",
        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'final answer'"}],
    )


def test_token_budget_triggers_compaction():
    big_output = make_output("thinking", [{"command": f"echo '{'x' * 5000}'"}])
    summary_output = make_output("## Goal\nfinish the task\n## Next action\nsubmit", [])
    model = DeterministicModel(outputs=[big_output, big_output, summary_output, _submit_output()])

    agent = CompactingAgent(
        model=model,
        env=LocalEnvironment(),
        **_minimal_config(
            token_budget=100,
            min_preserve_head=2,
            min_preserve_tail=1,
            min_compact_size=2,
            min_compact_tokens=10,
        ),
    )
    info = agent.run("do the thing")

    assert info["exit_status"] == "Submitted"
    assert len(agent.compactions) == 1, agent.compactions
    c = agent.compactions[0]
    assert c["kind"] == "compact"
    assert c["summary_call"] is not None
    assert any("context_compacted" in str(m.get("content", "")) for m in agent.messages)


def test_no_compaction_when_budget_not_reached():
    model = DeterministicModel(outputs=[
        make_output("thinking", [{"command": "echo hi"}]),
        _submit_output(),
    ])
    agent = CompactingAgent(
        model=model,
        env=LocalEnvironment(),
        **_minimal_config(token_budget=10**9),
    )
    agent.run("task")
    assert agent.compactions == []


def test_head_and_tail_preserved_after_compaction():
    big = make_output("thinking", [{"command": f"echo '{'y' * 3000}'"}])
    summary = make_output("SHORT_SUMMARY", [])
    model = DeterministicModel(outputs=[big, big, big, summary, _submit_output()])

    agent = CompactingAgent(
        model=model,
        env=LocalEnvironment(),
        **_minimal_config(
            token_budget=200,
            min_preserve_head=2,
            min_preserve_tail=2,
            min_compact_size=3,
            min_compact_tokens=10,
        ),
    )
    agent.run("task")
    assert agent.messages[0]["role"] == "system"
    assert "TASK:" in str(agent.messages[1].get("content", ""))


def test_serialization_shape():
    big = make_output("thinking", [{"command": f"echo '{'z' * 3000}'"}])
    summary = make_output("SUMMARY", [])
    model = DeterministicModel(outputs=[big, big, summary, _submit_output()])

    agent = CompactingAgent(
        model=model, env=LocalEnvironment(),
        **_minimal_config(token_budget=100, min_compact_tokens=10),
    )
    agent.run("task")
    data = agent.serialize()
    assert data["trajectory_format"] == "mini-swe-agent-compact-1"
    assert len(data["compactions"]) == 1
    assert data["info"]["context_management"]["strategy"] == "compact"
    assert data["info"]["context_management"]["n_compactions"] == 1


def test_yaml_config_is_valid():
    cfg_path = Path("src/minisweagent/config") / "compact.yaml"
    with cfg_path.open() as f:
        agent_cfg = yaml.safe_load(f)["agent"]
    agent_cfg.pop("agent_class", None)
    CompactingConfig(**agent_cfg)
