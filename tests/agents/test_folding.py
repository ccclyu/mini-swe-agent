"""Tests for FoldingAgent — ``<function=branch>`` / ``<function=return>``."""

from pathlib import Path

import yaml

from minisweagent.agents.folding import FoldingAgent, FoldingConfig, extract_fn_call
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


def _minimal_config(**overrides) -> dict:
    base = {
        "system_template": "SYSTEM",
        "instance_template": "TASK: {{ task }}",
        "cost_limit": 0,  # DefaultAgent defaults to 3.0; our determ. model charges 1.0/call
    }
    base.update(overrides)
    return base


def _submit_output() -> dict:
    """Last output that causes DefaultAgent to exit via Submitted."""
    return make_output(
        "Done",
        [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'final answer'"}],
    )


# ----------------------------------------------------------------------
# extract_fn_call (1:1 port of FoldAgent helper)
# ----------------------------------------------------------------------


def test_extract_fn_call_single_block():
    text = (
        "some thinking\n"
        "<function=branch>\n"
        "<parameter=description>explore src</parameter>\n"
        "<parameter=prompt>list files</parameter>\n"
        "</function>"
    )
    fc = extract_fn_call(text)
    assert fc == {"function": "branch", "arguments": {"description": "explore src", "prompt": "list files"}}


def test_extract_fn_call_last_of_many():
    """FoldAgent.extract_fn_call takes the LAST `<function=...>` block."""
    text = (
        "<function=think><parameter=content>x</parameter></function>"
        "<function=return><parameter=message>done</parameter></function>"
    )
    fc = extract_fn_call(text)
    assert fc["function"] == "return"
    assert fc["arguments"]["message"] == "done"


def test_extract_fn_call_none():
    assert extract_fn_call("just plain text") is None
    assert extract_fn_call(None) is None


# ----------------------------------------------------------------------
# branch/return mechanics
# ----------------------------------------------------------------------


def _branch_open_output(description: str, prompt: str) -> dict:
    """Assistant output that fires a `<function=branch>` call. No bash action."""
    content = (
        "Let me investigate this.\n"
        f"<function=branch>\n"
        f"<parameter=description>{description}</parameter>\n"
        f"<parameter=prompt>{prompt}</parameter>\n"
        "</function>"
    )
    return make_output(content, [])


def _branch_return_output(message: str) -> dict:
    """Branch-side: emits `<function=return>` to hand control back to parent."""
    content = (
        "Subtask complete.\n"
        "<function=return>\n"
        f"<parameter=message>{message}</parameter>\n"
        "</function>"
    )
    return make_output(content, [])


def test_spawn_branch_and_return():
    """Parent branches → child runs → child returns → parent sees return message."""
    model = DeterministicModel(outputs=[
        _branch_open_output("reproduce bug", "run the failing test"),  # main turn 1
        make_output("running test inside branch", [{"command": "echo 'test run'"}]),  # branch turn 1
        _branch_return_output("Reproduced: fails at line 42 with ValueError"),  # branch turn 2 → return
        _submit_output(),  # main turn 2 (post-return)
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **_minimal_config())
    info = agent.run("fix the bug")
    assert info["exit_status"] == "Submitted"

    # Two sub-agents recorded: main + the branch.
    assert "main" in agent.agents
    branch_names = [n for n in agent.agents if n.startswith("#")]
    assert len(branch_names) == 1
    branch_name = branch_names[0]
    assert "reproduce_bug" in branch_name

    # Return message landed as a user observation on main.
    main_texts = [str(m.get("content", "")) for m in agent.agents["main"]]
    assert any("Reproduced: fails at line 42 with ValueError" in t for t in main_texts)

    # Fold events log both the open and the return.
    kinds = [e["kind"] for e in agent.fold_events]
    assert kinds == ["branch_open", "branch_return"]


def test_branch_internals_do_not_leak_to_parent():
    """The branch's tool calls and observations must NOT appear in main's messages."""
    model = DeterministicModel(outputs=[
        _branch_open_output("explore", "look at the code"),
        make_output("exploring", [{"command": "echo 'SECRET_BRANCH_COMMAND_OUTPUT'"}]),
        _branch_return_output("explored; nothing unusual"),
        _submit_output(),
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **_minimal_config())
    agent.run("task")

    main_serialized = "\n".join(str(m.get("content", "")) for m in agent.agents["main"])
    assert "SECRET_BRANCH_COMMAND_OUTPUT" not in main_serialized, (
        "branch's internal bash output must not bleed into main's context"
    )


def test_branch_cannot_submit():
    """Inside a branch, COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is intercepted.

    The branch must be told to use `return` instead of ending the episode.
    """
    model = DeterministicModel(outputs=[
        _branch_open_output("explore", "look"),
        # Branch tries to submit instead of returning — should be intercepted.
        make_output("time to finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'"}]),
        # Branch is corrected and tries again with return.
        _branch_return_output("ok, returning properly"),
        _submit_output(),  # main finishes the real episode
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **_minimal_config())
    info = agent.run("task")
    # Main must be what submits — if the branch had ended the episode, the
    # post-return submit_output message would never have been consumed.
    assert info["exit_status"] == "Submitted"
    assert agent.n_calls == 4

    # The correction must have landed in the branch's history.
    branch_name = next(n for n in agent.agents if n.startswith("#"))
    branch_texts = [str(m.get("content", "")) for m in agent.agents[branch_name]]
    assert any("cannot branch further or submit" in t for t in branch_texts)


def test_branch_cannot_nest_branches():
    """Inside a branch, attempting another `<function=branch>` is intercepted."""
    model = DeterministicModel(outputs=[
        _branch_open_output("outer", "do outer"),
        # Branch tries to branch — should be corrected, NOT spawn a second branch.
        _branch_open_output("inner", "do inner"),
        _branch_return_output("returning without nesting"),
        _submit_output(),
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **_minimal_config())
    agent.run("task")
    # Only ONE branch should have been spawned.
    branch_names = [n for n in agent.agents if n.startswith("#")]
    assert len(branch_names) == 1, f"expected 1 branch, got {branch_names}"


def test_max_branches_enforced():
    """Past max_branches, a `<function=branch>` becomes a correction, not a spawn."""
    model = DeterministicModel(outputs=[
        _branch_open_output("first", "do first"),
        _branch_return_output("first done"),
        _branch_open_output("second", "do second"),  # would be #1 if allowed
        # max_branches=1 → this is rejected; main must keep running.
        make_output("continuing without branching", [{"command": "echo 'ok'"}]),
        _submit_output(),
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **_minimal_config(max_branches=1))
    info = agent.run("task")
    assert info["exit_status"] == "Submitted"
    branch_names = [n for n in agent.agents if n.startswith("#")]
    assert len(branch_names) == 1


def test_branch_return_template_formatting():
    """The return observation on the parent uses branch_return_template."""
    model = DeterministicModel(outputs=[
        _branch_open_output("subtask", "do it"),
        _branch_return_output("result-payload"),
        _submit_output(),
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **_minimal_config())
    agent.run("task")
    # The last user message on main before submit must carry the templated return.
    contents = [str(m.get("content", "")) for m in agent.agents["main"]]
    return_observations = [c for c in contents if "result-payload" in c]
    assert return_observations, f"no return observation found in main: {contents}"
    assert "finished its task" in return_observations[0]
    assert "result-payload" in return_observations[0]


def test_observation_carries_branch_reminder():
    """Observations inside a branch get branch_observation_prompt appended."""
    model = DeterministicModel(outputs=[
        _branch_open_output("obs_test", "run a command"),
        make_output("doing", [{"command": "echo 'inside-branch-output'"}]),
        _branch_return_output("done"),
        _submit_output(),
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **_minimal_config())
    agent.run("task")
    branch_name = next(n for n in agent.agents if n.startswith("#"))
    branch_contents = [str(m.get("content", "")) for m in agent.agents[branch_name]]
    # The observation immediately after the branch's bash call carries the reminder.
    reminders = [c for c in branch_contents if "now in branch mode" in c]
    assert reminders, "branch observation_prompt missing"
    assert any("obs_test" in r for r in reminders)


def test_forced_summary_on_max_turns():
    """Branch that never emits `<function=return>` hits max_branch_turns and is forced."""
    # Open a branch, then have the branch emit plain non-return content for 3 turns.
    # With max_branch_turns=3, the 4th would be forced to a summary.
    plain_branch_turn = make_output("still thinking", [{"command": "echo 'step'"}])
    # The forced summary query returns an assistant message with <function=return>.
    forced_return = _branch_return_output("forced summary: nothing conclusive")

    model = DeterministicModel(outputs=[
        _branch_open_output("endless", "do forever"),
        plain_branch_turn,  # branch step 1
        plain_branch_turn,  # branch step 2
        plain_branch_turn,  # branch step 3
        forced_return,      # forced-summary query response
        _submit_output(),   # main resumes
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **_minimal_config(max_branch_turns=3))
    info = agent.run("task")
    assert info["exit_status"] == "Submitted"

    branch_name = next(n for n in agent.agents if n.startswith("#"))
    branch_contents = [str(m.get("content", "")) for m in agent.agents[branch_name]]
    # Forced-summary prompt ("The context or turn limit...") appears in the branch.
    assert any("turn limit has been reached" in c for c in branch_contents)
    # The forced return landed on main.
    main_contents = [str(m.get("content", "")) for m in agent.agents["main"]]
    assert any("forced summary: nothing conclusive" in c for c in main_contents)


def test_serialization_exposes_per_agent_messages():
    """serialize() must emit each sub-agent's messages under `agents`."""
    model = DeterministicModel(outputs=[
        _branch_open_output("probe", "probe the repo"),
        _branch_return_output("probed"),
        _submit_output(),
    ])
    agent = FoldingAgent(model=model, env=LocalEnvironment(), **_minimal_config())
    agent.run("task")
    data = agent.serialize()
    assert data["trajectory_format"] == "mini-swe-agent-fold-1"
    assert data["info"]["context_management"]["strategy"] == "fold"
    assert "agents" in data
    assert "main" in data["agents"]
    branch_names = [n for n in data["agents"] if n.startswith("#")]
    assert len(branch_names) == 1
    assert data["info"]["context_management"]["n_branches"] == 1


def test_yaml_config_is_valid():
    cfg_path = Path("src/minisweagent/config") / "fold.yaml"
    with cfg_path.open() as f:
        agent_cfg = yaml.safe_load(f)["agent"]
    agent_cfg.pop("agent_class", None)
    FoldingConfig(**agent_cfg)
