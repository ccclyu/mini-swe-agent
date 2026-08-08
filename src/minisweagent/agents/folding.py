"""Folding agent — context-folding (arXiv 2510.11967-style).

Mirrors the branch/return mechanism in https://github.com/bytedance/FoldAgent:

* The assistant emits ``<function=branch description="..." prompt="...">``
  to spawn a sub-agent. The sub-agent inherits a deep copy of the
  parent's message history plus a BRANCH_MESSAGE prompt explaining its
  scope.
* The sub-agent emits ``<function=return message="...">`` to finish.
  Its ``message`` becomes a user observation on the PARENT, and the
  sub-agent's internals never land on the parent's context.
* Inside a branch, attempts to branch again or to submit (mini-swe's
  ``finish`` equivalent) are intercepted with a correction ("use the
  ``return`` tool instead").

The branch lifecycle is recorded in ``self.fold_events`` and each
sub-agent's full message list is serialized under ``agents`` with
``trajectory_format: "mini-swe-agent-fold-1"``, so FoldGRPO-style training
can emit one trajectory per sub-agent.

For flat budget-triggered summarization (Cursor-style), see
:mod:`minisweagent.agents.compacting` — the two strategies are deliberately
separate agents: they differ in trigger (model-chosen branch points vs.
involuntary budget), trajectory shape (one history per sub-agent vs. one
rewritten history), and training signal.
"""

from __future__ import annotations

import copy
import re
import time
from typing import Any

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.utils.context_utils import count_tokens, load_encoder, message_text
from minisweagent.exceptions import Submitted

_DEFAULT_BRANCH_PROMPT = """ROLE CHANGE: `MODE: BRANCH`

You are now a branch. You have been assigned a specific task by MAIN:

{{ message }}

You inherit MAIN's full context and understanding of the problem. Focus
exclusively on the assigned task. Never perform actions beyond the
specified task. Return to MAIN immediately once the task is complete.

Your final response must be clear and compact, while faithfully capturing:
* Outcome of your assigned task.
* Any files modified, created, or deleted; any commands that changed state.
* Key insights about the codebase, problem patterns, or architecture that
  MAIN needs to know.
* Any unresolved questions or potential issues.

When you have completed your assigned task, use the return tool:

<function=return>
<parameter=message>
Your message to MAIN here — outcome, state changes, insights, notes.
</parameter>
</function>
"""

_DEFAULT_BRANCH_OBS_PROMPT = (
    "* You are now in branch mode: {{ description }}. Conduct the sub task "
    "based on the instruction; when complete, use the `return` tool. Do not "
    "perform actions beyond the assigned sub task."
)

_DEFAULT_BRANCH_SAFE_FINISH_MSG = (
    "You are in branch mode and cannot branch further or submit/finish the "
    "task. Use the `return` tool to go back to MAIN."
)

_DEFAULT_BRANCH_SUMMARY_PROMPT = (
    "The context or turn limit has been reached for this branch. Finish the "
    "sub task directly and clearly state the progress made and the pending "
    "jobs of the sub task. Only summarize the sub task progress, using the "
    "`return` tool:\n\n"
    "<function=return>\n<parameter=message>...</parameter>\n</function>"
)

_DEFAULT_BRANCH_FULL_LIMIT_MSG = (
    "You've already reached the limit of {{ max_branches }} branch calls in "
    "this episode. Continue working independently without branching further."
)

_DEFAULT_BRANCH_RETURN_TEMPLATE = (
    "Branch {{ branch_name }} has finished its task; the returned message is:"
    "\n\n{{ message }}"
)


# ---------------------------------------------------------------------------
# Parser: match FoldAgent.extract_fn_call.
# ---------------------------------------------------------------------------


def extract_fn_call(text: str | None) -> dict | None:
    """Parse the last ``<function=NAME>…</function>`` block in ``text``.

    Matches FoldAgent.agents.fold_agent.extract_fn_call byte-for-byte so
    behavior stays in lockstep.
    """
    if text is None:
        return None
    func_matches = re.findall(r"<function=([^>]+)>", text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f"<function={last_function}>")
    text_after = text[last_func_pos:]
    params = dict(re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", text_after, re.DOTALL))
    return {"function": last_function, "arguments": params}


class FoldingConfig(AgentConfig):
    max_branches: int = 5
    """Max total branches spawned in one episode (FoldAgent default)."""
    max_branch_turns: int = 64
    """Max turns inside a single branch before forcing a summary return."""
    branch_session_timeout_s: int = 60 * 90
    """Wall-clock timeout per branch, in seconds (FoldAgent default 90 min)."""
    tokenizer_name: str = "cl100k_base"
    """tiktoken encoding name for event bookkeeping. Falls back to char/4."""
    branch_prompt_template: str = _DEFAULT_BRANCH_PROMPT
    """Rendered when a branch is spawned; substituted for BRANCH_MESSAGE.
    Available Jinja vars: ``message``, ``description``."""
    branch_observation_prompt: str = _DEFAULT_BRANCH_OBS_PROMPT
    """Appended to every observation inside a branch. Empty disables."""
    branch_safe_finish_msg: str = _DEFAULT_BRANCH_SAFE_FINISH_MSG
    """Correction shown when a branch attempts to branch-again or submit."""
    branch_summary_prompt: str = _DEFAULT_BRANCH_SUMMARY_PROMPT
    """Forced summary prompt when a branch hits max_branch_turns without returning."""
    branch_full_limit_msg: str = _DEFAULT_BRANCH_FULL_LIMIT_MSG
    """Correction when a branch call is attempted past max_branches."""
    branch_return_template: str = _DEFAULT_BRANCH_RETURN_TEMPLATE
    """Formats the user observation when a branch returns. Vars: ``branch_name``, ``message``."""


class FoldingAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, *, config_class: type = FoldingConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.fold_events: list[dict] = []
        self._encoder = load_encoder(self.config.tokenizer_name)

        self.agents: dict[str, list[dict]] = {}
        """Per-sub-agent message lists. Always contains at least ``"main"``
        once :meth:`run` has been called."""
        self.active_name: str = "main"
        self._branch_parents: list[str] = []
        """Stack of parent agent names. Non-empty iff we're inside a branch."""
        self._branch_tasks: dict[str, str] = {}
        self._branch_returns: dict[str, str] = {}
        self._branch_step_counts: dict[str, int] = {}
        self._branch_descriptions: dict[str, str] = {}
        self._branch_start_times: dict[str, float] = {}

    def _render_extra(self, template: str, **extra: Any) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars(**extra))

    def run(self, task: str = "", **kwargs) -> dict:
        """DefaultAgent.run, with ``main``'s message list registered in ``self.agents``."""
        self.agents = {"main": []}
        self.active_name = "main"
        self.messages = self.agents["main"]
        # Sharing the list object with DefaultAgent.run's add_messages calls
        # requires run() to extend self.messages rather than rebind it —
        # DefaultAgent.run rebinds (`self.messages = []`), so re-register after.
        result = super().run(task, **kwargs)
        return result

    def add_messages(self, *messages: dict) -> list[dict]:
        added = super().add_messages(*messages)
        # DefaultAgent.run rebinds self.messages to a fresh list before the
        # first add; keep the active sub-agent's registry entry pointed at
        # whatever list self.messages currently is.
        self.agents[self.active_name] = self.messages
        return added

    def step(self) -> list[dict]:
        """One step, dispatched to the currently-active sub-agent."""
        # Per-branch turn + timeout guards. If the branch has been running too
        # long, force a summary return before querying (FoldAgent Agent.react's
        # post-loop summary_prompt fallback).
        if self._in_branch:
            name = self.active_name
            if self._branch_step_counts.get(name, 0) >= self.config.max_branch_turns:
                self.logger.info("branch %s hit max_branch_turns; forcing summary return", name)
                return self._force_branch_summary_return()
            timeout = self.config.branch_session_timeout_s
            if timeout > 0 and time.time() - self._branch_start_times.get(name, time.time()) > timeout:
                self.logger.info("branch %s hit session timeout; forcing summary return", name)
                return self._force_branch_summary_return()
            self._branch_step_counts[name] = self._branch_step_counts.get(name, 0) + 1

        self.messages = self.agents[self.active_name]
        # DefaultAgent.query: step/cost/wall-time limit checks, model call,
        # cost accounting, and append to self.messages (the active list).
        model_msg = self.query()

        fn_call = extract_fn_call(message_text(model_msg))

        if self._in_branch and self._is_illegal_branch_action(fn_call, model_msg):
            self._append_correction(self.config.branch_safe_finish_msg)
            return [model_msg]

        if fn_call is not None and fn_call["function"] == "branch":
            self._spawn_branch(fn_call)
            return [model_msg]

        if fn_call is not None and fn_call["function"] == "return":
            if self._in_branch:
                self._finish_branch(fn_call["arguments"].get("message", ""))
                return [model_msg]
            self._append_correction("`return` is only valid inside a branch; use the normal submit flow.")
            return [model_msg]

        return self._execute_bash_actions(model_msg)

    def _execute_bash_actions(self, model_msg: dict) -> list[dict]:
        """Execute the bash actions on the current sub-agent's env.

        In a branch, Submitted is intercepted and turned into a correction so
        that the branch cannot end the episode on MAIN's behalf.
        """
        outputs: list[dict] = []
        try:
            for action in model_msg.get("extra", {}).get("actions", []):
                outputs.append(self.env.execute(action))
        except Submitted:
            if self._in_branch:
                self._append_correction(self.config.branch_safe_finish_msg)
                return [model_msg]
            raise

        obs_msgs = list(self.model.format_observation_messages(model_msg, outputs, self.get_template_vars()))
        if self._in_branch and self.config.branch_observation_prompt:
            desc = self._branch_descriptions.get(self.active_name, "")
            reminder = self._render_extra(self.config.branch_observation_prompt, description=desc)
            for om in obs_msgs:
                c = om.get("content")
                if isinstance(c, str):
                    om["content"] = c + "\n" + reminder
        return [model_msg, *self.add_messages(*obs_msgs)]

    # ---- branch/return -------------------------------------------------

    @property
    def _in_branch(self) -> bool:
        return bool(self._branch_parents)

    def _branch_names(self) -> list[str]:
        return [n for n in self.agents if n.startswith("#")]

    def _is_illegal_branch_action(self, fn_call: dict | None, model_msg: dict) -> bool:
        """FoldAgent safe_finish equivalent: branches can't branch again or submit."""
        if fn_call is not None and fn_call["function"] in ("branch", "finish"):
            return True
        for action in model_msg.get("extra", {}).get("actions", []) or []:
            cmd = action.get("command") if isinstance(action, dict) else None
            if isinstance(cmd, str) and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
                return True
        return False

    def _append_correction(self, text: str) -> None:
        self.add_messages(self.model.format_message(role="user", content=text, extra={"source": "correction"}))

    def _spawn_branch(self, fn_call: dict) -> None:
        """Create a new sub-agent whose history is a deep copy of the parent's."""
        existing = self._branch_names()
        max_b = self.config.max_branches
        if len(existing) + 1 > max_b:
            msg = self._render_extra(self.config.branch_full_limit_msg, max_branches=max_b)
            self._append_correction(msg)
            return

        description = fn_call["arguments"].get("description", f"branch_{len(existing)}")
        prompt = fn_call["arguments"].get("prompt", "")
        safe_desc = re.sub(r"\s+", "_", description.strip()) or f"branch_{len(existing)}"
        branch_name = f"#{len(existing)}-{safe_desc}"

        parent_name = self.active_name
        child_msgs = copy.deepcopy(self.agents[parent_name])
        branch_prompt_rendered = self._render_extra(
            self.config.branch_prompt_template, message=prompt, description=description
        )
        child_msgs.append(self.model.format_message(
            role="user", content=branch_prompt_rendered,
            extra={"source": "branch_open", "branch_name": branch_name, "description": description},
        ))

        self.agents[branch_name] = child_msgs
        self._branch_tasks[branch_name] = prompt
        self._branch_descriptions[branch_name] = description
        self._branch_start_times[branch_name] = time.time()
        self._branch_step_counts[branch_name] = 0
        self._branch_parents.append(parent_name)
        self.active_name = branch_name
        self.messages = child_msgs
        self.fold_events.append({
            "event_id": branch_name,
            "kind": "branch_open",
            "parent": parent_name,
            "description": description,
            "prompt": prompt,
            "parent_messages_len_at_open": len(self.agents[parent_name]),
        })
        self.logger.info("opened branch %s (parent=%s, desc=%s)", branch_name, parent_name, description)

    def _finish_branch(self, message: str) -> None:
        """Pop back to parent; append the return ``message`` as its observation."""
        if not self._in_branch:
            return
        branch_name = self.active_name
        parent_name = self._branch_parents.pop()
        self._branch_returns[branch_name] = message

        self.active_name = parent_name
        self.messages = self.agents[parent_name]
        formatted = self._render_extra(
            self.config.branch_return_template, branch_name=branch_name, message=message,
        )
        self.add_messages(self.model.format_message(
            role="user", content=formatted,
            extra={"source": "branch_return", "branch_name": branch_name},
        ))
        self.fold_events.append({
            "event_id": f"{branch_name}:return",
            "kind": "branch_return",
            "parent": parent_name,
            "branch_name": branch_name,
            "return_message_tokens": count_tokens([self.messages[-1]], self._encoder),
        })
        self.logger.info("branch %s returned to %s (%d chars)", branch_name, parent_name, len(message))

    def _force_branch_summary_return(self) -> list[dict]:
        """Synthesize a return: append summary prompt, query, wrap as return.

        Mirrors FoldAgent.Agent.react's post-loop ``summary_prompt`` fallback.
        Used when a branch hits its turn/time budget without emitting return.
        """
        summary_prompt = self._render_extra(self.config.branch_summary_prompt)
        prompt_msg = self.model.format_message(
            role="user", content=summary_prompt,
            extra={"source": "branch_forced_summary"},
        )
        self.add_messages(prompt_msg)

        response = self.query()

        content = message_text(response)
        fn_call = extract_fn_call(content)
        if fn_call is not None and fn_call["function"] == "return":
            message = fn_call["arguments"].get("message", content)
        else:
            # No return emitted even under duress — use the raw response as the
            # branch's return message. Matches FoldAgent's "last_response" fallback.
            message = content or "(branch ended without summary)"

        self._finish_branch(message)
        return [prompt_msg, response]

    # ==================================================================
    # serialization
    # ==================================================================

    def serialize(self, *extra_dicts) -> dict:
        data = super().serialize(*extra_dicts)
        data["trajectory_format"] = "mini-swe-agent-fold-1"
        data["fold_events"] = list(self.fold_events)
        data.setdefault("info", {})["context_management"] = {
            "strategy": "fold",
            "n_branches": len(self._branch_names()),
            "open_branches": list(self._branch_parents),
            "branch_tasks": dict(self._branch_tasks),
            "branch_returns": dict(self._branch_returns),
        }
        # Expose every sub-agent's message list so FoldGRPO training can
        # emit one trajectory per agent (matching FoldAgent.process_item's
        # list[AgentLoopOutput] return).
        data["agents"] = {name: list(msgs) for name, msgs in self.agents.items()}
        return data
