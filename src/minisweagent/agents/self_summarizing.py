"""Self-summarizing agent.

Two modes:

- ``selfsum`` — flat summarization. When the estimated token
  count of ``self.messages`` crosses ``token_budget``, synthesize a summary
  of the interior messages and replace them in-place with a single user
  message carrying the summary. The head (system + initial task) and tail
  (last ``min_preserve_tail`` messages) are always kept verbatim.

- ``fold`` — context-folding (arXiv 2510.11967-style). Mirrors the
  branch/return mechanism in https://github.com/bytedance/FoldAgent:

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
  * Budget trigger (token_budget, if > 0) is not used in fold mode —
    the paper relies on branching, not summarization, for context
    reduction. Add ``mode: selfsum`` separately if you want both.
"""

from __future__ import annotations

import copy
import re
import time
from typing import Any, Literal

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded, Submitted

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


class SelfSummarizingConfig(AgentConfig):
    """Config for :class:`SelfSummarizingAgent`. See module docstring for modes."""

    mode: Literal["selfsum", "fold"] = "selfsum"

    # --- selfsum: budget trigger ------------------------------------
    token_budget: int = 80_000
    """Compact once estimated tokens cross this. Only used in selfsum mode."""
    step_budget: int = 0
    """Compact every N steps. Use 0 to disable. Only used in selfsum mode."""
    tokenizer_name: str = "cl100k_base"
    """tiktoken encoding name. Falls back to char/4 if tiktoken is unavailable."""

    # --- selfsum: slice selection -----------------------------------
    min_preserve_head: int = 2
    min_preserve_tail: int = 4
    min_fold_size: int = 6
    min_fold_tokens: int = 1024

    # --- selfsum: summary call --------------------------------------
    summary_system_template: str = (
        "You are preparing to hand off this task to your future self.\n"
        "Write a condensed brief under {{ summary_max_tokens }} tokens."
    )
    summary_user_template: str = "Produce the handoff summary now."
    summary_max_tokens: int = 2048
    resumption_template: str = (
        "<context_compacted n=\"{{ n_compactions }}\">\n{{ summary }}\n"
        "</context_compacted>"
    )

    # --- fold: branch/return mechanics ------------------------------
    max_branches: int = 5
    """Max total branches spawned in one episode (FoldAgent default)."""
    max_branch_turns: int = 64
    """Max turns inside a single branch before forcing a summary return."""
    branch_session_timeout_s: int = 60 * 90
    """Wall-clock timeout per branch, in seconds (FoldAgent default 90 min)."""
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


class SelfSummarizingAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, *, config_class: type = SelfSummarizingConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)

        # selfsum state
        self.compactions: list[dict] = []
        self._encoder = self._load_encoder(self.config.tokenizer_name)

        # fold state
        self.agents: dict[str, list[dict]] = {}
        """Per-sub-agent message lists. Populated in fold mode; always contains
        at least ``"main"`` once :meth:`run` has been called."""
        self.active_name: str = "main"
        self._branch_parents: list[str] = []
        """Stack of parent agent names. Non-empty iff we're inside a branch."""
        self._branch_tasks: dict[str, str] = {}
        self._branch_returns: dict[str, str] = {}
        self._branch_step_counts: dict[str, int] = {}
        self._branch_descriptions: dict[str, str] = {}
        self._branch_start_times: dict[str, float] = {}

    @staticmethod
    def _load_encoder(name: str):
        try:
            import tiktoken

            try:
                return tiktoken.get_encoding(name)
            except (KeyError, ValueError):
                return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return None

    def _message_text(self, msg: dict) -> str:
        from minisweagent.models.utils.content_string import get_content_string

        return get_content_string(msg) or ""

    def _count_tokens(self, messages: list[dict]) -> int:
        text = "\n".join(self._message_text(m) for m in messages)
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return max(1, len(text) // 4)

    def _extract_assistant_text(self, message: dict) -> str:
        return self._message_text(message)

    def _render_template(self, template: str, **extra: Any) -> str:
        vars_ = self.get_template_vars()
        vars_.update(extra)
        return Template(template, undefined=StrictUndefined).render(**vars_)

    def run(self, task: str = "", **kwargs) -> dict:
        """Run until the agent exits. Supports selfsum (default) and fold modes."""
        self.extra_template_vars |= {"task": task, **kwargs}

        main_msgs: list[dict] = [
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        ]
        if self.config.mode == "fold":
            self.agents = {"main": main_msgs}
            self.active_name = "main"
        self.messages = main_msgs

        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Dispatch on mode. selfsum uses DefaultAgent's step with a pre-query
        compaction check; fold uses a branch/return-aware step."""
        if self.config.mode == "selfsum":
            self._maybe_compact_on_budget()
            return super().step()
        return self._fold_step()

    # ==================================================================
    # selfsum mode (unchanged mechanics)
    # ==================================================================

    def _maybe_compact_on_budget(self) -> None:
        budget = self.config.token_budget
        step_budget = self.config.step_budget
        should_compact = False
        if budget > 0 and self._count_tokens(self.messages) >= budget:
            should_compact = True
        elif step_budget > 0 and self.n_calls > 0 and self.n_calls % step_budget == 0:
            should_compact = True
        if not should_compact:
            return
        lo = self.config.min_preserve_head
        hi = len(self.messages) - self.config.min_preserve_tail
        if hi - lo < self.config.min_fold_size:
            return
        if self._count_tokens(self.messages[lo:hi]) < self.config.min_fold_tokens:
            return
        self._compact_slice(lo, hi, subtask="", kind="compact")

    def _compact_slice(
        self,
        lo: int,
        hi: int,
        *,
        subtask: str,
        kind: str,
    ) -> None:
        """selfsum in-place slice replacement. See module docstring."""
        assert 0 <= lo < hi <= len(self.messages), f"bad slice [{lo}, {hi}] of {len(self.messages)}"

        slice_msgs = list(self.messages[lo:hi])
        replaced_tokens = self._count_tokens(slice_msgs)

        summary, summary_call_info = self._run_summary_call(slice_msgs, subtask=subtask)

        compaction_id = f"c{len(self.compactions) + 1}"
        resumption = self._render_template(
            self.config.resumption_template,
            summary=summary,
            n_compactions=len(self.compactions) + 1,
            compaction_id=compaction_id,
            subtask=subtask,
        )
        replacement_msg = self.model.format_message(
            role="user",
            content=resumption,
            extra={"source": "compaction", "compaction_id": compaction_id, "kind": kind, "subtask": subtask},
        )

        self.messages[lo:hi] = [replacement_msg]
        self.compactions.append({
            "compaction_id": compaction_id,
            "kind": kind,
            "subtask": subtask,
            "replaced_range": [lo, hi],
            "replaced_token_count": replaced_tokens,
            "summary_message_index": lo,
            "summary_call": summary_call_info,
            "summary_token_count": self._count_tokens([replacement_msg]),
        })
        self.logger.info(
            "compacted %s [%d, %d): %d -> %d tokens",
            kind, lo, hi, replaced_tokens, self.compactions[-1]["summary_token_count"],
        )

    def _run_summary_call(self, slice_msgs: list[dict], *, subtask: str) -> tuple[str, dict]:
        """Ask the model for a summary over ``slice_msgs``.

        Bypasses the model's action parser: the summary response is free-form
        text, so running it through the bash/tool-call parser would raise
        FormatError on every valid summary. We call ``model._query`` directly
        (raw litellm call) and extract the content ourselves.
        """
        from minisweagent.exceptions import FormatError
        from minisweagent.models.utils.content_string import get_content_string

        sys_text = self._render_template(self.config.summary_system_template, subtask=subtask)
        user_text = self._render_template(self.config.summary_user_template, subtask=subtask)
        slice_text = "\n\n".join(
            f"[{m.get('role') or m.get('type', 'unknown')}]\n{get_content_string(m)}"
            for m in slice_msgs
        )
        derived = [
            self.model.format_message(role="system", content=sys_text),
            self.model.format_message(
                role="user",
                content=f"<conversation_to_summarize>\n{slice_text}\n</conversation_to_summarize>\n\n{user_text}",
            ),
        ]
        self.n_calls += 1

        # Preferred path: most LitellmModel subclasses expose ``_query`` which
        # returns the raw litellm response. When available we grab that so the
        # action parser never runs. Fallback: swallow FormatError from the
        # regular ``query`` so the summary's parse-failure doesn't kill the
        # compaction.
        summary_text = ""
        cost = 0.0
        timestamp = None
        if hasattr(self.model, "_query"):
            try:
                raw = self.model._query(self.model._prepare_messages_for_api(derived))
                summary_text = raw.choices[0].message.content or ""
            except Exception as e:  # network/auth/parse — fall through to wrapped path
                self.logger.warning("summary _query failed (%s); falling back to query", e)
                try:
                    response = self.model.query(derived)
                    summary_text = get_content_string(response) or ""
                    cost = response.get("extra", {}).get("cost", 0.0)
                    timestamp = response.get("extra", {}).get("timestamp")
                except FormatError as fe:
                    # The summary response was free-form text; harvest it from
                    # the FormatError's attached message instead of erroring.
                    self.logger.info("summary response was free-form text; harvesting from FormatError")
                    for msg in fe.messages:
                        c = get_content_string(msg) or ""
                        if c:
                            summary_text = c
                            break
        else:
            try:
                response = self.model.query(derived)
                summary_text = get_content_string(response) or ""
                cost = response.get("extra", {}).get("cost", 0.0)
                timestamp = response.get("extra", {}).get("timestamp")
            except FormatError as fe:
                for msg in fe.messages:
                    c = get_content_string(msg) or ""
                    if c:
                        summary_text = c
                        break

        self.cost += cost
        return summary_text, {
            "cost": cost,
            "timestamp": timestamp,
            "prompt_tokens": self._count_tokens(derived),
            "completion_tokens": self._count_tokens(
                [self.model.format_message(role="assistant", content=summary_text)]
            ),
        }

    # ==================================================================
    # fold mode
    # ==================================================================

    @property
    def _in_branch(self) -> bool:
        return bool(self._branch_parents)

    def _append_to_active(self, msg: dict) -> None:
        self.agents[self.active_name].append(msg)
        self.messages = self.agents[self.active_name]

    def _fold_step(self) -> list[dict]:
        """One step in fold mode, dispatched to the currently-active sub-agent."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {"role": "exit", "content": "LimitsExceeded",
                 "extra": {"exit_status": "LimitsExceeded", "submission": ""}}
            )

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

        active_msgs = self.agents[self.active_name]
        self.messages = active_msgs

        self.n_calls += 1
        model_msg = self.model.query(active_msgs)
        self.cost += model_msg.get("extra", {}).get("cost", 0.0)
        self._append_to_active(model_msg)

        content = self._extract_assistant_text(model_msg)
        fn_call = extract_fn_call(content)

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
            reminder = self._render_template(self.config.branch_observation_prompt, description=desc)
            for om in obs_msgs:
                c = om.get("content")
                if isinstance(c, str):
                    om["content"] = c + "\n" + reminder
        for om in obs_msgs:
            self._append_to_active(om)
        return [model_msg, *obs_msgs]

    # ---- branch/return -------------------------------------------------

    def _branch_names(self) -> list[str]:
        return [n for n in self.agents if n.startswith("#")]

    def _is_illegal_branch_action(self, fn_call: dict | None, model_msg: dict) -> bool:
        """FoldAgent safe_finish equivalent: branches can't branch again or submit."""
        if fn_call is not None and fn_call["function"] == "branch":
            return True
        if fn_call is not None and fn_call["function"] == "finish":
            return True
        for action in model_msg.get("extra", {}).get("actions", []) or []:
            cmd = action.get("command") if isinstance(action, dict) else None
            if isinstance(cmd, str) and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in cmd:
                return True
        return False

    def _append_correction(self, text: str) -> None:
        self._append_to_active(
            self.model.format_message(role="user", content=text, extra={"source": "correction"})
        )

    def _spawn_branch(self, fn_call: dict) -> None:
        """Create a new sub-agent whose history is a deep copy of the parent's."""
        existing = self._branch_names()
        max_b = self.config.max_branches
        if len(existing) + 1 > max_b:
            msg = self._render_template(self.config.branch_full_limit_msg, max_branches=max_b)
            self._append_correction(msg)
            return

        description = fn_call["arguments"].get("description", f"branch_{len(existing)}")
        prompt = fn_call["arguments"].get("prompt", "")
        safe_desc = re.sub(r"\s+", "_", description.strip()) or f"branch_{len(existing)}"
        branch_name = f"#{len(existing)}-{safe_desc}"

        parent_name = self.active_name
        child_msgs = copy.deepcopy(self.agents[parent_name])
        branch_prompt_rendered = self._render_template(
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
        self.compactions.append({
            "compaction_id": branch_name,
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
        parent_msgs = self.agents[parent_name]
        formatted = self._render_template(
            self.config.branch_return_template, branch_name=branch_name, message=message,
        )
        parent_msgs.append(self.model.format_message(
            role="user", content=formatted,
            extra={"source": "branch_return", "branch_name": branch_name},
        ))
        self.messages = parent_msgs
        self.compactions.append({
            "compaction_id": f"{branch_name}:return",
            "kind": "branch_return",
            "parent": parent_name,
            "branch_name": branch_name,
            "return_message_tokens": self._count_tokens([parent_msgs[-1]]),
        })
        self.logger.info("branch %s returned to %s (%d chars)", branch_name, parent_name, len(message))

    def _force_branch_summary_return(self) -> list[dict]:
        """Synthesize a return: append summary prompt, query, wrap as return.

        Mirrors FoldAgent.Agent.react's post-loop ``summary_prompt`` fallback.
        Used when a branch hits its turn/time budget without emitting return.
        """
        active_msgs = self.agents[self.active_name]
        summary_prompt = self._render_template(self.config.branch_summary_prompt)
        prompt_msg = self.model.format_message(
            role="user", content=summary_prompt,
            extra={"source": "branch_forced_summary"},
        )
        self._append_to_active(prompt_msg)

        self.n_calls += 1
        response = self.model.query(active_msgs)
        self.cost += response.get("extra", {}).get("cost", 0.0)
        self._append_to_active(response)

        content = self._extract_assistant_text(response)
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
        data["trajectory_format"] = "mini-swe-agent-selfsum-1"
        data["compactions"] = list(self.compactions)
        data.setdefault("info", {})["self_summarizing"] = {
            "mode": self.config.mode,
            "token_budget": self.config.token_budget,
            "n_compactions": len(self.compactions),
        }
        if self.config.mode == "fold":
            data["info"]["self_summarizing"].update({
                "n_branches": len(self._branch_names()),
                "open_branches": list(self._branch_parents),
                "branch_tasks": dict(self._branch_tasks),
                "branch_returns": dict(self._branch_returns),
            })
            # Expose every sub-agent's message list so FoldGRPO training can
            # emit one trajectory per agent (matching FoldAgent.process_item's
            # list[AgentLoopOutput] return).
            data["agents"] = {name: list(msgs) for name, msgs in self.agents.items()}
        return data
