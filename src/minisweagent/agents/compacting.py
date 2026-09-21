"""Compacting agent — Cursor-style flat self-summarization (inline mode).

The summary is elicited INLINE: an elicitation prompt is appended to the live
conversation and the model writes the handoff summary as its natural next turn
(raw transport, action parser bypassed). The interior slice is then replaced by
the summary; the elicitation exchange is preserved in ``compactions[].summary_
exchange`` only. At training time the summary call is a prefix-extension of its
segment, so an N-compaction episode yields 1+N rows.


When the estimated token count of ``self.messages`` crosses ``token_budget``
(checked before every query), the agent synthesizes a summary of the interior
messages and replaces them in-place with a single user message carrying the
summary. The head (system + initial task) and tail (last
``min_preserve_tail`` messages) are always kept verbatim.

Every compaction is recorded in ``self.compactions`` and serialized under
``trajectory_format: "mini-swe-agent-compact-1"`` so training code can
reconstruct which tokens were replaced.

For the hierarchical branch/return variant (context-folding), see
:mod:`minisweagent.agents.folding` — the two strategies are deliberately
separate agents: they differ in trigger (involuntary budget vs. model-chosen
branch points), trajectory shape (one rewritten history vs. one history per
sub-agent), and training signal.
"""

from __future__ import annotations

from typing import Any

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.utils.context_utils import count_tokens, load_encoder, message_text


class CompactingConfig(AgentConfig):
    # --- trigger ------------------------------------------------------
    token_budget: int = 80_000
    """Compact once estimated tokens cross this. 0 disables."""
    step_budget: int = 0
    """Compact every N steps. 0 disables."""
    tokenizer_name: str = "cl100k_base"
    """tiktoken encoding name. Falls back to char/4 if tiktoken is unavailable."""

    # --- slice selection ----------------------------------------------
    min_preserve_head: int = 2
    min_preserve_tail: int = 4
    min_compact_size: int = 6
    """Skip compaction unless the compactable slice has at least this many messages."""
    min_compact_tokens: int = 1024
    """Skip compaction unless the compactable slice has at least this many tokens."""
    max_compactions: int = 0
    """Maximum compactions per rollout (0 = unlimited). After the cap, the
    context grows unbounded until the serving window rejects it — matching
    arXiv:2607.05378's 'at most three compaction operations per rollout'."""

    # --- summary call ---------------------------------------------------
    summary_max_tokens: int = 2048
    """Target length of the brief (prompted); longer bodies are injected unchanged but logged."""
    summary_max_completion_tokens: int = 4096
    """Hard ``max_tokens`` for the summary generation, reasoning included. 0 = model default."""
    summary_max_attempts: int = 2
    """Summary generations that are cut off, empty or all reasoning are retried this many times
    in total; after that the extractive fallback below is injected instead."""
    fallback_summary_template: str = (
        "## Goal\nSee the task statement above.\n## Commands run in the compacted span\n{{ commands }}\n"
        "## Last observation\n{{ last_observation }}\n## Next action\n"
        "Re-inspect the current state of the files above before editing.\n"
        "(automatic extractive summary: the model-written summary was unusable)"
    )
    summary_inline_template: str = (
        "Your context is about to be compacted: every message between the task "
        "statement and the {{ min_preserve_tail }} most recent messages will be "
        "REPLACED by the summary you write now. The recent messages are retained "
        "verbatim, so focus on what is about to be lost. Write a condensed "
        "handoff brief under {{ summary_max_tokens }} tokens using this structure:\n"
        "## Goal\n## Plan (checked / remaining)\n## Files touched\n"
        "## Last error / state\n## Next action\n"
        "Preserve verbatim: file paths, exact symbol names, failing test IDs, "
        "assertion messages. Do NOT emit a bash command block this turn."
    )
    resumption_template: str = (
        "<context_compacted n=\"{{ n_compactions }}\">\n{{ summary }}\n"
        "</context_compacted>"
    )


class CompactingAgent(DefaultAgent):
    def __init__(self, model: Model, env: Environment, *, config_class: type = CompactingConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.compactions: list[dict] = []
        self._encoder = load_encoder(self.config.tokenizer_name)

    def _count_tokens(self, messages: list[dict]) -> int:
        return count_tokens(messages, self._encoder)

    def _render_extra(self, template: str, **extra: Any) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars(**extra))

    def step(self) -> list[dict]:
        self._maybe_compact_on_budget()
        return super().step()

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
        if self.config.max_compactions and len(self.compactions) >= self.config.max_compactions:
            return
        lo = self.config.min_preserve_head
        hi = len(self.messages) - self.config.min_preserve_tail
        if hi - lo < self.config.min_compact_size:
            return
        if self._count_tokens(self.messages[lo:hi]) < self.config.min_compact_tokens:
            return
        self._compact_slice(lo, hi)

    def _compact_slice(self, lo: int, hi: int) -> None:
        """Replace ``messages[lo:hi]`` with a single summary-bearing user message."""
        assert 0 <= lo < hi <= len(self.messages), f"bad slice [{lo}, {hi}] of {len(self.messages)}"

        slice_msgs = list(self.messages[lo:hi])
        replaced_tokens = self._count_tokens(slice_msgs)

        summary, summary_call_info, summary_exchange = self._run_summary_call_inline(slice_msgs)

        compaction_id = f"c{len(self.compactions) + 1}"
        resumption = self._render_extra(
            self.config.resumption_template,
            summary=summary,
            n_compactions=len(self.compactions) + 1,
            compaction_id=compaction_id,
        )
        replacement_msg = self.model.format_message(
            role="user",
            content=resumption,
            extra={"source": "compaction", "compaction_id": compaction_id, "kind": "compact"},
        )

        self.messages[lo:hi] = [replacement_msg]
        self.compactions.append({
            "compaction_id": compaction_id,
            "kind": "compact",
            "replaced_range": [lo, hi],
            "replaced_token_count": replaced_tokens,
            "summary_message_index": lo,
            "summary_call": summary_call_info,
            "summary_token_count": self._count_tokens([replacement_msg]),
            "summary_exchange": summary_exchange,
        })
        self.logger.info(
            "compacted [%d, %d): %d -> %d tokens",
            lo, hi, replaced_tokens, self.compactions[-1]["summary_token_count"],
        )

    def _run_summary_call_inline(self, slice_msgs: list[dict]) -> tuple[str, dict, list[dict]]:
        """Elicit the summary as an appended turn of the LIVE conversation.

        ``self.messages`` is never mutated with the elicitation: the prompt list
        is built ad hoc, so the agent loop never sees a parseless assistant
        turn and the FormatError-retry path cannot fire on the free-form
        summary. The raw ``_query`` transport bypasses the action parser; at
        the HTTP layer the call is a prefix-extension of the running segment,
        which is what makes the trained row structure 1+N instead of 1+2N.

        The generation is hard-capped at ``summary_max_completion_tokens``. A
        response that was cut off, is empty, or never leaves its reasoning is
        unusable as context and is retried; after ``summary_max_attempts`` the
        extractive fallback is injected instead. The completion cap bounds the
        injected body (worst case ~summary_max_completion_tokens of the budget);
        its length is recorded as ``body_tokens`` for monitoring, not truncated,
        because cutting a structured brief drops its "Next action" section.
        """
        elicit = self.model.format_message(
            role="user",
            content=self._render_extra(self.config.summary_inline_template),
            extra={"source": "summary_elicitation"},
        )
        derived = self.messages + [elicit]
        kwargs = {"max_tokens": self.config.summary_max_completion_tokens} if self.config.summary_max_completion_tokens else {}

        cost, reason = 0.0, ""
        for attempt in range(1, self.config.summary_max_attempts + 1):
            self.n_calls += 1
            summary_text, cost_i, timestamp, finish_reason = self._summary_query(derived, **kwargs)
            cost += cost_i
            body = self._strip_reasoning(summary_text)
            reason = self._summary_failure(summary_text, body, finish_reason)
            if not reason:
                break
            self.logger.warning("summary attempt %d unusable (%s)", attempt, reason)
        self.cost += cost

        # The recorded turn keeps the raw generation (training rows must match
        # what the model emitted); only the post-reasoning body goes into the
        # <context_compacted> block.
        summary_msg = self.model.format_message(role="assistant", content=summary_text, extra={"source": "summary"})
        if reason:
            body = self._extractive_summary(slice_msgs)
        info = {
            "cost": cost,
            "timestamp": timestamp,
            "mode": "inline",
            "attempts": attempt,
            "fallback": bool(reason),
            "finish_reason": finish_reason,
            "prompt_tokens": self._count_tokens(derived),
            "completion_tokens": self._count_tokens([summary_msg]),
            "body_tokens": self._count_tokens([{"role": "user", "content": body}]),
        }
        if info["body_tokens"] > self.config.summary_max_tokens:
            self.logger.warning("summary body %d tokens exceeds summary_max_tokens=%d", info["body_tokens"], self.config.summary_max_tokens)
        return body, info, [elicit, summary_msg]

    def _summary_query(self, derived: list[dict], **kwargs) -> tuple[str, float, Any, str | None]:
        """One summary generation -> (text, cost, timestamp, finish_reason)."""
        if hasattr(self.model, "_query"):
            try:
                raw = self.model._query(self.model._prepare_messages_for_api(derived), **kwargs)
                return raw.choices[0].message.content or "", 0.0, None, raw.choices[0].finish_reason
            except Exception as e:
                self.logger.warning("inline summary _query failed (%s); falling back to query", e)
        return self._wrapped_summary_query(derived, **kwargs)

    @staticmethod
    def _summary_failure(raw: str, body: str, finish_reason: str | None) -> str:
        if finish_reason == "length":
            return "cut off at max_tokens"
        if not body:
            return "empty"
        if "<think>" in raw and "</think>" not in raw:
            return "all reasoning"
        return ""

    def _strip_reasoning(self, text: str) -> str:
        # Thinking models pre-open <think> via the generation prompt, so raw
        # content may start with deliberation without an opening tag.
        return text.rsplit("</think>", 1)[-1].strip()

    def _extractive_summary(self, slice_msgs: list[dict]) -> str:
        """Deterministic stand-in for an unusable model summary: commands run + last observation."""
        commands: list[str] = []
        for m in slice_msgs:
            if m.get("role") != "assistant":
                continue
            for a in m.get("extra", {}).get("actions", []) or []:
                c = " ".join(a.get("command", "").split())[:200]
                if c and c not in commands:
                    commands.append(c)
        observations = [message_text(m) for m in slice_msgs if m.get("role") == "user"]
        return self._render_extra(
            self.config.fallback_summary_template,
            commands="\n".join(f"- {c}" for c in commands[-25:]) or "- (none)",
            last_observation=(observations[-1] if observations else "")[-1500:],
        )

    def _wrapped_summary_query(self, derived: list[dict], **kwargs) -> tuple[str, float, Any, str | None]:
        """Summary call via the parsing ``query`` path, harvesting FormatError."""
        from minisweagent.exceptions import FormatError
        from minisweagent.models.utils.content_string import get_content_string

        try:
            response = self.model.query(derived, **kwargs)
            extra = response.get("extra", {})
            return get_content_string(response) or "", extra.get("cost", 0.0), extra.get("timestamp"), extra.get("finish_reason")
        except FormatError as fe:
            # The summary response was free-form text; harvest it from the
            # FormatError's attached message instead of erroring.
            self.logger.info("summary response was free-form text; harvesting from FormatError")
            for msg in fe.messages:
                c = get_content_string(msg) or ""
                if c:
                    return c, 0.0, None, None
            return "", 0.0, None, None

    def serialize(self, *extra_dicts) -> dict:
        data = super().serialize(*extra_dicts)
        data["trajectory_format"] = "mini-swe-agent-compact-1"
        data["compactions"] = list(self.compactions)
        data.setdefault("info", {})["context_management"] = {
            "strategy": "compact",
            "token_budget": self.config.token_budget,
            "n_compactions": len(self.compactions),
        }
        return data
