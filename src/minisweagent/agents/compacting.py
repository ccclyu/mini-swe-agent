"""Compacting agent — Cursor-style flat self-summarization.

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
from minisweagent.agents.utils.context_utils import count_tokens, load_encoder


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

    # --- summary call ---------------------------------------------------
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

        summary, summary_call_info = self._run_summary_call(slice_msgs)

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
        })
        self.logger.info(
            "compacted [%d, %d): %d -> %d tokens",
            lo, hi, replaced_tokens, self.compactions[-1]["summary_token_count"],
        )

    def _run_summary_call(self, slice_msgs: list[dict]) -> tuple[str, dict]:
        """Ask the model for a summary over ``slice_msgs``.

        Bypasses the model's action parser: the summary response is free-form
        text, so running it through the bash/tool-call parser would raise
        FormatError on every valid summary. We call ``model._query`` directly
        (raw litellm call) and extract the content ourselves.
        """
        from minisweagent.models.utils.content_string import get_content_string

        sys_text = self._render_extra(self.config.summary_system_template)
        user_text = self._render_extra(self.config.summary_user_template)
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
                summary_text, cost, timestamp = self._wrapped_summary_query(derived)
        else:
            summary_text, cost, timestamp = self._wrapped_summary_query(derived)

        self.cost += cost
        return summary_text, {
            "cost": cost,
            "timestamp": timestamp,
            "prompt_tokens": self._count_tokens(derived),
            "completion_tokens": self._count_tokens(
                [self.model.format_message(role="assistant", content=summary_text)]
            ),
        }

    def _wrapped_summary_query(self, derived: list[dict]) -> tuple[str, float, Any]:
        """Summary call via the parsing ``query`` path, harvesting FormatError."""
        from minisweagent.exceptions import FormatError
        from minisweagent.models.utils.content_string import get_content_string

        try:
            response = self.model.query(derived)
            return (
                get_content_string(response) or "",
                response.get("extra", {}).get("cost", 0.0),
                response.get("extra", {}).get("timestamp"),
            )
        except FormatError as fe:
            # The summary response was free-form text; harvest it from the
            # FormatError's attached message instead of erroring.
            self.logger.info("summary response was free-form text; harvesting from FormatError")
            for msg in fe.messages:
                c = get_content_string(msg) or ""
                if c:
                    return c, 0.0, None
            return "", 0.0, None

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
