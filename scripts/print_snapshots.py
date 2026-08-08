#!/usr/bin/env python3
"""Pretty-print a snapshots.jsonl produced by smoke_test_snapshots.py.

Each line is one moment in the rollout. We print a short header per
snapshot and the message list in a compact form, truncating long contents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def truncate(text: str, n: int = 120) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= n:
        return text
    return text[:n] + f"…(+{len(text) - n} chars)"


def render(snap: dict, content_chars: int) -> str:
    lines = [
        f"### {snap['event']:<15} "
        f"| n_calls={snap['n_calls']} n_compactions={snap['n_compactions']} "
        f"| {len(snap['messages'])} msgs",
    ]
    if snap["event"] in {"before_compact", "after_compact"}:
        for k in ("slice", "compaction_id", "replaced_range",
                  "replaced_token_count", "summary_token_count"):
            if k in snap:
                lines.append(f"   {k}: {snap[k]}")
    for i, m in enumerate(snap["messages"]):
        role = m.get("role") or m.get("type") or "?"
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        lines.append(f"   [{i:>2}] {role:<9}: {truncate(content, content_chars)}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    p.add_argument("--chars", type=int, default=140,
                   help="Max chars per message content (default 140)")
    p.add_argument("--only", nargs="*", default=None,
                   help="Filter events (e.g. --only before_compact after_compact)")
    args = p.parse_args()

    with args.path.open() as f:
        snaps = [json.loads(line) for line in f if line.strip()]

    if args.only:
        snaps = [s for s in snaps if s["event"] in set(args.only)]

    for i, snap in enumerate(snaps):
        print(f"\n{'=' * 78}\nSNAPSHOT #{i}  [{snap['event']}]")
        print(render(snap, args.chars))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
