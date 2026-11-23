"""Render JSON-like values (states, structured instructions, descriptions) as prose.

This is the official CLM client's rendering (contrastive-lm 0.1.0, ``clm.schema.to_text``). Objects
become ``key: value`` fields, with top-level fields separated by a blank line and nested ones indented.
Arrays become one ``- item`` line per element. Key order is preserved. The CLM decider and the
``/v1/systemone`` server share it.
"""

from __future__ import annotations

import json
from typing import Any


def to_text(x: Any, indent: int = 0) -> str:
    """A state or description as prose: ``key: value`` fields for objects, ``- item`` lines for arrays."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int | float):
        return str(x)
    pad = " " * indent
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            if isinstance(v, dict | list) and v:
                parts.append(f"{pad}{k}:\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}{k}: {to_text(v)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(x, list | tuple):
        parts = []
        for v in x:
            if isinstance(v, dict | list) and v:
                parts.append(f"{pad}-\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}- {to_text(v)}")
        return "\n".join(parts)
    return json.dumps(x, ensure_ascii=False)
