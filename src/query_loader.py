"""
Parses a .sql file containing multiple `-- Qxx` labeled statements
(the shape of tpc-h-queries.sql) into a list of (label, sql) pairs.

Kept separate from collect_training_data.py so the parsing logic is
independently testable without touching a database.
"""

from __future__ import annotations

import re

_LABEL_RE = re.compile(r"^--\s*(Q\w+)\s*$", re.MULTILINE)


def load_queries_from_file(path: str) -> list[tuple[str, str]]:
    with open(path, "r") as f:
        text = f.read()

    matches = list(_LABEL_RE.finditer(text))
    if not matches:
        # No -- Qxx labels found -- treat the whole file as one query,
        # or split naively on blank-line-separated statements.
        statements = [s.strip() for s in text.split(";") if s.strip()]
        return [(f"stmt_{i+1}", s) for i, s in enumerate(statements)]

    queries: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        label = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]

        # Strip separator comment lines and leading/trailing whitespace.
        lines = [
            ln for ln in chunk.splitlines()
            if not re.match(r"^\s*--\s*-+\s*$", ln)
        ]
        sql = "\n".join(lines).strip()
        sql = sql.rstrip(";").strip()
        if sql:
            queries.append((label, sql))

    return queries
