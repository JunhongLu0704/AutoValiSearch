from __future__ import annotations

from html import escape
from typing import Iterable, Mapping, Sequence


def markdown_table(headers: Sequence[str], rows: Iterable[Mapping[str, object]]) -> str:
    headers = list(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def html_table(headers: Sequence[str], rows: Iterable[Mapping[str, object]]) -> str:
    headers = list(headers)
    parts = ["<table><thead><tr>"]
    for header in headers:
        parts.append(f"<th>{escape(str(header))}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for header in headers:
            parts.append(f"<td>{escape(str(row.get(header, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)

