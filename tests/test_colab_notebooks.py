"""Hermetic syntax checks for Colab notebook code cells."""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

COLAB_DIR = Path(__file__).resolve().parents[1] / "colab"
LINE_MAGIC = re.compile(r"^\s*[!%]")


def _cell_source(cell: dict) -> str:
    """Return the joined source text for a notebook cell."""
    src = cell.get("source", [])
    if isinstance(src, str):
        return src
    return "".join(src)


def _strip_ipython_for_ast(source: str) -> str:
    """Replace IPython shell/magic lines so ``ast.parse`` can run."""
    lines: list[str] = []
    for line in source.splitlines():
        if LINE_MAGIC.match(line):
            indent = re.match(r"^(\s*)", line).group(1)
            lines.append(f"{indent}pass  # ipython magic stripped")
            continue
        lines.append(line)
    if not lines:
        return ""
    text = "\n".join(lines)
    if source.endswith("\n"):
        text += "\n"
    return text


def _is_bash_cell(source: str) -> bool:
    first = source.splitlines()[0] if source else ""
    return first.strip() == "%%bash"


def _bash_body(source: str) -> str:
    lines = source.splitlines()
    if lines and lines[0].strip() == "%%bash":
        body = "\n".join(lines[1:])
        return body + "\n" if body else ""
    return source


def _bash_literal_backslash_n_lines(body: str) -> list[str]:
    """Lines that use literal ``\\n`` where a ``\\`` continuation was intended."""
    bad: list[str] = []
    for i, line in enumerate(body.splitlines(), 1):
        stripped = line.rstrip()
        if stripped.endswith("\\n") and not stripped.endswith("\\\\n"):
            bad.append(f"line {i}: literal \\n instead of line continuation")
    return bad


def _bash_n_syntax_ok(body: str) -> str | None:
    bash = shutil.which("bash")
    if bash is None:
        return None
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".sh",
        delete=False,
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(body)
        path = handle.name
    try:
        proc = subprocess.run(
            [bash, "-n", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return err or f"bash -n exit {proc.returncode}"
    return None


@pytest.mark.parametrize(
    "notebook_path",
    sorted(COLAB_DIR.glob("*.ipynb")),
    ids=lambda p: p.name,
)
def test_colab_notebook_code_cell_syntax(notebook_path: Path) -> None:
    """Every Colab code cell must parse as Python or valid %%bash shell."""
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        if not source.strip():
            continue
        label = f"{notebook_path.name}:cell{idx}"
        if _is_bash_cell(source):
            body = _bash_body(source)
            literal = _bash_literal_backslash_n_lines(body)
            assert not literal, f"{label}: " + "; ".join(literal)
            bash_err = _bash_n_syntax_ok(body)
            if bash_err is not None:
                pytest.fail(f"{label}: bash -n failed: {bash_err}")
            continue
        py = _strip_ipython_for_ast(source)
        if not py.strip():
            continue
        ast.parse(py, filename=label)
