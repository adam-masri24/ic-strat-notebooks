"""Exports each executed notebook to a Markdown page with its charts as PNG files, so GitHub shows the charts.

GitHub's notebook viewer gives up on large notebooks and shows the raw file instead; a Markdown page with image files
always renders. Run from this folder after executing the notebooks:

    python _export_markdown.py

It writes ``<notebook>.md`` and ``<notebook>_files/`` next to each notebook. The pages leave out the code cells (the
code is in the ``.ipynb`` and in ``analysis/``), escape ``$`` so GitHub doesn't read dollar amounts as maths, and drop
pandas' ``<style>`` blocks, which GitHub would show as text.
"""

import copy
import re
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter

NOTEBOOKS = ["ic_analysis.ipynb", "correlation_mo.ipynb"]
HERE = Path(__file__).resolve().parent


def _escape_dollars(text: str) -> str:
    """Escape $ outside code spans so GitHub's Markdown doesn't treat dollar amounts as inline maths."""
    parts = re.split(r"(`[^`]*`)", text)
    return "".join(p if p.startswith("`") else re.sub(r"(?<!\\)\$", r"\\$", p) for p in parts)


def export(name: str) -> Path:
    nb = nbformat.read(HERE / name, as_version=4)
    nb = copy.deepcopy(nb)
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = _escape_dollars(cell.source)
    stem = Path(name).stem
    exporter = MarkdownExporter(exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    body, resources = exporter.from_notebook_node(nb, resources={"output_files_dir": f"{stem}_files"})
    out_dir = HERE / f"{stem}_files"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("*"):
        old.unlink()
    for path, data in resources.get("outputs", {}).items():
        (HERE / path).write_bytes(data)
    # GitHub shows <style> blocks (pandas' table CSS) as text: drop them, and the blank lines they leave inside tables.
    body = re.sub(r"<style[^>]*>.*?</style>\s*", "", body, flags=re.DOTALL)
    md_path = HERE / f"{stem}.md"
    md_path.write_text(body, encoding="utf-8")
    return md_path


if __name__ == "__main__":
    for nb_name in NOTEBOOKS:
        p = export(nb_name)
        print(f"wrote {p.name} and {p.stem}_files/ ({len(list((HERE / f'{p.stem}_files').glob('*')))} images)")
