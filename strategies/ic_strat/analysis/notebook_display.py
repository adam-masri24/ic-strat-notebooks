"""Compact chart output for the notebooks, so GitHub renders them.

GitHub stops rendering a notebook once the file gets large (it falls back to showing the raw file), and
almost all of these notebooks' size is chart images. ``compact_figures()`` makes every matplotlib figure
display as an 8-bit palette PNG instead of a full-colour one: the charts use a handful of flat colours,
so they look the same at about a third of the size.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
from PIL import Image


def _palette_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi="figure", bbox_inches="tight", facecolor=fig.get_facecolor())
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    im = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()


def compact_figures() -> None:
    """Display every matplotlib figure in this kernel as a palette PNG (call once, after importing pyplot)."""
    from IPython import get_ipython

    ip = get_ipython()
    if ip is None:
        return
    ip.run_line_magic("matplotlib", "inline")   # activate the backend now, so it doesn't override us at the first plot

    def _register(*_):
        ip.display_formatter.formatters["image/png"].for_type(plt.Figure, _palette_png)

    # The inline backend (re)installs its own PNG formatter when it activates, so re-apply before every cell.
    _register()
    ip.events.register("pre_run_cell", _register)
