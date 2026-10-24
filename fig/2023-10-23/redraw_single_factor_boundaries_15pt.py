"""Render the full single-factor boundary figure with 15 pt text."""

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pcdh_lstm_aekf.plot_major_revision import (  # noqa: E402
    FULL_FACTOR_BOUNDARY_PANELS,
    _draw_factor_panel,
)
from pcdh_lstm_aekf.plot_monte_carlo import _configure_matplotlib  # noqa: E402

sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


def draw() -> None:
    with (ROOT / "artifacts/major_revision_v9/single_factor_scan.json").open(
        encoding="utf-8"
    ) as handle:
        factor_data = json.load(handle)

    _configure_matplotlib()
    mpl.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "svg.fonttype": "none",
    })
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 11.3))
    figure.subplots_adjust(left=0.105, right=0.985, top=0.89, bottom=0.10,
                           wspace=0.26, hspace=0.36)
    for axis, (factor, title) in zip(axes.flat, FULL_FACTOR_BOUNDARY_PANELS):
        _draw_factor_panel(axis, factor_data, factor, title)
        axis.set_title(title, loc="left", fontsize=15, pad=8)
        axis.tick_params(axis="both", labelsize=15)
        axis.xaxis.label.set_fontsize(15)
        axis.yaxis.label.set_fontsize(15)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=15)
    figure.suptitle(
        "Full single-factor performance boundaries: "
        "5 fixed training instances × 30 shared test seeds",
        fontsize=15, y=0.98,
    )
    figure.canvas.draw()
    stem = "A1_single_factor_boundaries_full_15pt"
    require_matplotlib_panel_alignment(figure, json_out=HERE / f"{stem}.alignment.json")
    for suffix in ("png", "pdf", "svg"):
        kwargs = {"facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 600
        figure.savefig(HERE / f"{stem}.{suffix}", **kwargs)
    plt.close(figure)


if __name__ == "__main__":
    draw()
