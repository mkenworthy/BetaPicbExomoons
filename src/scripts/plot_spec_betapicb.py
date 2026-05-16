"""Figure 2: best-fit planet spectrum with molecular templates.

Produces the three-row, two-column figure showing:

* the H2O contribution to the best-fit planet model (top row),
* the 12CO contribution (middle row),
* the star-subtracted, telluric-corrected data with the best-fit model
  overplotted (bottom row).

The figure is split into a left and a right panel, each containing three
spectral orders, with a broken x-axis between them. Each chip is independently
normalised by the median of its data flux to remove an arbitrary scale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke

import paths

DATA_FILE = paths.data / "figure_2_spectrum.txt"
assert DATA_FILE.exists(), f"Data file not found: {DATA_FILE}"
FIGURE_FILE = paths.figures / "spec_betapicb.pdf"
#assert FIGURE_FILE.exists(), f"Figure file not found: {FIGURE_FILE}"

CHIPS_LEFT: tuple[int, ...] = (6, 7, 8)
CHIPS_RIGHT: tuple[int, ...] = (9, 10, 11)
SPECTRUM_YLIM = (0.40, 1.45)


def configure_axes() -> None:
    """Apply the shared axis styling used throughout the paper."""
    plt.rcParams.update(
        {
            "axes.linewidth": 1.5,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
        }
    )


def load_data(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Read the spectrum table and group columns by chip index.

    Returns
    -------
    dict
        Mapping ``chip_index -> {wave, flux, err, model, h2o, co}``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Data file {path} not found. Run prepare_data_figure_2.py first."
        )
    raw = np.loadtxt(path)
    chip = raw[:, 0].astype(int)
    by_chip: dict[int, dict[str, np.ndarray]] = {}
    for chip_id in np.unique(chip):
        mask = chip == chip_id
        by_chip[int(chip_id)] = {
            "wave": raw[mask, 1],
            "flux": raw[mask, 2],
            "err": raw[mask, 3],
            "model": raw[mask, 4],
            "h2o": raw[mask, 5],
            "co": raw[mask, 6],
        }
    return by_chip


def plot_spectrum_panel(
    ax: plt.Axes,
    by_chip: dict[int, dict[str, np.ndarray]],
    chip_ids: tuple[int, ...],
    show_legend: bool,
) -> None:
    """Plot the data and best-fit model for the requested chips on one axis."""
    for i, chip_id in enumerate(chip_ids):
        chip = by_chip[chip_id]
        norm = np.nanmedian(chip["flux"])
        ax.errorbar(
            chip["wave"], chip["flux"] / norm, yerr=chip["err"] / norm,
            fmt="o", markersize=0, lw=0.5, color="black", alpha=0.3,
            label="Data" if (show_legend and i == 0) else None,
        )
        ax.plot(
            chip["wave"], chip["model"] / norm,
            color="lightcoral", lw=1.0, alpha=1.0,
            label="Model" if (show_legend and i == 0) else None,
        )

    if show_legend:
        ax.legend(
            loc="lower left", fontsize=9, ncol=3, frameon=True,
            facecolor="w", edgecolor="k", framealpha=0.7,
            handlelength=1.5, handletextpad=0.5,
        )


def plot_template_panel(
    ax: plt.Axes,
    by_chip: dict[int, dict[str, np.ndarray]],
    chip_ids: tuple[int, ...],
    species: str,
    label: str | None,
) -> tuple[float, float]:
    """Plot one molecular template (column ``species``) for the requested chips.

    Returns the (min, max) of the data shown so that y-limits can be shared
    across the left and right halves of the figure.
    """
    ymin, ymax = np.inf, -np.inf
    for i, chip_id in enumerate(chip_ids):
        chip = by_chip[chip_id]
        template = chip[species]
        ax.plot(chip["wave"], template, color="k", lw=1.0, alpha=0.6)
        ymin = min(ymin, np.nanmin(template))
        ymax = max(ymax, np.nanmax(template))
        if i == 0 and label is not None:
            ax.text(
                0.02, 1.50, label, transform=ax.transAxes, fontsize=12,
                weight="bold", ha="left", va="top", alpha=0.6,
                path_effects=[withStroke(linewidth=2, foreground="w")],
            )

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ymin, ymax


def add_axis_break(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    """Draw the small diagonal lines that mark the broken x-axis."""
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    dx, dy = 0.02, 0.06
    kwargs = dict(color="k", clip_on=False, linewidth=1.5)
    ax_left.plot([1 - dx, 1 + dx], [-dy, dy], transform=ax_left.transAxes, **kwargs)
    ax_right.plot([-dx, dx], [-dy, dy], transform=ax_right.transAxes, **kwargs)


def plot_figure(by_chip: dict[int, dict[str, np.ndarray]]) -> plt.Figure:
    """Assemble the full Figure 2 layout."""
    fig, ax = plt.subplots(
        3, 2, figsize=(12, 2.5),
        gridspec_kw={
            "height_ratios": [1, 1, 4],
            "hspace": 0.01,
            "wspace": 0.020,
            "bottom": 0.18,
        },
        sharex=False,
    )
    ax_h2o, ax_co, ax_spec = ax[0, :], ax[1, :], ax[2, :]

    columns = ((ax_h2o[0], ax_co[0], ax_spec[0], CHIPS_LEFT),
               (ax_h2o[1], ax_co[1], ax_spec[1], CHIPS_RIGHT))

    h2o_ylims: list[tuple[float, float]] = []
    co_ylims: list[tuple[float, float]] = []
    for col_idx, (ax_h, ax_c, ax_s, chip_ids) in enumerate(columns):
        h2o_ylims.append(
            plot_template_panel(
                ax_h, by_chip, chip_ids, species="h2o",
                label=r"H$_2$O" if col_idx == 0 else None,
            )
        )
        co_ylims.append(
            plot_template_panel(
                ax_c, by_chip, chip_ids, species="co",
                label=r"CO" if col_idx == 0 else None,
            )
        )
        plot_spectrum_panel(ax_s, by_chip, chip_ids, show_legend=(col_idx == 0))

        wave_min = min(by_chip[c]["wave"].min() for c in chip_ids)
        wave_max = max(by_chip[c]["wave"].max() for c in chip_ids)
        for axi in (ax_h, ax_c, ax_s):
            axi.set_xlim(wave_min, wave_max)
        ax_s.set_ylim(SPECTRUM_YLIM)
        ax_s.spines["top"].set_visible(False)
        ax_s.spines["right"].set_visible(False)

    h2o_lim = (min(y[0] for y in h2o_ylims), max(y[1] for y in h2o_ylims))
    co_lim = (min(y[0] for y in co_ylims), max(y[1] for y in co_ylims))
    for axi in ax_h2o:
        axi.set_ylim(*h2o_lim)
    for axi in ax_co:
        axi.set_ylim(*co_lim)

    add_axis_break(ax_spec[0], ax_spec[1])
    ax_spec[1].set_yticks([])
    ax_spec[0].set_ylabel("Flux [normalized]", fontsize=11)
    fig.text(0.52, 0.01, "Wavelength [nm]", ha="center", va="bottom", fontsize=11)
    return fig


def main() -> None:
    configure_axes()
    by_chip = load_data(DATA_FILE)
    fig = plot_figure(by_chip)
    fig.savefig(FIGURE_FILE, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {FIGURE_FILE}")


if __name__ == "__main__":
    main()
