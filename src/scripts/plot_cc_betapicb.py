"""Figure 1: cross-correlation functions of beta Pic b.

Produces the two-panel figure showing:

* (a) the normalised total CCF together with the autocorrelation of the
       v sin i = 0 km/s template,
* (b) the CCFs of the H2O- and 12CO-only templates compared with the
       full (H2O + CO) template.
"""
from __future__ import annotations

#from ddpathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paths

DATA_FILE = paths.data / "figure_1_ccf.txt"
FIGURE_FILE = paths.figures / "cc_betapicb.pdf"

RV_LIM_KMS = (-200.0, 200.0)


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


def load_data(path: Path) -> dict[str, np.ndarray]:
    """Read the plain-text CCF table written by ``prepare_data_figure_1.py``."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data file {path} not found. Run prepare_data_figure_1.py first."
        )
    rv, ccf, acf, ccf_h2o, ccf_co = np.loadtxt(path, unpack=True)
    return {"rv": rv, "ccf": ccf, "acf": acf, "ccf_h2o": ccf_h2o, "ccf_co": ccf_co}


def plot_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    """Build the two-panel cross-correlation figure."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.8, 5.5))

    ax_top.plot(
        data["rv"],
        data["ccf"] / np.nanmax(data["ccf"]),
        color="k", linestyle="-",
        label="CCF\n" + r"$v\sin{i}=19$",
    )
    ax_top.plot(
        data["rv"],
        data["acf"] / np.nanmax(data["acf"]),
        color="k", linestyle=":",
        label="ACF\n" + r"$v\sin{i}=0$",
    )
    ax_top.set_ylabel("CCF [normalized]")

    ax_bot.plot(data["rv"], data["ccf"], color="k", linestyle="-", alpha=0.5,
                label=r"H$_2$O + CO")
    ax_bot.plot(data["rv"], data["ccf_h2o"], color="k", linestyle="--",
                label=r"H$_2$O")
    ax_bot.plot(data["rv"], data["ccf_co"], color="k", linestyle=":",
                label=r"CO")
    ax_bot.set_ylabel("CCF [signal-to-noise ratio]")

    for i, axi in enumerate((ax_top, ax_bot)):
        axi.legend(
            loc=(0.58 + 0.01 * i, 0.60 + 0.04 * i),
            frameon=False, ncol=1, handlelength=1.5, handletextpad=0.5,
        )
        axi.set_xlabel("Velocity [km/s]")
        axi.set_xlim(RV_LIM_KMS)
        axi.text(
            0.02, 0.98, "ab"[i], transform=axi.transAxes,
            fontsize=12, weight="bold", ha="left", va="top",
        )

    fig.tight_layout()
    return fig


def main() -> None:
    configure_axes()
    data = load_data(DATA_FILE)
    fig = plot_figure(data)
    fig.savefig(FIGURE_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {FIGURE_FILE}")


if __name__ == "__main__":
    main()
