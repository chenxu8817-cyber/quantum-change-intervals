"""Generate the three publication figures for the Quantum Paper-I manuscript."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".local_pydeps"
if os.environ.get("QCI_USE_LOCAL_DEPS") == "1" and LOCAL_DEPS.exists():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(".local_pydeps is a CPython 3.12 cache")
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from quantum_interval_numerics import p1


BLUE = "#0F4D92"
LIGHT_BLUE = "#3775BA"
GREEN = "#278C64"
ORANGE = "#D97904"
RED = "#B64342"
PURPLE = "#7A5195"
GRAY = "#767676"
LIGHT_GRAY = "#E7E7E7"


def apply_publication_style() -> None:
    """Apply a compact, print-safe style suitable for quantumarticle."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.4,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def fixed_length_limit_curve(c_values: np.ndarray, length: int) -> np.ndarray:
    """Vectorized Gauss--Legendre evaluation of P_infty(i,c^2)."""
    c_values = np.asarray(c_values, dtype=float)
    if length < 1 or np.any((c_values < 0.0) | (c_values > 1.0)):
        raise ValueError("require length >= 1 and c in [0,1]")
    nodes, weights = np.polynomial.legendre.leggauss(512)
    theta = math.pi * (nodes + 1.0)
    r = c_values[:, None] ** 2
    symbol = np.broadcast_to(1.0 - r**length, (len(c_values), len(theta))).copy()
    for distance in range(1, length):
        symbol += (
            2.0
            * (r**distance - r**length)
            * np.cos(distance * theta)[None, :]
        )
    means = 0.5 * (np.sqrt(np.maximum(symbol, 0.0)) @ weights)
    values = means**2
    values[c_values == 0.0] = 1.0
    values[c_values == 1.0] = 0.0
    return values


def _read_numeric_csv(path: Path, required: set[str]) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows: list[dict[str, float | str]] = []
        for source in reader:
            row: dict[str, float | str] = {}
            for key, value in source.items():
                if key in {"schedule", "method", "solver", "sdp_status"}:
                    row[key] = value
                else:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return [pdf, png]


def make_figure_1(output_dir: Path) -> list[Path]:
    """Draw the returning interval and collective-localization task."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 2.65))

    # A sparse sequence makes the two returning boundaries visible without
    # asking the reader to count sites.  State labels and fill patterns provide
    # redundant encoding, so the model remains legible in grayscale.
    y_sequence = 0.72
    reference_x = [0.0, 0.82, 2.43, 7.15, 8.76]
    anomaly_x = [3.52, 4.34, 5.97]
    ax.plot([reference_x[0], reference_x[-1]], [y_sequence, y_sequence],
            color=GRAY, lw=1.0, zorder=0)
    ax.scatter(reference_x, [y_sequence] * len(reference_x), s=325,
               facecolor=LIGHT_GRAY, edgecolor="black", lw=0.7, zorder=2)
    ax.scatter(anomaly_x, [y_sequence] * len(anomaly_x), s=325,
               facecolor=BLUE, edgecolor="black", lw=0.7, zorder=2)
    for x in reference_x:
        ax.text(x, y_sequence, r"$0$", ha="center", va="center",
                fontsize=8.5, color="black", zorder=3)
    for x in anomaly_x:
        ax.text(x, y_sequence, r"$\psi$", ha="center", va="center",
                fontsize=8.5, color="white", zorder=3)

    # Ellipses indicate arbitrary numbers of omitted systems.
    for x in (1.63, 5.16, 7.96):
        ax.text(x, y_sequence + 0.02, r"$\cdots$", ha="center", va="center",
                fontsize=12, color=GRAY)

    position_labels = {
        0.0: r"$1$",
        2.43: r"$a-1$",
        3.52: r"$a$",
        5.97: r"$b$",
        7.15: r"$b+1$",
        8.76: r"$n$",
    }
    for x, label in position_labels.items():
        ax.text(x, 0.31, label, ha="center", va="top", fontsize=7.7,
                color=GRAY)

    ax.text(1.22, 1.40, r"reference state $|0\rangle$", ha="center",
            fontsize=8.6, color="black")
    ax.text(4.76, 1.40, r"anomalous state $|\psi\rangle$", ha="center",
            fontsize=8.6, color=BLUE, weight="bold")
    ax.text(7.96, 1.40, r"reference state $|0\rangle$", ha="center",
            fontsize=8.6, color="black")

    # Dashed boundaries distinguish the entry and return locations from the
    # occupied sites themselves.
    for x, label in ((2.98, r"entry at $a$"), (6.56, r"return at $b+1$")):
        ax.plot([x, x], [0.43, 1.15], color=ORANGE, lw=1.15,
                ls=(0, (3, 2)), zorder=1)
        ax.text(x, 1.08, label, ha="center", va="bottom", fontsize=7.5,
                color=ORANGE, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25})

    ax.annotate("", xy=(6.25, 0.02), xytext=(3.24, 0.02),
                arrowprops={"arrowstyle": "|-|", "lw": 1.25, "color": BLUE})
    ax.text(4.75, -0.18, r"interval $I=[a,b]$, length $i=b-a+1$",
            ha="center", va="top", fontsize=8.0, color=BLUE)

    # The inference layer states explicitly that the full sequence is measured
    # jointly and that the output is an exact interval estimate.
    measurement = FancyBboxPatch(
        (2.78, -1.28), 3.30, 0.48,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor="#E8F1F8", edgecolor=BLUE, linewidth=1.0,
    )
    estimate = FancyBboxPatch(
        (7.02, -1.28), 2.20, 0.48,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor="#FFF1D2", edgecolor=ORANGE, linewidth=1.0,
    )
    ax.add_patch(measurement)
    ax.add_patch(estimate)
    ax.annotate("", xy=(4.43, -0.77), xytext=(4.43, -0.42),
                arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": GRAY})
    ax.annotate("", xy=(6.98, -1.04), xytext=(6.14, -1.04),
                arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": GRAY})
    ax.text(4.43, -1.04, r"collective POVM $\{M_I\}$ on all $n$ systems",
            ha="center", va="center", fontsize=7.8, color=BLUE)
    ax.text(8.12, -1.04, r"estimate $\widehat I=[\widehat a,\widehat b]$",
            ha="center", va="center", fontsize=7.8, color="#955100")

    ax.set(xlim=(-0.48, 9.72), ylim=(-1.52, 1.63))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    return _save(fig, output_dir, "figure1_model_geometry")


def make_figure_2(output_dir: Path) -> list[Path]:
    """Plot the fixed-, growing-, and unknown-length analytic limits."""
    apply_publication_style()
    c_values = np.linspace(0.0, 1.0, 301)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    colors = [GRAY, ORANGE, GREEN, LIGHT_BLUE, PURPLE]
    styles = [":", "--", "-.", "-", (0, (5, 1))]
    for length, color, style in zip((1, 2, 4, 8, 16), colors, styles):
        axes[0].plot(
            c_values,
            fixed_length_limit_curve(c_values, length),
            color=color,
            ls=style,
            label=rf"$i={length}$",
        )
    long_limit = np.array([p1(c * c) for c in c_values])
    unknown_limit = np.array([p1(c) ** 2 for c in c_values])
    axes[0].plot(c_values, long_limit, color="black", lw=2.3,
                 label=r"$p_1(c^2)$")
    axes[0].set(
        xlabel=r"state overlap $c=|\langle0|\psi\rangle|$",
        ylabel=r"asymptotic success probability",
        title=r"(a) $P_\infty(i,c^2)\to p_1(c^2)$",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[0].legend(ncol=2, loc="lower left")

    axes[1].plot(c_values, long_limit, color=BLUE, lw=2.3,
                 label=r"known growing length: $p_1(c^2)$")
    axes[1].plot(c_values, unknown_limit, color=RED, lw=2.3, ls="--",
                 label=r"unknown length: $p_1(c)^2$")
    axes[1].fill_between(c_values, unknown_limit, long_limit,
                         color=LIGHT_BLUE, alpha=0.14, linewidth=0)
    axes[1].set(
        xlabel=r"state overlap $c=|\langle0|\psi\rangle|$",
        ylabel=r"asymptotic success probability",
        title="(b) One translation versus two endpoints",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[1].legend(loc="lower left")
    for ax in axes:
        ax.grid(axis="y", color=LIGHT_GRAY, lw=0.6)
    fig.tight_layout(w_pad=2.2)
    return _save(fig, output_dir, "figure2_analytic_limits")


def make_figure_3(
    output_dir: Path,
    sdp_data_path: Path = ROOT / "certified_sdp_results.csv",
    srm_data_path: Path = ROOT / "srm_scaling_m1.csv",
) -> list[Path]:
    """Plot finite-size convergence and the certified SRM-optimum gap."""
    apply_publication_style()
    known_rows = _read_numeric_csv(
        ROOT / "paper1" / "paper1_fixed_and_growing_srm.csv",
        {"schedule", "c", "N", "srm", "target"},
    )
    unknown_rows = _read_numeric_csv(
        srm_data_path,
        {"n", "c", "srm", "target_p1_power_2m"},
    )
    sdp_rows = _read_numeric_csv(
        sdp_data_path,
        {
            "n",
            "c",
            "srm",
            "sdp_lower_minus_srm",
            "sdp_upper_minus_srm",
        },
    )

    # Match the final two-column width instead of drawing oversized and then
    # shrinking in LaTeX; the latter made all labels roughly 30% smaller.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    parameter_colors = {0.5: GREEN, 0.8: BLUE, 0.95: RED, 0.99: PURPLE}
    parameter_markers = {0.5: "o", 0.6: "v", 0.8: "s", 0.95: "^", 0.99: "D"}

    ax = axes[0]
    for c in (0.8, 0.95):
        color = parameter_colors[c]
        for schedule, marker, style, label_name in (
            ("sqrt_growth", "o", "-", r"$i=\lceil\sqrt{N}\rceil$"),
            ("balanced_growth", "s", "--", r"$i=\lfloor N/2\rfloor$"),
        ):
            rows = sorted(
                (row for row in known_rows
                 if row["schedule"] == schedule and math.isclose(float(row["c"]), c)),
                key=lambda row: float(row["N"]),
            )
            ax.plot(
                [float(row["N"]) for row in rows],
                [float(row["srm"]) for row in rows],
                color=color,
                marker=marker,
                ms=4,
                ls=style,
                label=rf"$c={c}$, {label_name}",
            )
        target = next(
            float(row["target"]) for row in known_rows
            if row["schedule"] == "sqrt_growth" and math.isclose(float(row["c"]), c)
        )
        ax.axhline(target, color=color, lw=1.0, alpha=0.55, ls=":")
    ax.set(
        xlabel=r"number of translations $N$",
        ylabel=r"$P_{\rm SRM}$",
        title="(a) Known growing length",
        xticks=[10, 20, 40, 80],
    )
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.99, 0.57),
        fontsize=5.9,
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRAY,
        framealpha=0.96,
        borderpad=0.35,
    )

    ax = axes[1]
    for c in (0.5, 0.8, 0.95, 0.99):
        rows = sorted(
            (row for row in unknown_rows if math.isclose(float(row["c"]), c)),
            key=lambda row: float(row["n"]),
        )
        color = parameter_colors[c]
        ax.plot(
            [float(row["n"]) for row in rows],
            [float(row["srm"]) for row in rows],
            color=color,
            marker=parameter_markers[c],
            ms=3.7,
            label=rf"$c={c}$",
        )
        ax.axhline(float(rows[0]["target_p1_power_2m"]),
                   color=color, lw=0.9, alpha=0.55, ls=":")
    ax.set(
        xlabel=r"sequence length $n$",
        ylabel=r"$P_{\rm SRM}$",
        title=r"(b) Unknown length",
        xticks=[10, 20, 30, 40, 50],
    )
    ax.legend(
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.55, 0.61),
        fontsize=6.2,
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRAY,
        framealpha=0.96,
        borderpad=0.35,
        columnspacing=0.9,
        handlelength=1.8,
    )

    ax = axes[2]
    for c in (0.6, 0.8, 0.95, 0.99):
        rows = sorted(
            (row for row in sdp_rows if math.isclose(float(row["c"]), c)),
            key=lambda row: float(row["n"]),
        )
        lower = np.array(
            [max(float(row["sdp_lower_minus_srm"]), 1e-14) for row in rows]
        )
        upper = np.array(
            [max(float(row["sdp_upper_minus_srm"]), 1e-14) for row in rows]
        )
        midpoint = (lower + upper) / 2.0
        ax.errorbar(
            [float(row["n"]) for row in rows],
            midpoint,
            yerr=np.vstack((midpoint - lower, upper - midpoint)),
            color=parameter_colors.get(c, ORANGE),
            marker=parameter_markers[c],
            ms=3.7,
            lw=1.0,
            capsize=2.0,
            elinewidth=0.8,
            label=rf"$c={c}$",
        )
    ax.set_yscale("log")
    ax.set(
        xlabel=r"sequence length $n$",
        ylabel=r"bounds on $P_{\rm opt}-P_{\rm SRM}$",
        title="(c) Certified gap intervals",
        xticks=[3, 4, 5, 6, 7],
    )
    ax.legend(
        loc="center right",
        bbox_to_anchor=(0.99, 0.49),
        fontsize=5.9,
        frameon=True,
        facecolor="white",
        edgecolor=LIGHT_GRAY,
        framealpha=0.96,
        borderpad=0.35,
    )

    for ax in axes:
        ax.grid(axis="y", color=LIGHT_GRAY, lw=0.6)
    fig.tight_layout(pad=0.45, w_pad=0.8)
    return _save(fig, output_dir, "figure3_finite_size")


def make_all_figures(
    output_dir: Path = ROOT / "paper1" / "figures",
    sdp_data_path: Path = ROOT / "certified_sdp_results.csv",
    srm_data_path: Path = ROOT / "srm_scaling_m1.csv",
) -> list[Path]:
    """Generate all three figures and return every PDF/PNG path."""
    outputs = []
    outputs.extend(make_figure_1(output_dir))
    outputs.extend(make_figure_2(output_dir))
    outputs.extend(
        make_figure_3(
            output_dir,
            sdp_data_path=sdp_data_path,
            srm_data_path=srm_data_path,
        )
    )
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "paper1" / "figures"
    )
    parser.add_argument(
        "--sdp-data", type=Path, default=ROOT / "certified_sdp_results.csv"
    )
    parser.add_argument(
        "--srm-data", type=Path, default=ROOT / "srm_scaling_m1.csv"
    )
    args = parser.parse_args()
    for generated in make_all_figures(
        output_dir=args.output_dir,
        sdp_data_path=args.sdp_data,
        srm_data_path=args.srm_data,
    ):
        print(generated)
