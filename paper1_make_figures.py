"""Generate the three publication figures for the Quantum Paper-I manuscript."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime, timezone
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
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
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
            "svg.fonttype": "none",
            "svg.hashsalt": "quantum-change-intervals-paper1",
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
    release_timestamp = datetime(2026, 8, 24, tzinfo=timezone.utc)
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Creator": "Quantum Change Intervals reproducibility workflow",
            "CreationDate": release_timestamp,
            "ModDate": release_timestamp,
        },
    )
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return [pdf, png]


def make_figure_1(output_dir: Path) -> list[Path]:
    """Draw the returning interval and collective-localization task."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.2, 2.55))

    # Figure contract: the state sequence is the hero, the interval annotation
    # identifies its two unknown boundaries, and the quieter lower row states
    # the collective decision task.  The fills redundantly encode the two
    # states, so the model remains legible in grayscale.
    ink = "#26323D"
    muted = "#6D7781"
    connector = "#929CA5"
    reference_fill = "#F5F6F7"
    anomaly_fill = "#E7F0F7"
    estimate_fill = "#FFF4E2"

    y_sequence = 0.76
    reference_x = [0.10, 0.96, 2.58, 7.28, 8.92]
    anomaly_x = [3.57, 4.44, 6.04]
    entry_x, return_x = 3.08, 6.68

    # A pale interval rail makes contiguity visually primary without adding a
    # decorative frame.  The sequence baseline remains visible through it.
    interval_rail = FancyBboxPatch(
        (entry_x + 0.04, y_sequence - 0.22),
        return_x - entry_x - 0.08,
        0.44,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=anomaly_fill,
        edgecolor="none",
        zorder=-1,
    )
    ax.add_patch(interval_rail)
    ax.plot(
        [reference_x[0], reference_x[-1]],
        [y_sequence, y_sequence],
        color=connector,
        lw=1.05,
        solid_capstyle="round",
        zorder=0,
    )
    ax.scatter(
        reference_x,
        [y_sequence] * len(reference_x),
        s=285,
        facecolor=reference_fill,
        edgecolor=ink,
        lw=0.75,
        zorder=2,
    )
    ax.scatter(
        anomaly_x,
        [y_sequence] * len(anomaly_x),
        s=285,
        facecolor=BLUE,
        edgecolor=ink,
        lw=0.75,
        zorder=2,
    )
    for x in reference_x:
        ax.text(x, y_sequence, r"$0$", ha="center", va="center",
                fontsize=8.3, color=ink, zorder=3)
    for x in anomaly_x:
        ax.text(x, y_sequence, r"$\psi$", ha="center", va="center",
                fontsize=8.3, color="white", zorder=3)

    # Ellipses indicate arbitrary numbers of omitted systems.
    for x in (1.77, 5.24, 8.10):
        ax.text(
            x,
            y_sequence + 0.02,
            r"$\cdots$",
            ha="center",
            va="center",
            fontsize=11.0,
            color=muted,
        )

    position_labels = {
        0.10: r"$1$",
        2.58: r"$a-1$",
        3.57: r"$a$",
        6.04: r"$b$",
        7.28: r"$b+1$",
        8.92: r"$n$",
    }
    for x, label in position_labels.items():
        ax.text(
            x,
            0.39,
            label,
            ha="center",
            va="top",
            fontsize=7.4,
            color=muted,
        )

    # Direct region labels eliminate a detached legend and establish a clear
    # reference--anomaly--reference reading order.
    ax.text(
        1.34,
        1.47,
        r"reference $|0\rangle$",
        ha="center",
        va="center",
        fontsize=8.1,
        color=ink,
    )
    ax.text(
        4.88,
        1.47,
        r"anomalous $|\psi\rangle$",
        ha="center",
        va="center",
        fontsize=8.2,
        color=BLUE,
        weight="bold",
    )
    ax.text(
        4.88,
        1.30,
        r"state overlap $c=|\langle0|\psi\rangle|$",
        ha="center",
        va="center",
        fontsize=6.5,
        color=muted,
    )
    ax.text(
        8.10,
        1.47,
        r"reference $|0\rangle$",
        ha="center",
        va="center",
        fontsize=8.1,
        color=ink,
    )

    # Dashed boundaries distinguish the entry and return locations from the
    # occupied sites themselves.
    for x, label in (
        (entry_x, r"entry $a$"),
        (return_x, r"return $b+1$"),
    ):
        ax.plot(
            [x, x],
            [0.50, 1.23],
            color=ORANGE,
            lw=1.15,
            ls=(0, (2.5, 2.0)),
            solid_capstyle="round",
            zorder=1,
        )
        ax.text(
            x,
            1.17,
            label,
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#B96500",
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.18},
        )

    # Exact horizontal bracket, avoiding arrowhead ambiguity at the endpoints.
    bracket_y = 0.04
    bracket_left, bracket_right = 3.31, 6.31
    ax.plot(
        [bracket_left, bracket_right],
        [bracket_y, bracket_y],
        color=BLUE,
        lw=1.25,
        solid_capstyle="round",
    )
    for x in (bracket_left, bracket_right):
        ax.plot(
            [x, x],
            [bracket_y - 0.11, bracket_y + 0.11],
            color=BLUE,
            lw=1.25,
            solid_capstyle="round",
        )
    ax.text(
        4.81,
        -0.15,
        r"interval $I=[a,b]$  ·  length $i=b-a+1$",
        ha="center",
        va="top",
        fontsize=7.8,
        color=BLUE,
    )

    # The inference layer states explicitly that the full sequence is measured
    # jointly and that the output is an exact interval estimate.
    measurement = FancyBboxPatch(
        (3.04, -1.34),
        3.55,
        0.52,
        boxstyle="round,pad=0.055,rounding_size=0.065",
        facecolor="#EDF4F9",
        edgecolor=BLUE,
        linewidth=0.95,
    )
    estimate = FancyBboxPatch(
        (7.16, -1.34),
        2.28,
        0.52,
        boxstyle="round,pad=0.055,rounding_size=0.065",
        facecolor=estimate_fill,
        edgecolor=ORANGE,
        linewidth=0.95,
    )
    ax.add_patch(measurement)
    ax.add_patch(estimate)
    ax.annotate(
        "",
        xy=(4.815, -0.78),
        xytext=(4.815, -0.40),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.9,
            "color": connector,
            "mutation_scale": 8,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
    )
    ax.annotate(
        "",
        xy=(7.10, -1.08),
        xytext=(6.65, -1.08),
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.95,
            "color": connector,
            "mutation_scale": 8,
            "shrinkA": 1.5,
            "shrinkB": 1.5,
        },
    )
    ax.text(
        4.815,
        -1.00,
        r"collective POVM $\{M_I\}$",
        ha="center",
        va="center",
        fontsize=7.8,
        color=BLUE,
        weight="bold",
    )
    ax.text(
        4.815,
        -1.20,
        r"jointly on all $n$ systems",
        ha="center",
        va="center",
        fontsize=6.7,
        color=muted,
    )
    ax.text(
        8.30,
        -0.99,
        "interval estimate",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#9B5900",
    )
    ax.text(
        8.30,
        -1.20,
        r"$\widehat I=[\widehat a,\widehat b]$",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#8A4E00",
        weight="bold",
    )

    ax.set(xlim=(-0.42, 9.78), ylim=(-1.52, 1.68))
    ax.axis("off")
    fig.tight_layout(pad=0.18)

    # SVG is retained as an editable-text master; PDF remains the manuscript
    # asset and PNG is the review/README preview.
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "figure1_model_geometry.svg",
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Creator": "Quantum Change Intervals reproducibility workflow",
            "Date": "2026-08-24",
        },
    )
    fig.savefig(
        output_dir / "figure1_model_geometry.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )
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

    axes[1].plot(
        c_values,
        long_limit,
        color=BLUE,
        lw=2.3,
        label=r"Known: uniform translations, $p_1(c^2)$",
    )
    axes[1].plot(
        c_values,
        unknown_limit,
        color=RED,
        lw=2.3,
        ls="--",
        label=r"Unknown: uniform intervals, $p_1(c)^2$",
    )
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
    known_data_path: Path = (
        ROOT / "paper1" / "paper1_fixed_and_growing_srm.csv"
    ),
    sdp_data_path: Path = ROOT / "certified_sdp_results.csv",
    srm_data_path: Path = ROOT / "srm_scaling_m1.csv",
) -> list[Path]:
    """Plot finite-size convergence and safeguarded SRM-gap diagnostics.

    Figure contract: panels (a) and (b) compare finite-size SRM values with
    their theorem-level limits, while panel (c) uses every small-instance SDP
    case to show a conservative upper endpoint for the SRM optimality gap.  The
    high-overlap stress tests are concentrated in panel (b), because the
    available ``c=0.95, i=ceil(sqrt(N))`` known-length data are visibly
    pre-asymptotic on ``N <= 80`` and would otherwise look like a monotone-rate
    claim that the paper does not make.
    """
    apply_publication_style()
    known_rows = _read_numeric_csv(
        known_data_path,
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
    # shrinking in LaTeX.  The novel full-physical-ensemble calculation in
    # panel (b) receives slightly more width than the two validation panels.
    fig = plt.figure(figsize=(7.2, 2.95), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.02, 1.13, 1.04))
    axes = np.array([fig.add_subplot(grid[0, index]) for index in range(3)])
    parameter_colors = {0.5: GREEN, 0.8: BLUE, 0.95: RED, 0.99: PURPLE}
    parameter_markers = {0.5: "o", 0.8: "s", 0.95: "^", 0.99: "D"}

    ax = axes[0]
    # The available c=0.95 square-root schedule is still non-monotone and
    # pre-asymptotic at N <= 80.  The moderate-overlap traces below show the
    # intended finite-size approach without concealing that limitation; panel
    # (b) retains c=0.95 and 0.99 as explicit high-overlap stress tests.
    for c in (0.5, 0.8):
        color = parameter_colors[c]
        for schedule, marker, style, marker_size, marker_face, zorder in (
            ("sqrt_growth", "o", "-", 3.3, color, 4),
            ("balanced_growth", "s", "--", 5.8, "none", 3),
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
                ms=marker_size,
                mfc=marker_face,
                mew=0.9,
                ls=style,
                zorder=zorder,
                label=(
                    f"c={c}, i=⌈√N⌉"
                    if schedule == "sqrt_growth"
                    else f"c={c}, i=⌊N/2⌋"
                ),
            )
        target = next(
            float(row["target"]) for row in known_rows
            if row["schedule"] == "sqrt_growth" and math.isclose(float(row["c"]), c)
        )
        ax.axhline(target, color=color, lw=1.0, alpha=0.55, ls=":")
    ax.set(
        xlabel=r"number of translations $N$",
        ylabel=r"$P_{\rm SRM}$",
        title="Growing known length",
        xticks=[10, 20, 40, 80],
        xlim=(7, 84),
        ylim=(0.74, 0.985),
    )
    ax.legend(
        loc="center",
        bbox_to_anchor=(0.62, 0.55),
        prop={"family": "DejaVu Sans", "size": 6.35},
        frameon=False,
        borderpad=0.15,
        labelspacing=0.28,
        handlelength=2.25,
        handletextpad=0.45,
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
        ax.annotate(
            rf"$c={c}$",
            xy=(float(rows[-1]["n"]), float(rows[-1]["srm"])),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=6.8,
            fontweight="bold",
            ha="left",
            va="center",
            annotation_clip=False,
        )
    ax.set(
        xlabel=r"sequence length $n$",
        ylabel=r"$P_{\rm SRM}$",
        title=r"Unknown length",
        xticks=[10, 20, 30, 40, 50],
        xlim=(8, 58),
        ylim=(-0.01, 0.82),
    )
    ax.legend(
        handles=(
            Line2D([0], [0], color=GRAY, marker="o", ms=3.5, lw=1.5,
                   label="Finite-n SRM"),
            Line2D([0], [0], color=GRAY, lw=1.0, ls=":",
                   label="Asymptotic limit"),
        ),
        loc="center",
        bbox_to_anchor=(0.48, 0.71),
        fontsize=6.35,
        frameon=False,
        handlelength=2.0,
        labelspacing=0.3,
    )

    ax = axes[2]
    n_values = sorted({int(float(row["n"])) for row in sdp_rows})
    c_values = sorted({float(row["c"]) for row in sdp_rows})
    upper_gap = np.full((len(c_values), len(n_values)), np.nan, dtype=float)
    seen_pairs: set[tuple[int, float]] = set()
    for row in sdp_rows:
        n_value = int(float(row["n"]))
        c_value = float(row["c"])
        key = (n_value, c_value)
        if key in seen_pairs:
            raise ValueError(f"duplicate SDP grid point n={n_value}, c={c_value}")
        seen_pairs.add(key)
        value = float(row["sdp_upper_minus_srm"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("SDP upper-gap endpoints must be finite and positive")
        upper_gap[c_values.index(c_value), n_values.index(n_value)] = value
    expected_pairs = len(n_values) * len(c_values)
    if len(seen_pairs) != expected_pairs or np.isnan(upper_gap).any():
        raise ValueError(
            "SDP data must form a complete rectangular (n,c) grid; "
            f"found {len(seen_pairs)} of {expected_pairs} points"
        )

    x_edges = np.arange(len(n_values) + 1, dtype=float) - 0.5
    y_edges = np.arange(len(c_values) + 1, dtype=float) - 0.5
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        upper_gap,
        shading="flat",
        cmap="viridis",
        norm=LogNorm(vmin=1e-6, vmax=2e-3),
    )
    ax.set(
        xlabel=r"sequence length $n$",
        ylabel=r"state overlap $c$",
        title=r"$U_{\mathrm{safe}}-P_{\mathrm{SRM}}$",
        xticks=np.arange(len(n_values)),
        yticks=np.arange(len(c_values)),
    )
    ax.set_xticklabels([str(value) for value in n_values])
    ax.set_yticklabels([f"{value:g}" for value in c_values])
    ax.set_xticks(np.arange(-0.5, len(n_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(c_values), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.65, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.055, pad=0.035)
    colorbar.set_label(
        r"$U_{\mathrm{safe}}-P_{\mathrm{SRM}}$",
        fontsize=9.4,
        labelpad=2.5,
    )
    colorbar.ax.tick_params(labelsize=7.4, width=0.65, length=2.5)
    colorbar.ax.minorticks_off()
    colorbar.outline.set_linewidth(0.65)
    colorbar.solids.set_rasterized(False)

    for axis in axes[:2]:
        axis.grid(axis="y", color=LIGHT_GRAY, lw=0.55, alpha=0.85)
        axis.set_axisbelow(True)
    for label, axis in zip("abc", axes):
        axis.text(
            -0.16,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=10.0,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    return _save(fig, output_dir, "figure3_finite_size")


def make_all_figures(
    output_dir: Path = ROOT / "paper1" / "figures",
    known_data_path: Path = (
        ROOT / "paper1" / "paper1_fixed_and_growing_srm.csv"
    ),
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
            known_data_path=known_data_path,
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
        "--known-data",
        type=Path,
        default=ROOT / "paper1" / "paper1_fixed_and_growing_srm.csv",
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
        known_data_path=args.known_data,
        sdp_data_path=args.sdp_data,
        srm_data_path=args.srm_data,
    ):
        print(generated)
