"""Generate the Task 8 candidate Figure 3 from its frozen diagnostic CSV.

This is a finite-size visualization only.  It does not plot or impute
``P_opt``.  Rows above the SDP cutoff contribute physical-Gram SRM values but
no small-SDP markers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "weighted_hull_diagnostics.csv"
DEFAULT_PDF = HERE / "task8_candidate_figure3.pdf"
DEFAULT_PNG = HERE / "task8_candidate_figure3.png"
EXPECTED_CSV_SHA256 = (
    "B9C5B773D6548CE87B1808C69ECB0CFF83502C6D1F5C76BE04DB7916B8C0884B"
)


def _read_diagnostic_rows(path: Path) -> list[dict[str, str]]:
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest().upper()
    if actual != EXPECTED_CSV_SHA256:
        raise RuntimeError(
            "Task 8 CSV hash mismatch: "
            f"expected {EXPECTED_CSV_SHA256}, observed {actual}"
        )
    rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 diagnostic rows, observed {len(rows)}")
    required = {
        "schedule",
        "n",
        "c",
        "lambda",
        "lambda_target",
        "P_SRM",
        "P_SRM_over_p1_squared",
        "repaired_primal_value",
        "shifted_dual_value",
        "P_opt",
        "sdp_status",
    }
    if not required.issubset(rows[0]):
        raise RuntimeError("Task 8 CSV is missing required Figure 3 columns")
    if any(row["P_opt"] != "" for row in rows):
        raise RuntimeError("candidate Figure 3 must not consume a P_opt value")
    return rows


def _float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not np.isfinite(value):
        raise RuntimeError(f"nonfinite {field} in Task 8 CSV")
    return value


def make_figure(csv_path: Path, pdf_path: Path, png_path: Path) -> None:
    rows = _read_diagnostic_rows(csv_path)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.0,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = ("#1768AC", "#2A9D8F", "#E09F3E", "#B33F62", "#665191")
    markers = ("o", "s", "^", "D", "v")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.55), constrained_layout=True)

    compact = [row for row in rows if row["schedule"] == "compact_lambda"]
    lambda_values = sorted({_float(row, "lambda_target") for row in compact})
    for index, lambda_value in enumerate(lambda_values):
        group = sorted(
            (
                row
                for row in compact
                if abs(_float(row, "lambda_target") - lambda_value) < 1.0e-12
            ),
            key=lambda row: int(row["n"]),
        )
        axes[0].plot(
            [int(row["n"]) for row in group],
            [_float(row, "P_SRM_over_p1_squared") for row in group],
            color=colors[index],
            marker=markers[index],
            label=rf"$\lambda={lambda_value:g}$",
        )
    axes[0].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[0].set_title(r"Compact $\lambda=n(1-c)$")
    axes[0].set_xlabel("$n$")
    axes[0].set_ylabel(r"$P_{\rm SRM}/p_1(c)^2$")
    axes[0].set_xticks((5, 8, 16))
    axes[0].legend(frameon=False, ncol=1, loc="best")

    outer_names = {
        "outer_log_log_n": r"$\lambda_n=\log\log n$",
        "outer_sqrt_log_n": r"$\lambda_n=\sqrt{\log n}$",
        "outer_n_one_third": r"$\lambda_n=n^{1/3}$",
    }
    for index, (schedule, label) in enumerate(outer_names.items()):
        group = sorted(
            (row for row in rows if row["schedule"] == schedule),
            key=lambda row: int(row["n"]),
        )
        axes[1].plot(
            [int(row["n"]) for row in group],
            [_float(row, "P_SRM_over_p1_squared") for row in group],
            color=colors[index],
            marker=markers[index],
            label=label,
        )
    axes[1].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_title("Moving-outer schedules")
    axes[1].set_xlabel("$n$")
    axes[1].set_ylabel(r"$P_{\rm SRM}/p_1(c)^2$")
    axes[1].set_xticks((8, 16, 32))
    axes[1].legend(frameon=False, loc="best")

    solved = sorted(
        (
            row
            for row in rows
            if row["sdp_status"] == "computed_residual_checked_diagnostic"
        ),
        key=lambda row: _float(row, "c"),
    )
    c_values = np.array([_float(row, "c") for row in solved])
    primal_offset = np.array(
        [
            _float(row, "repaired_primal_value") - _float(row, "P_SRM")
            for row in solved
        ]
    )
    dual_offset = np.array(
        [
            _float(row, "shifted_dual_value") - _float(row, "P_SRM")
            for row in solved
        ]
    )
    axes[2].plot(
        c_values,
        primal_offset,
        marker="o",
        color="#B33F62",
        label="re-feasibilized primal value",
    )
    axes[2].plot(
        c_values,
        dual_offset,
        marker="s",
        linestyle="--",
        color="#665191",
        label="shifted dual value",
    )
    axes[2].axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[2].set_title("Small-size floating SDP diagnostic")
    axes[2].set_xlabel("overlap $c$ ($n=5$)")
    axes[2].set_ylabel(r"floating value $-P_{\rm SRM}$")
    axes[2].legend(frameon=False, loc="upper left")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(
            -0.18,
            1.05,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8.0,
            va="top",
        )
        axis.grid(color="#D8D8D8", linewidth=0.45, alpha=0.55)
        axis.spines[["top", "right"]].set_visible(False)

    metadata = {
        "Title": "Task 8 candidate Figure 3: physical-Gram SRM and small-size floating SDP diagnostics",
        "Subject": f"Data CSV SHA-256 {EXPECTED_CSV_SHA256}; no P_opt values plotted",
        "Creator": "plot_task8_candidate_figure3.py",
        "CreationDate": datetime(2026, 8, 24, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 8, 24, tzinfo=timezone.utc),
    }
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
    fig.savefig(png_path, dpi=320, bbox_inches="tight", metadata={"Software": metadata["Creator"]})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args()
    make_figure(args.csv, args.pdf, args.png)
    print(f"CSV_SHA256={EXPECTED_CSV_SHA256}")
    print(
        "FIGURE_SEMANTICS=floating primal-dual span; "
        "re-feasibilized primal / shifted dual values; P_opt absent"
    )
    print(f"WROTE_PDF={args.pdf.resolve()}")
    print(f"WROTE_PNG={args.png.resolve()}")


if __name__ == "__main__":
    main()
