"""Tolerance-based verification for regenerated Paper I numerical tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TIMING_COLUMNS = {
    "runtime_seconds",
    "primal_solve_time_seconds",
    "dual_solve_time_seconds",
    "primal_iterations",
    "dual_iterations",
}
EXACT_COLUMNS = {
    "n",
    "M",
    "m",
    "candidate_count",
    "rank",
    "hypothesis_count",
    "numerical_rank",
    "solver",
    "cvxpy_version",
    "primal_status",
    "dual_status",
    "sdp_status",
    "method",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def report_path(path: Path) -> str:
    """Return a portable repository-relative path whenever possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def row_key(
    row: dict[str, str], key_columns: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(row.get(name, "") for name in key_columns)


def compare_tables(
    baseline: Path,
    candidate: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
    key_columns: tuple[str, ...] = ("m", "n", "c"),
) -> dict[str, object]:
    baseline_rows = {
        row_key(row, key_columns): row for row in read_rows(baseline)
    }
    candidate_rows = {
        row_key(row, key_columns): row for row in read_rows(candidate)
    }
    failures: list[dict[str, object]] = []
    if baseline_rows.keys() != candidate_rows.keys():
        failures.append(
            {
                "kind": "row-key mismatch",
                "baseline_only": sorted(baseline_rows.keys() - candidate_rows.keys()),
                "candidate_only": sorted(candidate_rows.keys() - baseline_rows.keys()),
            }
        )
    for key in sorted(baseline_rows.keys() & candidate_rows.keys()):
        left = baseline_rows[key]
        right = candidate_rows[key]
        for column in sorted(left.keys() & right.keys()):
            if column in TIMING_COLUMNS:
                continue
            if column in EXACT_COLUMNS:
                if left[column] != right[column]:
                    failures.append(
                        {
                            "kind": "exact mismatch",
                            "key": key,
                            "column": column,
                            "baseline": left[column],
                            "candidate": right[column],
                        }
                    )
                continue
            try:
                left_value = float(left[column])
                right_value = float(right[column])
            except ValueError:
                if left[column] != right[column]:
                    failures.append(
                        {
                            "kind": "text mismatch",
                            "key": key,
                            "column": column,
                        }
                    )
                continue
            if not (math.isfinite(left_value) and math.isfinite(right_value)):
                if left_value != right_value:
                    failures.append(
                        {"kind": "nonfinite mismatch", "key": key, "column": column}
                    )
                continue
            allowed = absolute_tolerance + relative_tolerance * max(
                abs(left_value), abs(right_value)
            )
            if abs(left_value - right_value) > allowed:
                failures.append(
                    {
                        "kind": "numeric mismatch",
                        "key": key,
                        "column": column,
                        "difference": abs(left_value - right_value),
                        "allowed": allowed,
                    }
                )
    return {
        "baseline": report_path(baseline),
        "candidate": report_path(candidate),
        "row_count": len(candidate_rows),
        "passed": not failures,
        "failures": failures,
    }


def certificate_thresholds(path: Path) -> dict[str, object]:
    rows = read_rows(path)
    thresholds = {
        "absolute_gap": 2e-7,
        "primal_equality_residual_fro": 2e-7,
        "primal_psd_violation": 2e-7,
        "dual_psd_violation": 2e-7,
        "complementarity_residual": 2e-6,
        "feasible_bound_gap": 2e-7,
        "primal_feasible_equality_residual_fro": 2e-10,
        "primal_feasible_psd_violation": 2e-10,
        "dual_feasible_psd_violation": 2e-10,
    }
    maxima = {
        column: max(abs(float(row[column])) for row in rows)
        for column in thresholds
    }
    failures = {
        column: {"maximum": maxima[column], "threshold": threshold}
        for column, threshold in thresholds.items()
        if maxima[column] > threshold
    }
    statuses_ok = all(
        row["primal_status"] in {"optimal", "optimal_inaccurate"}
        and row["dual_status"] in {"optimal", "optimal_inaccurate"}
        for row in rows
    )
    bounds_ordered = all(
        float(row["primal_feasible_objective"])
        <= float(row["dual_feasible_objective"])
        for row in rows
    )
    aliases_present = all(
        name in rows[0] for name in ("sdp_lower", "sdp_upper")
    )
    aliases_consistent = (not aliases_present) or all(
        math.isclose(
            float(row["sdp_lower"]),
            float(row["primal_feasible_objective"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(row["sdp_upper"]),
            float(row["dual_feasible_objective"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for row in rows
    )
    strictly_feasible = all(
        float(row["primal_feasible_min_eigenvalue"]) > 0.0
        and float(row["dual_feasible_min_slack"]) > 0.0
        for row in rows
    )
    return {
        "path": report_path(path),
        "row_count": len(rows),
        "maxima": maxima,
        "statuses_ok": statuses_ok,
        "bounds_ordered": bounds_ordered,
        "aliases_present": aliases_present,
        "aliases_consistent": aliases_consistent,
        "strictly_feasible": strictly_feasible,
        "passed": (
            statuses_ok
            and bounds_ordered
            and aliases_consistent
            and strictly_feasible
            and not failures
        ),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerated-dir",
        type=Path,
        default=ROOT / "paper1" / "reproduced",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper1" / "verification_report.json",
    )
    args = parser.parse_args()
    comparisons = [
        compare_tables(
            ROOT / "certified_sdp_results.csv",
            args.regenerated_dir / "certified_sdp_results.csv",
            absolute_tolerance=2e-7,
            relative_tolerance=2e-7,
        ),
        compare_tables(
            ROOT / "srm_scaling_m1.csv",
            args.regenerated_dir / "srm_scaling_m1.csv",
            absolute_tolerance=2e-10,
            relative_tolerance=2e-10,
        ),
        compare_tables(
            ROOT / "paper1" / "paper1_fixed_and_growing_srm.csv",
            args.regenerated_dir / "paper1_fixed_and_growing_srm.csv",
            absolute_tolerance=2e-10,
            relative_tolerance=2e-10,
            key_columns=("schedule", "c", "N"),
        ),
        compare_tables(
            ROOT / "paper1" / "paper1_h0_certified_sdp.csv",
            args.regenerated_dir / "paper1_h0_certified_sdp.csv",
            absolute_tolerance=2e-7,
            relative_tolerance=2e-7,
            key_columns=("c", "pi0", "N", "i"),
        ),
    ]
    certificates = certificate_thresholds(
        args.regenerated_dir / "certified_sdp_results.csv"
    )
    h0_certificates = certificate_thresholds(
        args.regenerated_dir / "paper1_h0_certified_sdp.csv"
    )
    report = {
        "passed": all(item["passed"] for item in comparisons)
        and certificates["passed"]
        and h0_certificates["passed"],
        "comparisons": comparisons,
        "certificate_thresholds": certificates,
        "h0_certificate_thresholds": h0_certificates,
        "timing_columns_ignored": sorted(TIMING_COLUMNS),
    }
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
