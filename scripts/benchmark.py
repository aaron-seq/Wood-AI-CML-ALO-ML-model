#!/usr/bin/env python3
"""Measure where scoring time actually goes.

Run with:

    python scripts/benchmark.py            # default sizes
    python scripts/benchmark.py --rows 100000

Reports wall-clock and peak memory per stage so capacity questions
("how big a file can one worker take?") have a number behind them rather
than a guess. Not a regression gate -- timings on shared CI runners are
too noisy for that. It exists so the limits in .env.example and the
sizing advice in docs/DEPLOYMENT.md are grounded.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import engineer_features  # noqa: E402
from app.ingestion import parse_bytes  # noqa: E402
from app.risk import classify_series  # noqa: E402
from app.utils import validate_cml_dataframe  # noqa: E402

COMMODITIES = ["Crude Oil", "Natural Gas", "Steam", "Fuel Gas", "Condensate"]
FEATURE_TYPES = ["Pipe", "Elbow", "Tee", "Flange", "Weld"]
SHAPES = ["Internal", "External", "Both"]


def synthetic_frame(rows: int, seed: int = 0) -> pd.DataFrame:
    """A CML dataset of the requested size, shaped like the real one."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "id_number": [f"CML-{index:06d}" for index in range(rows)],
            "average_corrosion_rate": rng.uniform(0.01, 0.5, rows).round(3),
            "thickness_mm": rng.uniform(3.5, 16.0, rows).round(1),
            "commodity": rng.choice(COMMODITIES, rows),
            "feature_type": rng.choice(FEATURE_TYPES, rows),
            "cml_shape": rng.choice(SHAPES, rows),
            "last_inspection_date": "2024-01-15",
            "risk_score": rng.integers(0, 100, rows),
        }
    )


def measure(label: str, work: Callable[[], Any], repeats: int = 3) -> dict[str, Any]:
    """Time and memory-profile a callable.

    The median of several runs, not the mean: one slow run caused by the
    OS scheduler should not move the reported number.
    """
    # Warm up so import-time and first-touch costs are not attributed here.
    work()

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        work()
        timings.append(time.perf_counter() - started)

    tracemalloc.start()
    work()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "stage": label,
        "median_s": statistics.median(timings),
        "peak_mib": peak / 1_048_576,
    }


def benchmark(rows: int, model: Any | None) -> list[dict[str, Any]]:
    frame = synthetic_frame(rows)
    csv_bytes = frame.to_csv(index=False).encode()

    results = [
        measure("parse CSV", lambda: parse_bytes(csv_bytes, "bench.csv")),
        measure("validate", lambda: validate_cml_dataframe(frame)),
        measure("engineer features", lambda: engineer_features(frame)),
    ]

    engineered = engineer_features(frame)
    results.append(
        measure(
            "classify risk",
            lambda: classify_series(
                engineered["remaining_life_years"],
                engineered["average_corrosion_rate"],
                engineered["thickness_mm"],
            ),
        )
    )

    if model is not None:
        columns = list(model.feature_names_in_)
        matrix = engineered[columns]
        results.append(measure("model.predict", lambda: model.predict(matrix)))
        results.append(measure("model.predict_proba", lambda: model.predict_proba(matrix)))

    results.append(
        {"stage": "upload size", "median_s": None, "peak_mib": len(csv_bytes) / 1_048_576}
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[500, 10_000, 100_000],
        help="Dataset sizes to measure",
    )
    parser.add_argument(
        "--model",
        default="models/cml_elimination_model.joblib",
        help="Model artifact; skipped if absent",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    model = joblib.load(model_path) if model_path.exists() else None
    if model is None:
        print(f"No model at {model_path}; skipping inference stages.\n")

    for rows in args.rows:
        print(f"=== {rows:,} CMLs")
        print(f"{'stage':<24}{'median':>12}{'peak memory':>14}")
        total = 0.0
        for result in benchmark(rows, model):
            seconds = result["median_s"]
            if seconds is None:
                print(f"{result['stage']:<24}{'-':>12}{result['peak_mib']:>13.1f}M")
                continue
            total += seconds
            print(f"{result['stage']:<24}{seconds * 1000:>10.1f}ms{result['peak_mib']:>13.1f}M")
        print(f"{'end-to-end':<24}{total * 1000:>10.1f}ms\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
