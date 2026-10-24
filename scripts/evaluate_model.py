from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oasis_pair_analysis import binary_labels, endpoint_metrics, pair_metrics, pair_results, predict_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--pairs-out", required=True, type=Path)
    parser.add_argument("--expected-pairs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    model = joblib.load(args.model)
    probabilities = predict_probabilities(model, frame, args.expected_pairs)
    pairs = pair_results(frame, probabilities)
    summary = {
        "endpoint": endpoint_metrics(binary_labels(frame["expected_label"]), probabilities),
        "pair": pair_metrics(pairs),
        "predict_proba_count": 1,
        "fit_count": 0,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pairs.to_csv(args.pairs_out, index=False, encoding="utf-8", lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
