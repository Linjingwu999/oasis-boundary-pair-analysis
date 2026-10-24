from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oasis_pair_analysis import FEATURE_COLUMNS, fit_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model-out", required=True, type=Path)
    parser.add_argument("--audit-out", required=True, type=Path)
    parser.add_argument("--expected-pairs", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input)
    model, audit = fit_pipeline(frame, args.expected_pairs)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    audit.update(
        {
            "feature_order": list(FEATURE_COLUMNS),
            "pipeline": {
                "imputer": {"strategy": "median", "add_indicator": True},
                "scaler": "StandardScaler",
                "classifier": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 2000,
                    "random_state": 9801,
                    "class_weight": "balanced",
                },
            },
            "fit_count": 1,
        }
    )
    args.audit_out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
