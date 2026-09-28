"""Train an anomaly detector and export scored transactions and metrics."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data import validate_transactions
from src.model import fit_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/creditcard.csv", help="Input transaction CSV")
    parser.add_argument("--model", choices=["isolation_forest", "lof"], default="isolation_forest")
    parser.add_argument("--contamination", type=float, default=0.0017,
                        help="Expected anomaly fraction, between 0.0001 and 0.5")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        parser.error(f"Dataset not found: {data_path}. Generate demo data or provide --data.")
    data = validate_transactions(pd.read_csv(data_path))
    result = fit_score(data, model_name=args.model, contamination=args.contamination)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result["scored"].to_csv(output_dir / "scored_transactions.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    joblib.dump({"pipeline": result["pipeline"], "threshold": result["threshold"],
                 "features": result["metrics"]["features"], "model": args.model},
                output_dir / "fraud_detector.joblib")
    print(json.dumps(result["metrics"], indent=2))
    print(f"\nSaved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
