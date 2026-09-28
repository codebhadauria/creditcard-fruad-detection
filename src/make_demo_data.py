"""Generate a small synthetic CSV for trying the workflow."""

import argparse
from pathlib import Path

from src.data import make_demo_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo_transactions.csv")
    parser.add_argument("--rows", type=int, default=2500)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_demo_data(n_rows=args.rows).to_csv(output, index=False)
    print(f"Wrote {args.rows} synthetic transactions to {output}")


if __name__ == "__main__":
    main()
