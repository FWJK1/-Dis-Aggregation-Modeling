from pathlib import Path

import pandas as pd


def main():
    in_dir = Path("project/data/per_task")
    out_path = Path("project/data/results.csv")

    files = sorted(in_dir.glob("results_*.csv"))
    if not files:
        raise SystemExit(f"no per-task CSVs found in {in_dir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows from {len(files)} files to {out_path}")


if __name__ == "__main__":
    main()
