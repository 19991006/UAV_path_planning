#!/usr/bin/env python3
"""Aggregate per-cell metrics.json files into summary CSVs for plotting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def find_metrics(base_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(base_dir.rglob("metrics.json")):
        with open(p) as f:
            d = json.load(f)
        d["_source"] = str(p.parent.relative_to(base_dir))
        rows.append(d)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print("No rows to write.")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate experiment results")
    p.add_argument("--exp1-dir", type=str, default="eval_outputs/exp1")
    p.add_argument("--exp2-dir", type=str, default="eval_outputs/exp2")
    p.add_argument("--output-dir", type=str, default="eval_outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    for tag, d in [("exp1", args.exp1_dir), ("exp2", args.exp2_dir)]:
        p = Path(d)
        if p.exists():
            write_csv(find_metrics(p), out / f"{tag}_summary.csv")
        else:
            print(f"{tag} dir not found: {p}")


if __name__ == "__main__":
    main()
