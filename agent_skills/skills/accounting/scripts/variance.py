# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import argparse
import json
import sys

import pandas as pd

# The movement of each account between two periods: each account's previous and current
# totals, the variance and its percentage, the largest movement first. An amount that is
# not a number is an input error, never a zero: a zero would read as a drop.


def money(frame, column, path):
    raw = frame[column]
    text = raw.astype(str).str.strip()
    parsed = pd.to_numeric(raw, errors="coerce")
    bad = parsed.isna() & raw.notna() & (text != "")
    if bad.any():
        lines = [int(i) + 2 for i in frame.index[bad][:5]]
        print(
            f"Error: {path} has {int(bad.sum())} value(s) in {column!r} that are not numbers"
            f" (CSV lines {lines}): {list(text[bad][:3])}",
            file=sys.stderr,
        )
        sys.exit(1)
    return parsed.fillna(0.0)


def totals(path, account, amount):
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {path} is not there", file=sys.stderr)
        sys.exit(1)
    for column in (account, amount):
        if column not in frame.columns:
            print(
                f"Error: {path} has no column {column!r}; it has {list(frame.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)
    frame[amount] = money(frame, amount, path)
    return frame.groupby(frame[account].fillna("").astype(str).str.strip())[amount].sum()


def variance(previous, current, threshold):
    accounts = sorted(set(previous.index) | set(current.index))
    rows = []
    for name in accounts:
        before = float(previous.get(name, 0.0))
        after = float(current.get(name, 0.0))
        delta = after - before
        if abs(delta) < threshold:
            continue
        rows.append(
            {
                "account": name,
                "previous": round(before, 2),
                "current": round(after, 2),
                "variance": round(delta, 2),
                "percent": round(delta / before * 100, 1) if before else None,
            }
        )
    rows.sort(key=lambda row: -abs(row["variance"]))
    return rows


def table(rows):
    if not rows:
        return "No account moved by the threshold."
    width = max(len(row["account"]) for row in rows)
    lines = [
        f"{'account'.ljust(width)}  {'previous':>14}  {'current':>14}  {'variance':>14}  {'%':>8}"
    ]
    for row in rows:
        percent = "n/a" if row["percent"] is None else f"{row['percent']:.1f}"
        lines.append(
            f"{row['account'].ljust(width)}  {row['previous']:>14,.2f}  {row['current']:>14,.2f}  {row['variance']:>14,.2f}  {percent:>8}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The variance of each account between two periods."
    )
    parser.add_argument("previous")
    parser.add_argument("current")
    parser.add_argument("--account", default="account")
    parser.add_argument("--amount", default="amount")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    options = parser.parse_args()
    rows = variance(
        totals(options.previous, options.account, options.amount),
        totals(options.current, options.account, options.amount),
        options.threshold,
    )
    print(json.dumps(rows, indent=2) if options.format == "json" else table(rows))
