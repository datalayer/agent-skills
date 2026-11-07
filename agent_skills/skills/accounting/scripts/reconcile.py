# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import argparse
import json
import sys

import pandas as pd

# Matches the lines of two ledgers — a bank statement against the books — on the amount
# and the reference: two lines that both carry a reference match only when it is the same;
# a line without one matches on the amount alone. Prints the pairs, then what stands alone
# on each side, with the totals. An amount that is not a number is an input error, never
# a zero: a zero would reconcile against a genuine zero.


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


def read(path, amount, reference):
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {path} is not there", file=sys.stderr)
        sys.exit(1)
    if amount not in frame.columns:
        print(
            f"Error: {path} has no column {amount!r}; it has {list(frame.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)
    frame[amount] = money(frame, amount, path)
    if reference not in frame.columns:
        reference = "description" if "description" in frame.columns else None
    frame["_ref"] = (
        frame[reference].fillna("").astype(str).str.strip().str.lower() if reference else ""
    )
    return frame.reset_index(drop=True)


def record(line):
    """A line as JSON reads it: a missing cell is null, never NaN."""
    return {key: (None if pd.isna(value) else value) for key, value in line.items()}


def reconcile(left, right, amount, tolerance):
    matched, used_left, used_right = [], set(), set()
    for i, line in left.iterrows():
        candidates = right[~right.index.isin(used_right)]
        close = candidates[(candidates[amount] - line[amount]).abs() <= tolerance]
        if line["_ref"]:
            # The same reference, or none on the other side: two different
            # references never reconcile, whatever the amount.
            close = close[(close["_ref"] == line["_ref"]) | (close["_ref"] == "")]
        if close.empty:
            continue
        j = close.index[0]
        used_left.add(i)
        used_right.add(j)
        matched.append(
            {
                "left": record(line.drop("_ref")),
                "right": record(right.loc[j].drop("_ref")),
                "difference": round(float(right.loc[j][amount] - line[amount]), 2),
            }
        )
    only_left = left[~left.index.isin(used_left)].drop(columns="_ref")
    only_right = right[~right.index.isin(used_right)].drop(columns="_ref")
    return {
        "matched": matched,
        "only_left": [record(line) for _, line in only_left.iterrows()],
        "only_right": [record(line) for _, line in only_right.iterrows()],
        "totals": {
            "left": round(float(left[amount].sum()), 2),
            "right": round(float(right[amount].sum()), 2),
            "matched": len(matched),
            "only_left": round(float(only_left[amount].sum()), 2),
            "only_right": round(float(only_right[amount].sum()), 2),
        },
    }


def table(result):
    totals = result["totals"]
    lines = [f"Matched {totals['matched']} line(s)."]
    for pair in result["matched"]:
        lines.append(f"  = {pair['left']}  ~  {pair['right']}  (diff {pair['difference']})")
    lines.append(f"Only on the left ({totals['only_left']}):")
    lines += [f"  < {line}" for line in result["only_left"]] or ["  none"]
    lines.append(f"Only on the right ({totals['only_right']}):")
    lines += [f"  > {line}" for line in result["only_right"]] or ["  none"]
    lines.append(f"Totals: left {totals['left']}, right {totals['right']}.")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconcile two ledgers.")
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--amount", default="amount")
    parser.add_argument("--reference", default="reference")
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--format", choices=["table", "json"], default="table")
    options = parser.parse_args()
    outcome = reconcile(
        read(options.left, options.amount, options.reference),
        read(options.right, options.amount, options.reference),
        options.amount,
        options.tolerance,
    )
    if options.format == "json":
        print(json.dumps(outcome, indent=2, default=str))
    else:
        print(table(outcome))
