# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import argparse
import json
import sys

import pandas as pd

# Matches the lines of two ledgers — a bank statement against the books — on the amount
# and, when both sides carry one, the reference; prints the pairs, then what stands alone
# on each side, with the totals.


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
    frame[amount] = pd.to_numeric(frame[amount], errors="coerce").fillna(0.0)
    if reference not in frame.columns:
        reference = "description" if "description" in frame.columns else None
    frame["_ref"] = frame[reference].astype(str).str.strip().str.lower() if reference else ""
    return frame.reset_index(drop=True)


def reconcile(left, right, amount, tolerance):
    matched, used_left, used_right = [], set(), set()
    for i, line in left.iterrows():
        candidates = right[~right.index.isin(used_right)]
        close = candidates[(candidates[amount] - line[amount]).abs() <= tolerance]
        if line["_ref"]:
            same = close[close["_ref"] == line["_ref"]]
            close = same if not same.empty else close
        if close.empty:
            continue
        j = close.index[0]
        used_left.add(i)
        used_right.add(j)
        matched.append(
            {
                "left": line.drop("_ref").to_dict(),
                "right": right.loc[j].drop("_ref").to_dict(),
                "difference": round(float(right.loc[j][amount] - line[amount]), 2),
            }
        )
    only_left = left[~left.index.isin(used_left)].drop(columns="_ref")
    only_right = right[~right.index.isin(used_right)].drop(columns="_ref")
    return {
        "matched": matched,
        "only_left": only_left.to_dict(orient="records"),
        "only_right": only_right.to_dict(orient="records"),
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
