# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import argparse
import json
import sys
from datetime import date, datetime

import pandas as pd

# How old the open invoices are: each one's age in days and its bucket, then the total per
# bucket and per customer. The open amount is the `balance` when the file has one, else the
# `amount` of the invoices whose `paid` is false, else every invoice's amount. An amount, a
# balance or a due date that cannot be read is an input error, never a zero or a dropped
# line: the report would understate what is owed.


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


def read(path, date_column):
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {path} is not there", file=sys.stderr)
        sys.exit(1)
    if date_column not in frame.columns:
        date_column = "date" if "date" in frame.columns else None
    if date_column is None or "amount" not in frame.columns:
        print(
            f"Error: {path} needs a date column and `amount`; it has {list(frame.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)
    frame["amount"] = money(frame, "amount", path)
    if "balance" in frame.columns:
        # What is still owed, whatever `paid` says: an invoice part paid is open for the rest.
        balance = money(frame, "balance", path)
        frame = frame.assign(amount=balance)[balance != 0]
    elif "paid" in frame.columns:
        paid = frame["paid"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        frame = frame[~paid]
    frame = frame.copy()
    frame["_due"] = pd.to_datetime(frame[date_column], errors="coerce")
    unreadable = frame["_due"].isna()
    if unreadable.any():
        lines = [int(i) + 2 for i in frame.index[unreadable][:5]]
        print(
            f"Error: {path} has {int(unreadable.sum())} open invoice(s) whose {date_column!r}"
            f" is missing or not a date (CSV lines {lines})",
            file=sys.stderr,
        )
        sys.exit(1)
    return frame.reset_index(drop=True)


def bucket_of(days, edges):
    low = 0
    for edge in edges:
        if days <= edge:
            return f"{low}-{edge}"
        low = edge + 1
    return f"{edges[-1] + 1}+"


def cell(line, key):
    value = line.get(key, "")
    return "" if pd.isna(value) else str(value)


def aging(frame, as_of, edges):
    rows = []
    for _, line in frame.iterrows():
        days = (as_of - line["_due"].date()).days
        rows.append(
            {
                "invoice": cell(line, "invoice"),
                "customer": cell(line, "customer"),
                "due": line["_due"].date().isoformat(),
                "amount": round(float(line["amount"]), 2),
                "days": days,
                "bucket": bucket_of(max(days, 0), edges),
            }
        )
    buckets = {}
    customers = {}
    for row in rows:
        buckets[row["bucket"]] = round(buckets.get(row["bucket"], 0.0) + row["amount"], 2)
        customers[row["customer"]] = round(customers.get(row["customer"], 0.0) + row["amount"], 2)
    return {
        "as_of": as_of.isoformat(),
        "open": rows,
        "by_bucket": dict(
            sorted(buckets.items(), key=lambda item: int(item[0].split("-")[0].rstrip("+")))
        ),
        "by_customer": dict(sorted(customers.items(), key=lambda item: -item[1])),
        "total": round(sum(row["amount"] for row in rows), 2),
    }


def table(result):
    lines = [
        f"Open invoices as of {result['as_of']}: {len(result['open'])}, {result['total']:,.2f} in all."
    ]
    for row in result["open"]:
        lines.append(
            f"  {row['invoice'] or '-':<12} {row['customer'] or '-':<24} due {row['due']}  {row['amount']:>12,.2f}  {row['days']:>5} d  {row['bucket']}"
        )
    lines.append("By bucket (days):")
    lines += [f"  {name:<8} {amount:>12,.2f}" for name, amount in result["by_bucket"].items()]
    lines.append("By customer:")
    lines += [
        f"  {name or '-':<24} {amount:>12,.2f}" for name, amount in result["by_customer"].items()
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The aging of the open invoices.")
    parser.add_argument("invoices")
    # `run_skill_script` writes a kwarg `as_of` as `--as-of`; both spellings are read.
    parser.add_argument("--as-of", "--as_of", dest="as_of", default=None, help="YYYY-MM-DD")
    parser.add_argument("--date", default="due_date")
    parser.add_argument("--buckets", default="30,60,90")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    options = parser.parse_args()
    try:
        as_of = (
            datetime.strptime(options.as_of, "%Y-%m-%d").date() if options.as_of else date.today()
        )
        edges = sorted(int(edge) for edge in options.buckets.split(",") if edge.strip())
    except ValueError:
        print(
            "Error: --as-of is YYYY-MM-DD and --buckets a comma-separated list of days",
            file=sys.stderr,
        )
        sys.exit(2)
    if not edges:
        print("Error: --buckets needs at least one edge", file=sys.stderr)
        sys.exit(2)
    outcome = aging(read(options.invoices, options.date), as_of, edges)
    print(json.dumps(outcome, indent=2) if options.format == "json" else table(outcome))
