# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import argparse
import json
import sys
from datetime import date, datetime

import pandas as pd

# How old the open invoices are: each one's age in days and its bucket, then the total per
# bucket and per customer. An invoice is open when its `paid` column is false, or its
# `balance` is not zero; every invoice is open when the file says neither.


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
    frame["_due"] = pd.to_datetime(frame[date_column], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    if "paid" in frame.columns:
        paid = frame["paid"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        frame = frame[~paid]
    elif "balance" in frame.columns:
        balance = pd.to_numeric(frame["balance"], errors="coerce").fillna(0.0)
        frame = frame[balance != 0]
        frame = frame.assign(amount=balance[balance != 0])
    return frame[frame["_due"].notna()].reset_index(drop=True)


def bucket_of(days, edges):
    low = 0
    for edge in edges:
        if days <= edge:
            return f"{low}-{edge}"
        low = edge + 1
    return f"{edges[-1] + 1}+"


def aging(frame, as_of, edges):
    rows = []
    for _, line in frame.iterrows():
        days = (as_of - line["_due"].date()).days
        rows.append(
            {
                "invoice": str(line.get("invoice", "")),
                "customer": str(line.get("customer", "")),
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
    parser.add_argument("--as_of", default=None, help="YYYY-MM-DD; today when unsaid")
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
            "Error: --as_of is YYYY-MM-DD and --buckets a comma-separated list of days",
            file=sys.stderr,
        )
        sys.exit(2)
    if not edges:
        print("Error: --buckets needs at least one edge", file=sys.stderr)
        sys.exit(2)
    outcome = aging(read(options.invoices, options.date), as_of, edges)
    print(json.dumps(outcome, indent=2) if options.format == "json" else table(outcome))
