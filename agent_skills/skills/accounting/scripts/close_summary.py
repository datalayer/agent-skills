# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import argparse
import json
import sys

import pandas as pd

# Whether a trial balance balances, and what it says: the totals by type — assets,
# liabilities, equity, revenue, expenses — and the net result. Exits with code 3 when
# debits and credits differ by more than a cent, so the person is told first.

TYPES = ("asset", "liability", "equity", "revenue", "expense")


def read(path):
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: {path} is not there", file=sys.stderr)
        sys.exit(1)
    needed = {"account", "type"}
    if not needed <= set(frame.columns):
        print(
            f"Error: {path} needs {sorted(needed)} and debit/credit or amount; it has {list(frame.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if "debit" in frame.columns and "credit" in frame.columns:
        frame["debit"] = pd.to_numeric(frame["debit"], errors="coerce").fillna(0.0)
        frame["credit"] = pd.to_numeric(frame["credit"], errors="coerce").fillna(0.0)
    elif "amount" in frame.columns:
        amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
        frame["debit"] = amount.clip(lower=0.0)
        frame["credit"] = (-amount).clip(lower=0.0)
    else:
        print(f"Error: {path} needs debit and credit columns, or a signed amount", file=sys.stderr)
        sys.exit(1)
    frame["type"] = frame["type"].astype(str).str.strip().str.lower()
    unknown = sorted(set(frame["type"]) - set(TYPES))
    if unknown:
        print(f"Error: unknown account types {unknown}; one of {list(TYPES)}", file=sys.stderr)
        sys.exit(1)
    return frame


def summary(frame):
    debit_total = round(float(frame["debit"].sum()), 2)
    credit_total = round(float(frame["credit"].sum()), 2)
    by_type = {}
    for kind in TYPES:
        part = frame[frame["type"] == kind]
        # Assets and expenses carry debit balances; the others credit balances.
        signed = (
            (part["debit"] - part["credit"])
            if kind in ("asset", "expense")
            else (part["credit"] - part["debit"])
        )
        by_type[kind] = round(float(signed.sum()), 2)
    net = round(by_type["revenue"] - by_type["expense"], 2)
    return {
        "debits": debit_total,
        "credits": credit_total,
        "difference": round(debit_total - credit_total, 2),
        "balances": abs(debit_total - credit_total) < 0.01,
        "by_type": by_type,
        "net_result": net,
        "accounts": len(frame),
        "equation": {
            "assets": by_type["asset"],
            "liabilities_and_equity": round(by_type["liability"] + by_type["equity"] + net, 2),
        },
    }


def table(result):
    lines = [
        f"Debits {result['debits']:,.2f}  Credits {result['credits']:,.2f}  "
        + (
            "balanced."
            if result["balances"]
            else f"NOT balanced: difference {result['difference']:,.2f}."
        ),
        f"Accounts: {result['accounts']}",
    ]
    lines += [f"  {kind:<10} {amount:>14,.2f}" for kind, amount in result["by_type"].items()]
    lines.append(f"Net result (revenue - expenses): {result['net_result']:,.2f}")
    lines.append(
        f"Assets {result['equation']['assets']:,.2f} vs liabilities + equity + result {result['equation']['liabilities_and_equity']:,.2f}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An audit-ready summary of a trial balance.")
    parser.add_argument("trial_balance")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    options = parser.parse_args()
    result = summary(read(options.trial_balance))
    print(json.dumps(result, indent=2) if options.format == "json" else table(result))
    if not result["balances"]:
        sys.exit(3)
