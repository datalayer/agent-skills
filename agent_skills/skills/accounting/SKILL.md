---
name: accounting
description: Accounting analysis over CSV books — a reconciliation of two ledgers, the variance between two periods, the aging of open invoices, and an audit-ready close summary of a trial balance. Use it when the person hands you exports from their books and asks what matches, what moved, what is overdue or whether the books balance.
license: BSD 3-Clause. LICENSE has complete terms
version: 1.0.0
tags:
  - accounting
  - finance
  - reconciliation
  - close
author: Datalayer
metadata:
  visibility: public
---

# Accounting Skill

Four analyses an accountant runs at a close, over CSV exports of the books:
which entries of two ledgers match and which stand alone, how each account
moved between two periods, how old the open invoices are, and whether a trial
balance balances and what it says. pandas does the arithmetic; every script
reads CSV files and prints a table or JSON, and never writes to the books.

## Required Environment Variables

- None required.

## Required Software

- `pip install "agent_skills[skills-accounting]"` — pandas.

## Invocation Contract

- Always call `run_skill_script` with:
  - `skill_name`: `accounting` (lowercase, exactly as returned by `list_skills`)
  - `script_name`: one of `reconcile`, `variance`, `aging`, `close_summary`
- Pass positional parameters with `args` in exact order; every path is a CSV file in the sandbox.
- Pass optional flags with `kwargs` using the documented keys below. Unknown keys exit with code `2`.
- Amounts are read as numbers; a column named `amount` is expected unless a flag names another.

## Column conventions

- A ledger line has `date`, `amount` and a `reference` (or `description`); `reconcile` matches on the amount and, when both sides have one, the reference.
- A period file has `account` and `amount`; `variance` sums each account before comparing.
- An invoice has `invoice`, `customer`, `date` (or `due_date`) and `amount`, and a `paid` column (`true`/`false`) or a `balance`; `aging` keeps the open ones.
- A trial balance has `account`, `type` (`asset`, `liability`, `equity`, `revenue`, `expense`) and `debit` and `credit`, or one signed `amount`.

## Scripts API

### `script_name: reconcile`

- Purpose: match the lines of two ledgers, a bank statement against the books for instance.
- Positional `args`: `["left.csv", "right.csv"]`
- Supported `kwargs`:
  - `amount`: the amount column's name (default `amount`)
  - `reference`: the reference column's name (default `reference`, else `description`)
  - `tolerance`: the largest amount difference still a match (default `0.0`)
  - `format`: `table | json`
- Prints the matched pairs, then the lines only on the left and only on the right, with each side's total.

### `script_name: variance`

- Purpose: the movement of each account between two periods.
- Positional `args`: `["previous.csv", "current.csv"]`
- Supported `kwargs`:
  - `account`: the account column's name (default `account`)
  - `amount`: the amount column's name (default `amount`)
  - `threshold`: only accounts whose absolute variance is at least this (default `0`)
  - `format`: `table | json`
- Prints each account's previous and current totals, the variance and its percentage, largest first.

### `script_name: aging`

- Purpose: how old the open invoices are, in buckets.
- Positional `args`: `["invoices.csv"]`
- Supported `kwargs`:
  - `as_of`: the day to age from, `YYYY-MM-DD` (default today)
  - `date`: the date column's name (default `due_date`, else `date`)
  - `buckets`: the bucket edges in days, comma-separated (default `30,60,90`)
  - `format`: `table | json`
- Prints each open invoice's age and bucket, then the total per bucket and per customer.

### `script_name: close_summary`

- Purpose: whether a trial balance balances, and the totals by type: assets, liabilities, equity, revenue, expenses, the net result.
- Positional `args`: `["trial_balance.csv"]`
- Supported `kwargs`:
  - `format`: `table | json`
- Prints the check (debits equal credits, within a cent) and the summary; exits with code `3` when the books do not balance, so the person is told first.

## Common Failure Modes

- Exit code `1`: a file is missing or a required column is not there; the error names the columns found.
- Exit code `2`: an unknown `kwargs` key or a missing positional path.
- Exit code `3` (`close_summary`): the trial balance does not balance; the difference is printed.
