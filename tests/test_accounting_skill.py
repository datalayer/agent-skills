# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The accounting skill's scripts on small books: what they refuse (an amount, a balance
or a due date that cannot be read is an input error, never a zero or a dropped line), what
they match (two references never reconcile unless they are the same), and what they say
(JSON null for a missing cell, a one-cent limit on the trial balance)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

SCRIPTS = Path(__file__).parent.parent / "agent_skills" / "skills" / "accounting" / "scripts"


def run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - the skill's own scripts, with this test's files
        [sys.executable, str(SCRIPTS / f"{script}.py"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_reconcile_matches_on_the_reference_and_refuses_an_unreadable_amount(tmp_path):
    left = write(
        tmp_path,
        "left.csv",
        "date,amount,reference\n2026-11-01,100,INV-A\n2026-11-02,50,INV-C\n2026-11-03,20,\n",
    )
    right = write(
        tmp_path,
        "right.csv",
        "date,amount,description\n2026-11-01,100,INV-B\n2026-11-02,50,inv-c\n2026-11-03,20,FEE\n",
    )
    done = run("reconcile", left, right, "--format", "json")
    assert done.returncode == 0, done.stderr
    outcome = json.loads(done.stdout)
    # INV-A against INV-B: the same amount, two different references — not a match.
    assert [pair["left"]["reference"] for pair in outcome["matched"]] == ["INV-C", None]
    assert [line["reference"] for line in outcome["only_left"]] == ["INV-A"]
    assert outcome["only_right"][0]["description"] == "INV-B"
    # A missing cell is null, never NaN.
    assert "NaN" not in done.stdout
    bad = write(tmp_path, "bad.csv", 'date,amount,reference\n2026-11-01,"1,000.00",INV-A\n')
    refused = run("reconcile", bad, right)
    assert refused.returncode == 1 and "not numbers" in refused.stderr


def test_variance_refuses_an_unreadable_amount(tmp_path):
    previous = write(tmp_path, "p.csv", "account,amount\nSales,1000\n")
    current = write(tmp_path, "c.csv", 'account,amount\nSales,"1,200.00"\n')
    refused = run("variance", previous, current)
    assert refused.returncode == 1 and "'amount'" in refused.stderr
    fine = run(
        "variance",
        previous,
        write(tmp_path, "c2.csv", "account,amount\nSales,1200\n"),
        "--format",
        "json",
    )
    assert json.loads(fine.stdout) == [
        {
            "account": "Sales",
            "previous": 1000.0,
            "current": 1200.0,
            "variance": 200.0,
            "percent": 20.0,
        }
    ]


def test_aging_reads_the_balance_first_and_refuses_what_it_cannot_read(tmp_path):
    invoices = write(
        tmp_path,
        "inv.csv",
        "invoice,customer,due_date,amount,paid,balance\n"
        "A1,Acme,2026-10-01,100,false,25\n"
        "A2,Acme,2026-10-15,80,false,0\n",
    )
    done = run("aging", invoices, "--as-of", "2026-11-07", "--format", "json")
    assert done.returncode == 0, done.stderr
    outcome = json.loads(done.stdout)
    # The balance is what is owed: 25, not the 100 of the invoice; A2's balance is zero, so it is settled.
    assert [row["amount"] for row in outcome["open"]] == [25.0]
    assert outcome["total"] == 25.0
    # The underscore spelling `run_skill_script` documents is read too.
    assert run("aging", invoices, "--as_of", "2026-11-07").returncode == 0
    undated = write(tmp_path, "undated.csv", "invoice,customer,due_date,amount\nA1,Acme,,100\n")
    refused = run("aging", undated, "--as-of", "2026-11-07")
    assert refused.returncode == 1 and "not a date" in refused.stderr
    unreadable = write(
        tmp_path, "unreadable.csv", "invoice,customer,due_date,amount\nA1,Acme,2026-10-01,abc\n"
    )
    refused = run("aging", unreadable, "--as-of", "2026-11-07")
    assert refused.returncode == 1 and "not numbers" in refused.stderr


def test_close_summary_rounds_to_the_cent_and_refuses_what_it_cannot_read(tmp_path):
    books = write(
        tmp_path,
        "tb.csv",
        "account,type,debit,credit\nCash,asset,1.00,0\nCapital,equity,0,0.99\n",
    )
    done = run("close_summary", books, "--format", "json")
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["balances"] is True
    off = write(
        tmp_path, "off.csv", "account,type,debit,credit\nCash,asset,1.00,0\nCapital,equity,0,0.98\n"
    )
    assert run("close_summary", off).returncode == 3
    unreadable = write(
        tmp_path, "bad.csv", "account,type,debit,credit\nCash,asset,one,0\nCapital,equity,0,1\n"
    )
    refused = run("close_summary", unreadable)
    assert refused.returncode == 1 and "not numbers" in refused.stderr
