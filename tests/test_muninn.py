#!/usr/bin/env python3
"""Muninn test suite — regression guard for the engine.

Runs entirely against a temp DB (MUNINN_DB_PATH override) plus a temp
Huginn client DB; never touches production muninn.db or huginn.db.
Core invariants under test:
  1. learn_fact writes are instantly FTS-searchable
  2. Supersession: same subject+predicate replaces old live fact
  3. recall_facts: two-tier ordering (lexical winners, entity tail),
     synonym expansion, predicate-exact + subject boost, chatter filter
  4. hybrid_search returns no said_in_* facts
  5. Recall never crashes on empty/garbage queries
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TMP = tempfile.mkdtemp(prefix="muninn_test_")
os.environ["MUNINN_DB_PATH"] = os.path.join(TMP, "test_muninn.db")

import muninn  # noqa: E402  (after env override)

PASS = FAIL = 0
FAILURES = []


def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  PASS {name}")
    except Exception as e:
        FAIL += 1
        FAILURES.append((name, str(e)))
        print(f"  FAIL {name}: {e}")


# ─── Setup: fresh DB ────────────────────────────────────────────────────────
muninn.init_db()

# ─── Seed: facts with known predicates ─────────────────────────────────────
r = muninn.learn_fact("Alex", "employer", "Acme Staffing (APAC MSP contract)")
assert r.get("ok"), r
r = muninn.learn_fact("Alex", "has_child", "Sam")
assert r.get("ok"), r
r = muninn.learn_fact("Alex", "has_child", "Riley")
assert r.get("ok"), r
r = muninn.learn_fact("Alex", "philosophy", "Stoicism, Marcus Aurelius")
assert r.get("ok"), r
r = muninn.learn_fact("Huginn", "memory_backend", "Muninn (standalone)")
assert r.get("ok"), r
# chatter row that must never surface
r = muninn.learn_fact("P", "said_in_g", "a", source_session_id="chat")
assert r.get("ok"), r


def triples(rows):
    return [(x["subject"], x["predicate"], str(x["object"])) for x in rows]


# ─── 1. Write path: instant FTS indexing ───────────────────────────────────
def test_instant_fts():
    rows = muninn.lexical_search_facts("Acme Staffing", top_k=5)
    assert any(t[1] == "employer" for t in triples(rows)), f"not indexed: {rows}"


check("learn_fact -> instantly FTS searchable", test_instant_fts)


# ─── 2. Supersession ───────────────────────────────────────────────────────
def test_supersession():
    r = muninn.learn_fact("Alex", "employer", "Acme Staffing v2")
    assert r["ok"] and r["superseded"] >= 1, r
    conn = sqlite3.connect(os.environ["MUNINN_DB_PATH"])
    live = conn.execute(
        "SELECT COUNT(*) FROM facts f JOIN entities e ON e.id=f.subject_entity_id "
        "WHERE e.name='Alex' AND f.predicate='employer' AND "
        "f.superseded_by IS NULL AND f.valid_until IS NULL").fetchone()[0]
    conn.close()
    assert live == 1, f"{live} live employer facts after supersession"


check("same subject+predicate supersedes old fact", test_supersession)


# ─── 3. Recall behaviour ───────────────────────────────────────────────────
def test_recall_employer():
    rows = muninn.recall_facts("Who is Alex's employer?", top_k=3)
    blob = json.dumps(triples(rows)).lower()
    assert "guidant" in blob, f"miss: {rows}"


def test_recall_kids_synonym():
    rows = muninn.recall_facts("Alex's kids", top_k=3)
    blob = json.dumps(triples(rows)).lower()
    assert "keian" in blob or "tiarn" in blob, f"synonym miss: {rows}"


def test_recall_no_chatter():
    rows = muninn.recall_facts("anything Alex said", top_k=10)
    assert all(not t[1].startswith("said_in") for t in triples(rows)), rows


def test_recall_subject_priority():
    rows = muninn.recall_facts("What tools does Alex use?", top_k=3)
    # first lexical hit should be a Alex-subject fact, not a random tool
    if rows and rows[0]["predicate"].lower().find("tool") >= 0:
        assert rows[0]["subject"] == "Alex", f"tiebreak lost: {rows[0]}"


check("recall: employer query", test_recall_employer)
check("recall: kids synonym expansion", test_recall_kids_synonym)
check("recall: said_in chatter never surfaces", test_recall_no_chatter)
check("recall: subject tiebreak on predicate match", test_recall_subject_priority)


# ─── 4. Hybrid search cleanliness ──────────────────────────────────────────
def test_hybrid_no_chatter():
    res = muninn.hybrid_search("Alex employer", top_k=5)
    for f in res.get("facts", []):
        assert not str(f.get("predicate", "")).startswith("said_in"), f


check("hybrid_search: fact fusion excludes chatter", test_hybrid_no_chatter)


# ─── 5. Robustness ─────────────────────────────────────────────────────────
def test_empty_query():
    assert muninn.recall_facts("", top_k=5) == []
    assert muninn.recall_facts("   ", top_k=5) == []


def test_garbage_query():
    rows = muninn.recall_facts("zzzqqq nonexistent", top_k=5)
    assert isinstance(rows, list)


def test_bad_learn():
    r = muninn.learn_fact("", "", "")
    assert r.get("ok") in (False, True)  # must not raise


check("recall: empty queries return []", test_empty_query)
check("recall: garbage query no crash", test_garbage_query)
check("learn_fact: degenerate input no crash", test_bad_learn)


# ─── Teardown ───────────────────────────────────────────────────────────────
shutil.rmtree(TMP, ignore_errors=True)
print(f"\nPASSED: {PASS}  FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)