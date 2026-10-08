"""Cockpit DB access — STRICTLY READ-ONLY.

Every connection opens muninn.db via SQLite's mode=ro URI. No write SQL
exists anywhere in the cockpit; verified by the repo-wide grep check in
MUNINN_COCKPIT_SPEC.md section 8.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "muninn.db"

RO_URI = f"file:{DB_PATH}?mode=ro"


def connect_ro() -> sqlite3.Connection:
    conn = sqlite3.connect(RO_URI, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_config(conn, key, default=None):
    row = conn.execute(
        "SELECT value FROM retrieval_config WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default