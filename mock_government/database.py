"""
mock_government/database.py
===========================
Low-level SQLite storage for the MOCK challan database.

This file ONLY knows how to store/read (plate -> total_challan). It contains NO
ANPR logic and NO random test-seeding logic (that lives in testing_seed.py). It
stands in for the government's real records, so the few pre-loaded plates below
are treated as "real" known records, not test data.

Table:
    challans(plate TEXT PRIMARY KEY, total_challan INTEGER NOT NULL DEFAULT 0)
"""

import os
import sqlite3

# Named 'mock_*' on purpose so it never gets mistaken for a production DB.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_challan.db")

# Known "government" records (mock). These are real-looking data, NOT test logic.
_KNOWN_RECORDS = {
    "TN11A6701": 1500,
    "TN12AJ2643": 500,
    "TN13C6289": 2500,
}


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the table (if missing) and load the known records once."""
    conn = _connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS challans ("
        "  plate         TEXT PRIMARY KEY,"
        "  total_challan INTEGER NOT NULL DEFAULT 0"
        ")"
    )
    for plate, amount in _KNOWN_RECORDS.items():
        conn.execute(
            "INSERT OR IGNORE INTO challans (plate, total_challan) VALUES (?, ?)",
            (plate, amount),
        )
    conn.commit()
    conn.close()


def get_amount(plate):
    """Return the total challan (int) for a plate, or None if it isn't in the DB."""
    conn = _connect()
    cur = conn.execute("SELECT total_challan FROM challans WHERE plate = ?", (plate,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def exists(plate):
    return get_amount(plate) is not None


def insert(plate, amount):
    """Insert or overwrite a plate's total challan."""
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO challans (plate, total_challan) VALUES (?, ?)",
        (plate, int(amount)),
    )
    conn.commit()
    conn.close()


# Build the table + known records as soon as this module is imported.
init_db()
