import sqlite3
import os

DB_PATH = "history.db"

def init():
    """Initialize history database"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            time TEXT,
            label TEXT,
            confidence REAL,
            model_used TEXT
        )
    """)
    con.commit()
    con.close()

def log(time_str, label, confidence, model_used):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO history (time, label, confidence, model_used) VALUES (?, ?, ?, ?)",
        (time_str, label, confidence, model_used)
    )
    con.commit()
    con.close()

def get():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT time, label, confidence, model_used
        FROM history
        ORDER BY time DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    con.close()
    return rows
