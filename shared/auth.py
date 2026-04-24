"""
shared/auth.py — Unified authentication for the AI Vision System.
Uses SQLite + bcrypt for credential storage.
"""

import os
import sqlite3
import hashlib
import hmac

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "users.db")


def _get_db():
    """Get a connection to the users database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _hash_password(password: str) -> str:
    """Hash a password with SHA-256 + salt."""
    salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return (salt + pwd_hash).hex()


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    stored_bytes = bytes.fromhex(stored_hash)
    salt = stored_bytes[:16]
    stored_pwd_hash = stored_bytes[16:]
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return hmac.compare_digest(pwd_hash, stored_pwd_hash)


def register_user(username: str, password: str) -> tuple:
    """
    Register a new user.
    Returns (success: bool, message: str).
    """
    if not username or not password:
        return False, "Username and password are required."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."

    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
        if cur.fetchone():
            return False, "Username already exists."

        hashed = _hash_password(password)
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True, "Account created successfully!"
    except Exception as e:
        return False, f"Registration failed: {e}"
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> tuple:
    """
    Authenticate a user.
    Returns (success: bool, message: str).
    """
    if not username or not password:
        return False, "Username and password are required."

    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        if not row:
            return False, "Invalid username or password."

        if _verify_password(password, row[0]):
            return True, "Login successful!"
        else:
            return False, "Invalid username or password."
    except Exception as e:
        return False, f"Authentication failed: {e}"
    finally:
        conn.close()
