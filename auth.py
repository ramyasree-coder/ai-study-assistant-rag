"""
auth.py
Authentication & Per-User Session Isolation for AI Study Assistant

Features:
- Local hashed user storage in users.json (using bcrypt)
- Safe password hashing and constant-time verification
- Signup and login validation
- Per-user data directory (user_data/{username}) for isolated FAISS indices and chat history
- Session state serialization and restoration across login sessions
"""

import os
import json
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import bcrypt

USERS_FILE = Path(__file__).parent / "users.json"
USER_DATA_DIR = Path(__file__).parent / "user_data"


# ---------------------------------------------------------------------------
# 1. User Credential Store & Hashing
# ---------------------------------------------------------------------------
def _load_users_db() -> Dict[str, Dict[str, Any]]:
    """Load users database from users.json."""
    if not USERS_FILE.exists():
        USERS_FILE.write_text("{}", encoding="utf-8")
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users_db(users: Dict[str, Dict[str, Any]]) -> None:
    """Save users database to users.json."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def hash_password(password: str) -> str:
    """Generate secure bcrypt salt and hash for the password."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify plain password against stored bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def signup_user(username: str, password: str, full_name: str = "") -> Tuple[bool, str]:
    """
    Register a new user account.
    Returns: (success: bool, message: str)
    """
    clean_username = username.strip().lower()
    if not clean_username:
        return False, "Username cannot be empty."
    if len(clean_username) < 3:
        return False, "Username must be at least 3 characters long."
    if not clean_username.isalnum() and "_" not in clean_username:
        return False, "Username can only contain alphanumeric characters and underscores."
    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters long."

    users = _load_users_db()
    if clean_username in users:
        return False, f"Username '{clean_username}' already exists. Please choose another."

    pwd_hash = hash_password(password)
    users[clean_username] = {
        "username": clean_username,
        "full_name": full_name.strip() or clean_username.capitalize(),
        "password_hash": pwd_hash,
    }
    _save_users_db(users)

    # Initialize user's isolated data directory
    user_dir = get_user_dir(clean_username)
    user_dir.mkdir(parents=True, exist_ok=True)

    return True, f"Account created successfully for '{clean_username}'! You can now log in."


def login_user(username: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Authenticate a user.
    Returns: (success: bool, message: str, user_dict: Optional[dict])
    """
    clean_username = username.strip().lower()
    if not clean_username or not password:
        return False, "Please enter both username and password.", None

    users = _load_users_db()
    user_info = users.get(clean_username)
    if not user_info:
        return False, "Invalid username or password.", None

    if not verify_password(password, user_info.get("password_hash", "")):
        return False, "Invalid username or password.", None

    return True, "Login successful!", {
        "username": clean_username,
        "full_name": user_info.get("full_name", clean_username.capitalize()),
    }


# ---------------------------------------------------------------------------
# 2. Per-User Workspace & Session Persistence
# ---------------------------------------------------------------------------
def get_user_dir(username: str) -> Path:
    """Get or create user-specific isolated storage directory."""
    clean_username = "".join(c for c in username.strip().lower() if c.isalnum() or c in ("_", "-"))
    user_dir = USER_DATA_DIR / clean_username
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def save_user_state(
    username: str,
    messages: List[Dict[str, Any]],
    indexed_info: Dict[str, Any],
) -> None:
    """Save user chat history and knowledge base metadata to user directory."""
    if not username:
        return
    user_dir = get_user_dir(username)
    session_file = user_dir / "session.json"
    
    data = {
        "messages": messages,
        "indexed_info": indexed_info,
    }
    try:
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving user session for {username}: {e}")


def load_user_state(username: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load user chat history and knowledge base metadata.
    Returns: (messages, indexed_info)
    """
    default_indexed = {
        "status": "Not Built",
        "chunks_count": 0,
        "sources": [],
    }
    if not username:
        return [], default_indexed

    user_dir = get_user_dir(username)
    session_file = user_dir / "session.json"
    if not session_file.exists():
        return [], default_indexed

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", []), data.get("indexed_info", default_indexed)
    except Exception as e:
        print(f"Error loading user session for {username}: {e}")
        return [], default_indexed


def clear_user_data(username: str) -> None:
    """Clear all persistent session data for a user."""
    if not username:
        return
    user_dir = get_user_dir(username)
    session_file = user_dir / "session.json"
    if session_file.exists():
        try:
            session_file.unlink()
        except Exception:
            pass
