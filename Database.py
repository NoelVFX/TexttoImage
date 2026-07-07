from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus


def load_local_env(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_mongo_uri(uri_template: str | None, *, password: str | None = None) -> str | None:
    if not uri_template or not uri_template.strip():
        return None
    uri = uri_template.strip().strip('"').strip("'")
    for assignment_prefix in ("MONGODB_URI=", "MONGO_URI="):
        if uri.startswith(assignment_prefix):
            uri = uri.split("=", 1)[1].strip().strip('"').strip("'")
    password = password if password is not None else os.getenv("MONGODB_PASSWORD")
    for placeholder in ("<" + "db_password" + ">", "__PASSWORD__"):
        if placeholder in uri:
            if not password:
                raise RuntimeError("MONGODB_URI contains a password placeholder; set MONGODB_PASSWORD or replace the placeholder.")
            uri = uri.replace(placeholder, quote_plus(password))
    if not (uri.startswith("mongodb://") or uri.startswith("mongodb+srv://")):
        raise RuntimeError("MONGODB_URI must begin with mongodb:// or mongodb+srv://")
    return uri


def get_mongo_uri() -> str | None:
    load_local_env()
    try:
        return build_mongo_uri(os.getenv("MONGODB_URI") or os.getenv("MONGO_URI"))
    except RuntimeError:
        return None


def get_database():
    uri = get_mongo_uri()
    if not uri:
        return None
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise RuntimeError("pymongo is required for MongoDB support. Install requirements.txt.") from exc

    client = MongoClient(uri, serverSelectionTimeoutMS=int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000")))
    db_name = os.getenv("MONGODB_DB", "tti_app")
    return client[db_name]
