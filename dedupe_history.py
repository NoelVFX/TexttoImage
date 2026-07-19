#!/usr/bin/env python3
"""
De-duplicate generation_history rows created before the deterministic-URL fix.

The interim server-side generation stored a fresh random GridFS id on every
call, so repeat generations of the same prompt produced multiple history rows
that the app could no longer collapse (the "duplicate images" you saw). This
keeps the NEWEST row per (user_id, prompt, aspect_ratio) for text-to-image
generations (provider=pollinations) and removes the older redundant ones.

It only deletes history rows — it never touches GridFS image blobs — so it is
safe to run, and it is a DRY RUN unless you pass --apply.

Usage:
    python dedupe_history.py                 # preview what would be removed
    python dedupe_history.py --apply         # actually delete the redundant rows
    python dedupe_history.py --user <id>     # limit to one user (optional)

Requirements:
    - MONGODB_URI and MONGODB_PASSWORD available (in the environment or a local
      .env). Copy them from your Vercel project settings if needed.
    - Run from the project root. WSL/Linux is the easiest place to run it.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main():
    apply_changes = "--apply" in sys.argv
    only_user = None
    if "--user" in sys.argv:
        idx = sys.argv.index("--user")
        if idx + 1 < len(sys.argv):
            only_user = sys.argv[idx + 1]

    load_env()
    from Database import get_database

    db = get_database()
    if db is None:
        print("❌ MongoDB not configured (set MONGODB_URI / MONGODB_PASSWORD).")
        sys.exit(1)

    coll = db["generation_history"]
    query = {"media_type": "image", "metadata.provider": "pollinations"}
    if only_user:
        query["user_id"] = str(only_user)

    # Oldest first so items[-1] is the newest row we keep.
    docs = list(coll.find(query).sort("created_at", 1))

    groups: dict[tuple, list] = {}
    for d in docs:
        key = (
            str(d.get("user_id")),
            (d.get("prompt") or "").strip(),
            (d.get("metadata") or {}).get("aspect_ratio"),
        )
        groups.setdefault(key, []).append(d)

    to_delete = []
    dup_groups = 0
    examples = []
    for key, items in groups.items():
        if len(items) <= 1:
            continue
        dup_groups += 1
        # Keep the newest row; queue the rest for deletion.
        for d in items[:-1]:
            to_delete.append(d["_id"])
        if len(examples) < 15:
            examples.append((len(items), key))

    print("=" * 64)
    print(f"Scanned {len(docs)} text-to-image history rows in {len(groups)} prompt groups.")
    print(f"Found {dup_groups} groups with duplicates → {len(to_delete)} redundant rows.")

    if not to_delete:
        print("✅ Nothing to clean up.")
        return

    for count, key in examples:
        user_short = (key[0] or "")[:8]
        print(f"  • {count:>2}×  user={user_short}…  ratio={key[2]}  prompt={key[1][:56]!r}")

    if not apply_changes:
        print("\n(DRY RUN) No changes made. Re-run with --apply to delete the redundant rows.")
        return

    result = coll.delete_many({"_id": {"$in": to_delete}})
    print(f"\n🧹 Deleted {result.deleted_count} redundant history rows. Kept the newest per prompt.")


if __name__ == "__main__":
    main()
