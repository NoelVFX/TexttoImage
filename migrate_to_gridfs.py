#!/usr/bin/env python3
"""
Migrate existing local generated images to MongoDB GridFS.

Run this script ONCE after deploying the GridFS changes to persist
any existing local files that survived the container restart.

Usage:
    python migrate_to_gridfs.py

Requirements:
    - MONGODB_URI and MONGODB_PASSWORD set in environment
    - Run from the project root directory
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def load_env():
    """Load .env file if present."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def main():
    load_env()
    
    # Import after env is loaded
    from Database import get_database, store_image_in_gridfs
    from app import app
    
    with app.app_context():
        db = get_database()
        if db is None:
            print("❌ ERROR: MongoDB not configured (check MONGODB_URI/MONGODB_PASSWORD)")
            sys.exit(1)
        
        gen_dir = Path("static/generated")
        if not gen_dir.exists():
            print(f"❌ Directory not found: {gen_dir}")
            sys.exit(1)
        
        files = list(gen_dir.glob("*"))
        if not files:
            print("✅ No files to migrate")
            return
        
        print(f"📁 Found {len(files)} file(s) in {gen_dir}")
        print("=" * 60)
        
        migrated = 0
        failed = 0
        skipped = 0
        
        for f in files:
            if not f.is_file():
                continue
            
            try:
                content = f.read_bytes()
                
                # Determine content type from extension
                ext = f.suffix.lower()
                content_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                }
                content_type = content_types.get(ext, "image/png")
                
                # Store in GridFS
                file_id = store_image_in_gridfs(db, content, f.name, content_type=content_type)
                
                size_kb = len(content) / 1024
                print(f"✅ Migrated: {f.name} ({size_kb:.1f} KB) -> {file_id}")
                migrated += 1
                
            except Exception as e:
                print(f"❌ Failed: {f.name} - {e}")
                failed += 1
        
        print("=" * 60)
        print(f"Summary: {migrated} migrated, {failed} failed, {skipped} skipped")
        
        if migrated > 0:
            print("\n🎉 Migration complete! Images are now persisted in MongoDB GridFS.")
            print("   They will survive container restarts and redeploys.")
        elif failed == 0:
            print("\n✅ Nothing to migrate.")

if __name__ == "__main__":
    main()