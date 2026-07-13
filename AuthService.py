from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from werkzeug.security import check_password_hash, generate_password_hash

DEFAULT_PLAN_ID = "free"
DEFAULT_CREDITS = {"image": 25, "video": 3}
PLAN_NAMES = {
    "free": "Free",
    "starter": "Starter",
    "creator": "Creator",
    "pro": "Pro",
}


def plan_payload(plan_id: str | None) -> dict[str, str]:
    plan_id = (plan_id or DEFAULT_PLAN_ID).strip().lower() or DEFAULT_PLAN_ID
    return {"id": plan_id, "name": PLAN_NAMES.get(plan_id, plan_id.title())}


def credits_payload(credits: dict[str, Any] | None) -> dict[str, int]:
    merged = {**DEFAULT_CREDITS, **(credits or {})}
    return {"image": int(merged.get("image", 0)), "video": int(merged.get("video", 0))}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def ensure_auth_indexes(db) -> None:
    db["users"].create_index("email", unique=True)
    db["users"].create_index("google_id")
    db["generation_history"].create_index("user_id")
    db["generation_history"].create_index("created_at")
    db["credit_purchases"].create_index("stripe_session_id", unique=True)
    db["credit_refreshes"].create_index("event_id", unique=True)


def serialize_user(user: dict[str, Any] | None) -> dict[str, Any] | None:
    if not user:
        return None
    return {
        "id": str(user.get("_id")),
        "email": user.get("email"),
        "display_name": user.get("display_name") or user.get("email"),
        "plan": plan_payload(user.get("plan_id")),
        "credits": credits_payload(user.get("credits")),
        "stripe_subscription_id": user.get("stripe_subscription_id"),
        "subscription_status": user.get("subscription_status"),
        "created_at": user.get("created_at").isoformat() if hasattr(user.get("created_at"), "isoformat") else user.get("created_at"),
    }


def create_user(db, *, email: str, password: str, display_name: str | None = None) -> dict[str, Any]:
    email = normalize_email(email)
    if not email or "@" not in email:
        raise ValueError("A valid email is required.")
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    now = utc_now()
    document = {
        "email": email,
        "display_name": (display_name or email).strip(),
        "password_hash": generate_password_hash(password),
        "plan_id": DEFAULT_PLAN_ID,
        "credits": DEFAULT_CREDITS.copy(),
        "created_at": now,
        "updated_at": now,
    }
    result = db["users"].insert_one(document)
    document["_id"] = result.inserted_id
    return document


def authenticate_user(db, email: str, password: str) -> dict[str, Any] | None:
    user = db["users"].find_one({"email": normalize_email(email)})
    if not user:
        return None
    if not check_password_hash(user.get("password_hash", ""), password or ""):
        return None
    return user


def get_user_by_id(db, user_id: str) -> dict[str, Any] | None:
    # ObjectId conversion is intentionally optional so tests/fakes and string ids work.
    candidates = [user_id]
    try:
        from bson import ObjectId
        if ObjectId.is_valid(user_id):
            candidates.insert(0, ObjectId(user_id))
    except Exception:
        pass
    for candidate in candidates:
        user = db["users"].find_one({"_id": candidate})
        if user:
            return user
    return None


def upsert_google_user(db, *, google_id: str, email: str, display_name: str | None = None, picture_url: str | None = None) -> dict[str, Any]:
    email = normalize_email(email)
    if not google_id:
        raise ValueError("Google user id is required.")
    if not email or "@" not in email:
        raise ValueError("Google account did not return a valid email.")
    now = utc_now()
    existing = db["users"].find_one({"google_id": google_id}) or db["users"].find_one({"email": email})
    update = {
        "google_id": google_id,
        "email": email,
        "display_name": (display_name or email).strip(),
        "picture_url": picture_url,
        "auth_provider": "google",
        "updated_at": now,
    }
    if existing:
        db["users"].update_one({"_id": existing["_id"]}, {"$set": update})
        existing.update(update)
        return existing
    document = {**update, "plan_id": DEFAULT_PLAN_ID, "credits": DEFAULT_CREDITS.copy(), "created_at": now}
    result = db["users"].insert_one(document)
    document["_id"] = result.inserted_id
    return document


def record_generation_history(
    db,
    *,
    user_id: str,
    media_type: str,
    prompt: str,
    result_url: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = db["generation_history"].find_one(
        {"user_id": str(user_id), "media_type": media_type, "result_url": result_url}
    )
    if existing:
        return existing
    now = utc_now()
    document = {
        "user_id": str(user_id),
        "media_type": media_type,
        "prompt": prompt,
        "result_url": result_url,
        "metadata": metadata or {},
        "created_at": now,
    }
    result = db["generation_history"].insert_one(document)
    document["_id"] = result.inserted_id
    return document


def serialize_history_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(item.get("_id")),
        "user_id": str(item.get("user_id")),
        "media_type": item.get("media_type"),
        "prompt": item.get("prompt"),
        "result_url": item.get("result_url"),
        "metadata": item.get("metadata") or {},
        "created_at": item.get("created_at").isoformat() if hasattr(item.get("created_at"), "isoformat") else item.get("created_at"),
    }


def list_generation_history(db, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    cursor = db["generation_history"].find({"user_id": str(user_id)}).sort("created_at", -1).limit(limit)
    return [serialize_history_item(item) for item in cursor]


def update_user_billing(
    db,
    user: dict[str, Any],
    *,
    plan_id: str | None = None,
    credits: dict[str, int] | None = None,
    stripe_customer_id: str | None = None,
    stripe_subscription_id: str | None = None,
    subscription_status: str | None = None,
) -> dict[str, Any]:
    update: dict[str, Any] = {"updated_at": utc_now()}
    if plan_id is not None:
        update["plan_id"] = plan_payload(plan_id)["id"]
    if credits is not None:
        update["credits"] = credits_payload(credits)
    if stripe_customer_id is not None:
        update["stripe_customer_id"] = stripe_customer_id
    if stripe_subscription_id is not None:
        update["stripe_subscription_id"] = stripe_subscription_id
    if subscription_status is not None:
        update["subscription_status"] = subscription_status
    db["users"].update_one({"_id": user["_id"]}, {"$set": update})
    updated = {**user, **update}
    return updated


def add_credits(db, *, user_id: str, image: int = 0, video: int = 0, plan_id: str | None = None) -> dict[str, Any] | None:
    user = get_user_by_id(db, user_id)
    if not user:
        return None
    credits = credits_payload(user.get("credits"))
    credits["image"] = max(0, credits["image"] + int(image))
    credits["video"] = max(0, credits["video"] + int(video))
    return update_user_billing(db, user, plan_id=plan_id, credits=credits)


def spend_credit(db, *, user_id: str, credit_type: str, amount: int = 1) -> tuple[bool, dict[str, Any] | None]:
    if credit_type not in {"image", "video"}:
        raise ValueError("credit_type must be 'image' or 'video'.")
    user = get_user_by_id(db, user_id)
    if not user:
        return False, None
    credits = credits_payload(user.get("credits"))
    amount = max(1, int(amount))
    if credits.get(credit_type, 0) < amount:
        return False, user
    credits[credit_type] -= amount
    return True, update_user_billing(db, user, credits=credits)


def record_credit_purchase(
    db,
    *,
    user_id: str,
    stripe_session_id: str,
    plan_id: str,
    image_credits: int,
    video_credits: int,
) -> tuple[bool, dict[str, Any] | None]:
    existing = db["credit_purchases"].find_one({"stripe_session_id": stripe_session_id})
    if existing:
        return False, get_user_by_id(db, user_id)
    now = utc_now()
    db["credit_purchases"].insert_one(
        {
            "user_id": str(user_id),
            "stripe_session_id": stripe_session_id,
            "plan_id": plan_id,
            "image_credits": int(image_credits),
            "video_credits": int(video_credits),
            "created_at": now,
        }
    )
    return True, add_credits(db, user_id=user_id, image=image_credits, video=video_credits, plan_id=plan_id)


def record_subscription_credit_refresh(
    db,
    *,
    user_id: str,
    event_id: str,
    plan_id: str,
    image_credits: int,
    video_credits: int,
    stripe_customer_id: str | None = None,
    stripe_subscription_id: str | None = None,
    subscription_status: str | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    existing = db["credit_refreshes"].find_one({"event_id": event_id})
    if existing:
        return False, get_user_by_id(db, user_id)
    user = get_user_by_id(db, user_id)
    if not user:
        return False, None
    credits = {"image": int(image_credits), "video": int(video_credits)}
    updated = update_user_billing(
        db,
        user,
        plan_id=plan_id,
        credits=credits,
        stripe_customer_id=stripe_customer_id,
        stripe_subscription_id=stripe_subscription_id,
        subscription_status=subscription_status,
    )
    db["credit_refreshes"].insert_one(
        {
            "user_id": str(user_id),
            "event_id": event_id,
            "plan_id": plan_id,
            "image_credits": int(image_credits),
            "video_credits": int(video_credits),
            "stripe_customer_id": stripe_customer_id,
            "stripe_subscription_id": stripe_subscription_id,
            "subscription_status": subscription_status,
            "created_at": utc_now(),
        }
    )
    return True, updated
