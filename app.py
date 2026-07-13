from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse, urlencode

import requests
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, url_for, has_request_context, session, redirect
from werkzeug.middleware.proxy_fix import ProxyFix

try:
    import stripe
except ImportError:  # pragma: no cover - handled as config error at runtime
    stripe = None

from AuthService import (
    DEFAULT_CREDITS,
    add_credits,
    authenticate_user,
    create_user,
    ensure_auth_indexes,
    get_user_by_id,
    list_generation_history,
    plan_payload,
    record_generation_history,
    record_subscription_credit_refresh,
    serialize_user,
    spend_credit,
    update_user_billing,
    upsert_google_user,
)
from Database import get_database
from FluxInpaint import FluxInpaintError, apply_flux_inpaint
from OpenAIImageEdit import OpenAIImageEditError, apply_openai_image_edit
from ImageEdit import (
    ImageEditError,
    build_inpaint_mask,
    build_masked_region_edit,
    build_openai_edit_mask,
    composite_masked_patch,
    detect_color_recolor_request,
    recolor_masked_region,
)
from OpenRouterVideo import (
    DEFAULT_VIDEO_ASPECT_RATIO,
    DEFAULT_VIDEO_DURATION,
    DEFAULT_VIDEO_RESOLUTION,
    SUPPORTED_VIDEO_ASPECT_RATIOS,
    OPENROUTER_VIDEO_MODEL,
    OpenRouterVideoError,
    extract_video_url,
    get_video_status,
    get_video_content,
    submit_video_job,
)
from OrchestratedVideo import FirstFrameResult, VideoOrchestrationError, VisionCritique, orchestrate_video_first_frame
from PromptRewrite import PromptRewriteError, rewrite_prompt
from Storyboard import build_storyboard_frames, regenerate_storyboard_frame
from TexttoImage import DEFAULT_MODEL, SUPPORTED_ASPECT_RATIOS, build_pollinations_url, _is_pollinations_queue_or_rate_limit

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
# Railway and similar platforms terminate HTTPS at a reverse proxy before
# forwarding traffic to gunicorn over HTTP. ProxyFix makes Flask honor
# X-Forwarded-Proto/Host so url_for(..., _external=True) builds the same
# public HTTPS URL Google OAuth has allowlisted.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["PREFERRED_URL_SCHEME"] = os.getenv("PREFERRED_URL_SCHEME", "https")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.getenv("SECRET_KEY") or "dev-secret-change-me"

APP_DB = get_database()
if APP_DB is not None:
    try:
        ensure_auth_indexes(APP_DB)
    except Exception:
        app.logger.exception("Failed to create MongoDB auth indexes")

IMAGE_EDIT_JOBS: dict[str, dict] = {}
IMAGE_EDIT_JOBS_LOCK = threading.Lock()
IMAGE_EDIT_JOB_TTL_SECONDS = int(os.getenv("IMAGE_EDIT_JOB_TTL_SECONDS", "3600"))
VIDEO_GENERATION_JOBS: dict[str, dict] = {}
VIDEO_GENERATION_JOBS_LOCK = threading.Lock()

VIDEO_START_FRAME_SIZES = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
}

BILLING_PLANS = [
    {
        "id": "free",
        "name": "Free",
        "price": "$0",
        "period": "forever",
        "image_credits": DEFAULT_CREDITS["image"],
        "video_credits": DEFAULT_CREDITS["video"],
        "price_env": None,
        "features": ["Pollinations image generation", "OpenAI masked edits", "Starter video trials"],
        "cta": "Current starter plan",
    },
    {
        "id": "starter",
        "name": "Starter",
        "price": "$9",
        "period": "month",
        "image_credits": 250,
        "video_credits": 30,
        "price_env": "STRIPE_PRICE_STARTER",
        "features": ["More image generations", "More masked edits", "Priority history retention"],
        "cta": "Buy Starter",
    },
    {
        "id": "creator",
        "name": "Creator",
        "price": "$29",
        "period": "month",
        "image_credits": 1200,
        "video_credits": 150,
        "price_env": "STRIPE_PRICE_CREATOR",
        "features": ["Creator credit pool", "Storyboard workflow", "Higher video allowance"],
        "cta": "Buy Creator",
    },
    {
        "id": "pro",
        "name": "Pro",
        "price": "$99",
        "period": "month",
        "image_credits": 6000,
        "video_credits": 800,
        "price_env": "STRIPE_PRICE_PRO",
        "features": ["Team-scale credits", "Production usage", "Priority support placeholder"],
        "cta": "Buy Pro",
    },
]


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


STORYBOARD_IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

GENERATED_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def stable_generation_seed(prompt: str, aspect_ratio: str) -> int:
    digest = hashlib.sha256(f"{aspect_ratio}|{prompt.strip()}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def local_generated_file_from_url(image_url: str) -> Path | None:
    parsed = urlparse(image_url)
    path = unquote(parsed.path if parsed.scheme else image_url)
    prefix = "/static/generated/"
    if not path.startswith(prefix):
        return None
    filename = Path(path.removeprefix(prefix)).name
    if not filename:
        return None
    candidate = (GENERATED_DIR / filename).resolve()
    if GENERATED_DIR.resolve() not in candidate.parents and candidate != GENERATED_DIR.resolve():
        return None
    return candidate


def download_image_bytes(image_url: str, *, timeout: int = 45) -> tuple[bytes, str]:
    local_path = local_generated_file_from_url(image_url)
    if local_path is not None:
        if not local_path.exists():
            raise ImageEditError(f"Generated image file not found: {local_path.name}")
        return local_path.read_bytes(), GENERATED_CONTENT_TYPES.get(local_path.suffix.lower(), "image/png")
    response = requests.get(image_url, timeout=timeout)
    content_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0].lower()
    if response.status_code != 200:
        raise ImageEditError(f"Image download returned HTTP {response.status_code}: {response.text[:200]}")
    if not content_type.startswith("image/"):
        raise ImageEditError(f"Image download returned non-image content ({content_type}): {response.text[:200]}")
    return response.content, content_type


def materialize_masked_region_edit(edit_payload: dict) -> str:
    original_bytes, _original_type = download_image_bytes(edit_payload["image_url"])
    patch_bytes, _patch_type = download_image_bytes(edit_payload["patch_url"])
    edited_bytes, _content_type = composite_masked_patch(original_bytes, patch_bytes, edit_payload["mask"])

    digest = hashlib.sha256(
        f"{edit_payload['image_url']}|{edit_payload['patch_url']}|{edit_payload['mask']}".encode("utf-8")
    ).hexdigest()[:16]
    filename = f"masked-edit-{digest}.png"
    output_path = GENERATED_DIR / filename
    output_path.write_bytes(edited_bytes)
    return public_generated_url(filename)


def materialize_color_recolor_edit(image_url: str, mask: dict, target_rgb: tuple[int, int, int], target_name: str) -> str:
    original_bytes, _original_type = download_image_bytes(image_url)
    edited_bytes, _content_type = recolor_masked_region(original_bytes, mask, target_rgb)
    digest = hashlib.sha256(f"{image_url}|{mask}|{target_name}|color-recolor".encode("utf-8")).hexdigest()[:16]
    filename = f"masked-recolor-{digest}.png"
    output_path = GENERATED_DIR / filename
    output_path.write_bytes(edited_bytes)
    return public_generated_url(filename)


def write_generated_bytes(filename_prefix: str, digest_input: str, content: bytes, suffix: str = ".png") -> str:
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
    filename = f"{filename_prefix}-{digest}{suffix}"
    output_path = GENERATED_DIR / filename
    output_path.write_bytes(content)
    return public_generated_url(filename)


def materialize_flux_inpaint_edit(image_url: str, mask: dict, micro_prompt: str, context_prompt: str | None = None) -> dict:
    original_bytes, _original_type = download_image_bytes(image_url)
    mask_bytes, _mask_type = build_inpaint_mask(
        original_bytes,
        mask,
        feather_px=max(0, int(os.getenv("INPAINT_MASK_FEATHER_PX", "4"))),
    )
    digest_base = f"{image_url}|{mask}|{micro_prompt}|flux-inpaint"
    original_image_url = write_generated_bytes("inpaint-source", digest_base, original_bytes)
    mask_url = write_generated_bytes("inpaint-mask", digest_base, mask_bytes)
    prompt = build_masked_region_edit(
        image_url=image_url,
        micro_prompt=micro_prompt,
        mask=mask,
        context_prompt=context_prompt,
        model_choice=DEFAULT_MODEL,
    )["patch_prompt"]
    flux_image_url = apply_flux_inpaint(
        image_url=original_image_url,
        mask_url=mask_url,
        prompt=prompt,
        image_size=os.getenv("FAL_INPAINT_IMAGE_SIZE") or None,
    )
    final_bytes, _final_type = download_image_bytes(flux_image_url)
    edited_image_url = write_generated_bytes("flux-inpaint", f"{digest_base}|{flux_image_url}", final_bytes)
    return {
        "edited_image_url": edited_image_url,
        "original_image_url": original_image_url,
        "mask_url": mask_url,
        "flux_image_url": flux_image_url,
        "inpaint_prompt": prompt,
    }


def materialize_openai_image_edit(image_url: str, mask: dict, micro_prompt: str, context_prompt: str | None = None) -> dict:
    original_bytes, _original_type = download_image_bytes(image_url)
    mask_bytes, _mask_type = build_openai_edit_mask(
        original_bytes,
        mask,
        feather_px=max(0, int(os.getenv("INPAINT_MASK_FEATHER_PX", "4"))),
    )
    prompt = build_masked_region_edit(
        image_url=image_url,
        micro_prompt=micro_prompt,
        mask=mask,
        context_prompt=context_prompt,
        model_choice=DEFAULT_MODEL,
    )["patch_prompt"]
    edited_bytes = apply_openai_image_edit(
        image_bytes=original_bytes,
        mask_bytes=mask_bytes,
        prompt=prompt,
    )
    digest_base = f"{image_url}|{mask}|{micro_prompt}|openai-image-edit"
    edited_image_url = write_generated_bytes("openai-inpaint", digest_base, edited_bytes)
    return {
        "edited_image_url": edited_image_url,
        "inpaint_prompt": prompt,
    }


def selected_inpaint_provider() -> str:
    provider = os.getenv("INPAINT_PROVIDER", "openai").strip().lower()
    return provider or "openai"


def public_generated_url(filename: str) -> str:
    public_base_url = (os.getenv("PUBLIC_BASE_URL") or os.getenv("APP_BASE_URL") or "").strip().rstrip("/")
    path = f"/static/generated/{filename}"
    if has_request_context():
        path = url_for("static", filename=f"generated/{filename}")
    if public_base_url:
        return f"{public_base_url}{path}"
    if not has_request_context():
        return path
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip()
    forwarded_host = request.headers.get("X-Forwarded-Host", "").split(",", 1)[0].strip()
    if forwarded_host:
        scheme = forwarded_proto or request.scheme
        return f"{scheme}://{forwarded_host}{path}"
    return url_for("static", filename=f"generated/{filename}", _external=True)


def materialize_storyboard_frame(frame, *, timeout: int | None = None, attempts: int | None = None) -> str:
    """Download a Pollinations storyboard frame and expose it as a stable app-served image URL.

    Some I2V providers reject dynamic image-generator URLs with errors like
    "failed to process the file". Serving a normal static image file from the
    app gives OpenRouter/Wan a direct, stable image URL to fetch.

    Pollinations can briefly return queue/rate-limit text on the first fetch of
    a generated URL. Retry those transient responses before giving up so the
    first storyboard click is less likely to surface a provider warm-up page.
    """
    timeout = timeout if timeout is not None else int(os.getenv("STORYBOARD_MATERIALIZE_TIMEOUT", "12"))
    attempts = attempts if attempts is not None else int(os.getenv("STORYBOARD_MATERIALIZE_ATTEMPTS", "2"))
    last_error = "Pollinations did not return a usable storyboard image."
    for _attempt in range(max(1, attempts)):
        try:
            response = requests.get(frame.url, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"Storyboard image download failed: {exc}"
            continue
        content_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0].lower()
        body_preview = response.text[:300] if response.status_code != 200 or not content_type.startswith("image/") else ""
        if response.status_code == 200 and content_type.startswith("image/"):
            suffix = STORYBOARD_IMAGE_EXTENSIONS.get(content_type, ".jpg")
            digest = hashlib.sha256(f"{frame.stage}|{frame.seed}|{frame.url}".encode("utf-8")).hexdigest()[:16]
            filename = f"storyboard-{frame.stage}-{digest}{suffix}"
            output_path = GENERATED_DIR / filename
            output_path.write_bytes(response.content)
            return public_generated_url(filename)
        if response.status_code != 200:
            last_error = f"Storyboard image download returned HTTP {response.status_code}: {body_preview}"
        else:
            last_error = f"Storyboard image download returned non-image content ({content_type}): {body_preview}"
        if not _is_pollinations_queue_or_rate_limit(response.status_code, content_type, body_preview):
            break
    raise RuntimeError(last_error)


def storyboard_frame_payload(frame) -> dict:
    payload = frame.to_dict()
    payload["source_url"] = frame.url
    try:
        payload["url"] = materialize_storyboard_frame(frame)
    except Exception as exc:
        app.logger.warning(
            "Could not materialize storyboard frame %s; falling back to source URL: %s",
            frame.stage,
            exc,
        )
        payload["url"] = frame.url
        payload["materialization_error"] = str(exc)
    return payload


def render_index(**overrides):
    context = {
        "aspect_ratios": SUPPORTED_ASPECT_RATIOS.keys(),
        "selected_ratio": "1024x1024",
        "prompt": "",
        "image_url": None,
        "download_url": None,
        "error": None,
        "video_aspect_ratios": SUPPORTED_VIDEO_ASPECT_RATIOS,
        "selected_video_ratio": DEFAULT_VIDEO_ASPECT_RATIO,
        "video_prompt": "",
        "video_error": None,
        "default_video_duration": DEFAULT_VIDEO_DURATION,
        "default_generate_audio": False,
        "is_logged_in": bool(current_user_id()) if has_request_context() else False,
    }
    context.update(overrides)
    return render_template("index.html", **context)


@app.get("/")
def index():
    return render_index()


@app.get("/generate")
def generate_form_redirect():
    return render_index()


def current_db():
    return APP_DB


def require_db():
    db = current_db()
    if db is None:
        return None, (jsonify({"error": "MongoDB database is not configured. Set MONGODB_URI and MONGODB_PASSWORD."}), 503)
    return db, None


def current_user_id() -> str | None:
    user_id = session.get("user_id")
    return str(user_id) if user_id else None


def current_user():
    user_id = current_user_id()
    db = current_db()
    if not user_id or db is None:
        return None
    return get_user_by_id(db, user_id)


@app.post("/auth/register")
def register_user():
    db, error_response = require_db()
    if error_response:
        return error_response
    payload = request.get_json(silent=True) or request.form
    try:
        user = create_user(
            db,
            email=payload.get("email", ""),
            password=payload.get("password", ""),
            display_name=payload.get("display_name") or payload.get("name"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "Could not create user. The email may already be registered.", "detail": str(exc)}), 409
    session["user_id"] = str(user.get("_id"))
    return jsonify({"user": serialize_user(user)}), 201


@app.post("/auth/login")
def login_user():
    db, error_response = require_db()
    if error_response:
        return error_response
    payload = request.get_json(silent=True) or request.form
    user = authenticate_user(db, payload.get("email", ""), payload.get("password", ""))
    if not user:
        return jsonify({"error": "Invalid email or password."}), 401
    session["user_id"] = str(user.get("_id"))
    return jsonify({"user": serialize_user(user)})


@app.post("/auth/logout")
def logout_user():
    session.clear()
    return jsonify({"ok": True})


@app.get("/auth/me")
def auth_me():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in."}), 401
    return jsonify({"user": serialize_user(user)})


@app.get("/auth/history")
def auth_history():
    db, error_response = require_db()
    if error_response:
        return error_response
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Not logged in."}), 401
    limit = max(1, min(100, int(request.args.get("limit", "50"))))
    return jsonify({"items": list_generation_history(db, user_id=user_id, limit=limit)})


def plan_by_id(plan_id: str) -> dict | None:
    return next((plan for plan in BILLING_PLANS if plan["id"] == plan_id), None)


def stripe_secret_key() -> str:
    return (os.getenv("STRIPE_SECRET_KEY") or "").strip()


def stripe_webhook_secret() -> str:
    return (os.getenv("STRIPE_WEBHOOK_SECRET") or "").strip()


def stripe_transfer_destination_account() -> str:
    return (os.getenv("STRIPE_TRANSFER_DESTINATION_ACCOUNT") or "").strip()


def stripe_checkout_enabled() -> bool:
    return bool(stripe_secret_key()) and stripe is not None


def configure_stripe_api() -> None:
    if stripe is None:
        raise RuntimeError("Stripe SDK is not installed.")
    key = stripe_secret_key()
    if not key:
        raise RuntimeError("Set STRIPE_SECRET_KEY to enable Stripe Checkout.")
    stripe.api_key = key
    try:
        timeout = float(os.getenv("STRIPE_REQUEST_TIMEOUT", "20"))
        stripe.default_http_client = stripe.http_client.RequestsClient(timeout=timeout)
    except Exception:
        app.logger.exception("Could not configure Stripe HTTP timeout")


def stripe_success_url() -> str:
    return public_external_url("billing_success") + "?session_id={CHECKOUT_SESSION_ID}"


def stripe_cancel_url() -> str:
    return public_external_url("billing_page")


def checkout_session_payload_for_plan(plan: dict, user: dict) -> dict:
    metadata = {
        "user_id": str(user.get("_id")),
        "plan_id": plan["id"],
        "image_credits": str(plan["image_credits"]),
        "video_credits": str(plan["video_credits"]),
    }
    price_id = (os.getenv(plan.get("price_env") or "") or "").strip()
    line_item = {"quantity": 1}
    if price_id:
        line_item["price"] = price_id
    else:
        amount = {"starter": 900, "creator": 2900, "pro": 9900}.get(plan["id"])
        if amount is None:
            raise ValueError("Free plan does not require checkout.")
        line_item["price_data"] = {
            "currency": "usd",
            "unit_amount": amount,
            "recurring": {"interval": "month"},
            "product_data": {
                "name": f"{plan['name']} monthly credits",
                "description": f"Monthly refresh: {plan['image_credits']} image credits and {plan['video_credits']} video credits",
            },
        }
    subscription_data = {"metadata": metadata}
    destination = stripe_transfer_destination_account()
    if destination:
        subscription_data["transfer_data"] = {"destination": destination}
    return {
        "mode": "subscription",
        "customer_email": user.get("email"),
        "line_items": [line_item],
        "success_url": stripe_success_url(),
        "cancel_url": stripe_cancel_url(),
        "metadata": metadata,
        "subscription_data": subscription_data,
        "client_reference_id": str(user.get("_id")),
    }


def create_stripe_checkout_session(plan: dict, user: dict):
    configure_stripe_api()
    return stripe.checkout.Session.create(**checkout_session_payload_for_plan(plan, user))


def fulfill_checkout_session(checkout_session: dict) -> dict:
    metadata = checkout_session.get("metadata") or {}
    db = current_db()
    if db is None:
        raise RuntimeError("MongoDB database is not configured.")
    paid = checkout_session.get("payment_status") in {"paid", "no_payment_required"}
    if not paid:
        raise RuntimeError("Checkout session is not paid yet.")
    changed, user = record_subscription_credit_refresh(
        db,
        user_id=metadata.get("user_id", ""),
        event_id=f"checkout:{checkout_session.get('id', '')}",
        plan_id=metadata.get("plan_id", "free"),
        image_credits=int(metadata.get("image_credits") or 0),
        video_credits=int(metadata.get("video_credits") or 0),
        stripe_customer_id=checkout_session.get("customer"),
        stripe_subscription_id=checkout_session.get("subscription"),
        subscription_status="active",
    )
    return {"credited": changed, "user": serialize_user(user) if user else None}


def fulfill_paid_invoice(invoice: dict) -> dict:
    if invoice.get("billing_reason") == "subscription_create":
        return {"credited": False, "skipped": "subscription_create"}
    metadata = invoice.get("subscription_details", {}).get("metadata") or invoice.get("metadata") or {}
    db = current_db()
    if db is None:
        raise RuntimeError("MongoDB database is not configured.")
    changed, user = record_subscription_credit_refresh(
        db,
        user_id=metadata.get("user_id", ""),
        event_id=f"invoice:{invoice.get('id', '')}",
        plan_id=metadata.get("plan_id", "free"),
        image_credits=int(metadata.get("image_credits") or 0),
        video_credits=int(metadata.get("video_credits") or 0),
        stripe_customer_id=invoice.get("customer"),
        stripe_subscription_id=invoice.get("subscription"),
        subscription_status="active",
    )
    return {"credited": changed, "user": serialize_user(user) if user else None}


def current_billing_payload(user: dict) -> dict:
    serialized = serialize_user(user) or {}
    return {
        "user": serialized,
        "plan": serialized.get("plan") or plan_payload(user.get("plan_id")),
        "credits": serialized.get("credits") or DEFAULT_CREDITS.copy(),
        "plans": BILLING_PLANS,
        "checkout_enabled": stripe_checkout_enabled(),
    }


@app.get("/billing")
def billing_page():
    user = current_user()
    if not user:
        return redirect(url_for("login_page"))
    return render_template("billing.html", **current_billing_payload(user))


@app.get("/billing/status")
def billing_status():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in."}), 401
    return jsonify(current_billing_payload(user))


@app.post("/billing/checkout/<plan_id>")
def billing_checkout(plan_id: str):
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in."}), 401
    plan = plan_by_id(plan_id)
    if not plan or plan["id"] == "free":
        return jsonify({"error": "Choose a paid plan."}), 400
    try:
        checkout_session = create_stripe_checkout_session(plan, user)
    except Exception as exc:
        app.logger.exception("Stripe Checkout Session creation failed")
        return jsonify({"error": str(exc)}), 503
    return jsonify({"checkout_url": checkout_session.url, "session_id": checkout_session.id})


@app.post("/billing/cancel")
def billing_cancel():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in."}), 401
    subscription_id = (user.get("stripe_subscription_id") or "").strip()
    if not subscription_id:
        return jsonify({"error": "No active subscription to cancel."}), 400
    if stripe is None:
        return jsonify({"error": "Stripe SDK is not installed."}), 503
    if not stripe_secret_key():
        return jsonify({"error": "Set STRIPE_SECRET_KEY to enable subscription cancellation."}), 503
    try:
        configure_stripe_api()
        subscription = stripe.Subscription.modify(subscription_id, cancel_at_period_end=True)
        update_user_billing(current_db(), user, subscription_status="cancel_at_period_end")
    except Exception as exc:
        app.logger.exception("Stripe subscription cancellation failed")
        return jsonify({"error": str(exc)}), 503
    return jsonify(
        {
            "ok": True,
            "subscription_id": subscription_id,
            "cancel_at_period_end": bool(subscription.get("cancel_at_period_end", True)),
            "status": subscription.get("status", "active"),
        }
    )


@app.get("/billing/success")
def billing_success():
    session_id = (request.args.get("session_id") or "").strip()
    message = "Payment complete. Your credits will appear after Stripe confirms the checkout."
    if session_id and stripe is not None and stripe_secret_key():
        try:
            configure_stripe_api()
            checkout_session = stripe.checkout.Session.retrieve(session_id)
            fulfill_checkout_session(dict(checkout_session))
            message = "Payment complete. Credits added to your account."
        except Exception:
            app.logger.exception("Could not fulfill Stripe success redirect")
    return render_template("billing_success.html", message=message)


@app.post("/stripe/webhook")
def stripe_webhook():
    if stripe is None:
        return jsonify({"error": "Stripe SDK is not installed."}), 503
    payload = request.get_data()
    signature = request.headers.get("Stripe-Signature")
    secret = stripe_webhook_secret()
    if not secret:
        return jsonify({"error": "Set STRIPE_WEBHOOK_SECRET to enable Stripe webhook fulfillment."}), 503
    try:
        event = stripe.Webhook.construct_event(payload, signature, secret)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    if event.get("type") == "checkout.session.completed":
        fulfill_checkout_session(event["data"]["object"])
    elif event.get("type") == "invoice.paid":
        fulfill_paid_invoice(event["data"]["object"])
    return jsonify({"received": True})


@app.get("/login")
def login_page():
    return render_index(show_login=True)


def google_oauth_config() -> tuple[str | None, str | None]:
    return (os.getenv("GOOGLE_CLIENT_ID"), os.getenv("GOOGLE_CLIENT_SECRET"))


def public_external_url(endpoint: str, **values) -> str:
    """Build a public absolute URL that is safe behind HTTPS proxies.

    Flask's plain url_for(..., _external=True) can produce http:// URLs on
    Railway because gunicorn receives proxied HTTP even though users visit the
    public site over HTTPS. Google OAuth redirect URIs must match exactly, so
    prefer explicit public base env vars and otherwise rely on forwarded proxy
    headers handled by ProxyFix.
    """
    path = url_for(endpoint, **values)
    public_base_url = (
        os.getenv("PUBLIC_BASE_URL")
        or os.getenv("APP_BASE_URL")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or ""
    ).strip().rstrip("/")
    if public_base_url:
        if not public_base_url.startswith(("http://", "https://")):
            public_base_url = f"https://{public_base_url}"
        return f"{public_base_url}{path}"
    return url_for(endpoint, _external=True, **values)


def google_redirect_uri() -> str:
    explicit_redirect_uri = (os.getenv("GOOGLE_REDIRECT_URI") or "").strip()
    if explicit_redirect_uri:
        return explicit_redirect_uri
    return public_external_url("auth_google_callback")


@app.get("/auth/google")
def auth_google():
    db, error_response = require_db()
    if error_response:
        return error_response
    client_id, client_secret = google_oauth_config()
    if not client_id or not client_secret:
        return jsonify({"error": "Google login is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."}), 503
    state = secrets.token_urlsafe(24)
    session["google_oauth_state"] = state
    redirect_uri = google_redirect_uri()
    query = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
            "prompt": "select_account",
        }
    )
    return redirect(f"https://accounts.google.com/o/oauth2/v2/auth?{query}")


@app.get("/auth/google/callback")
def auth_google_callback():
    db, error_response = require_db()
    if error_response:
        return error_response
    expected_state = session.get("google_oauth_state")
    if not expected_state or request.args.get("state") != expected_state:
        return jsonify({"error": "Invalid Google login state."}), 400
    code = request.args.get("code")
    if not code:
        return jsonify({"error": "Google login did not return an authorization code."}), 400
    client_id, client_secret = google_oauth_config()
    if not client_id or not client_secret:
        return jsonify({"error": "Google login is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."}), 503

    token_response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": google_redirect_uri(),
            "grant_type": "authorization_code",
        },
        timeout=20,
    )
    if token_response.status_code != 200:
        return jsonify({"error": "Google token exchange failed.", "detail": token_response.text[:300]}), 502
    access_token = token_response.json().get("access_token")
    if not access_token:
        return jsonify({"error": "Google token exchange did not return an access token."}), 502

    userinfo_response = requests.get(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    if userinfo_response.status_code != 200:
        return jsonify({"error": "Google userinfo lookup failed.", "detail": userinfo_response.text[:300]}), 502
    profile = userinfo_response.json()
    user = upsert_google_user(
        db,
        google_id=profile.get("sub", ""),
        email=profile.get("email", ""),
        display_name=profile.get("name"),
        picture_url=profile.get("picture"),
    )
    session.pop("google_oauth_state", None)
    session["user_id"] = str(user.get("_id"))
    return redirect("/")


@app.post("/prompt/rewrite")
def rewrite_generation_prompt():
    payload = request.get_json(silent=True) or request.form
    prompt = (payload.get("prompt") or "").strip()
    media_type = (payload.get("media_type") or "image").strip().lower()
    aspect_ratio = (payload.get("aspect_ratio") or "").strip() or None

    if not prompt:
        return jsonify({"error": "Please enter a prompt before rewriting it."}), 400
    if media_type not in {"image", "video"}:
        media_type = "image"

    try:
        rewritten = rewrite_prompt(prompt, media_type=media_type, aspect_ratio=aspect_ratio)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except PromptRewriteError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        app.logger.exception("Unexpected error while rewriting prompt")
        return jsonify({"error": "Prompt rewrite failed.", "detail": str(exc)}), 500

    return jsonify(
        {
            "original_prompt": prompt,
            "rewritten_prompt": rewritten,
            "media_type": media_type,
            "aspect_ratio": aspect_ratio,
        }
    )


def require_and_spend_credit(credit_type: str):
    db, error_response = require_db()
    if error_response:
        return None, error_response
    user_id = current_user_id()
    if not user_id:
        return None, (jsonify({"error": "Login is required to generate media."}), 401)
    ok, user = spend_credit(db, user_id=user_id, credit_type=credit_type)
    if not ok:
        credits = serialize_user(user).get("credits", {}) if user else {}
        return None, (
            jsonify(
                {
                    "error": f"Not enough {credit_type} credits. Please buy more credits on the billing page.",
                    "credits": credits,
                    "billing_url": url_for("billing_page"),
                }
            ),
            402,
        )
    return user, None


def remember_video_job(job_id: str, metadata: dict) -> None:
    with VIDEO_GENERATION_JOBS_LOCK:
        VIDEO_GENERATION_JOBS[job_id] = {**metadata, "recorded": False, "created_at": time.time()}


def record_completed_video_history(job_id: str, video_url: str | None) -> None:
    if not video_url:
        return
    with VIDEO_GENERATION_JOBS_LOCK:
        job_meta = VIDEO_GENERATION_JOBS.get(job_id)
        if not job_meta or job_meta.get("recorded"):
            return
        VIDEO_GENERATION_JOBS[job_id]["recorded"] = True
    db = current_db()
    if db is None or not job_meta.get("user_id"):
        return
    try:
        record_generation_history(
            db,
            user_id=job_meta["user_id"],
            media_type="video",
            prompt=job_meta.get("prompt") or "Generated video",
            result_url=video_url,
            metadata={
                "optimized_prompt": job_meta.get("optimized_prompt"),
                "aspect_ratio": job_meta.get("aspect_ratio"),
                "model": job_meta.get("model"),
                "provider": "openrouter",
                "job_id": job_id,
            },
        )
    except Exception:
        app.logger.exception("Failed to record video generation history")


def record_masked_edit_history(*, user_id: str | None, edit_payload: dict, context_prompt: str | None = None) -> None:
    if not user_id or not edit_payload.get("edited_image_url"):
        return
    db = current_db()
    if db is None:
        return
    try:
        record_generation_history(
            db,
            user_id=user_id,
            media_type="image",
            prompt=edit_payload.get("micro_prompt") or context_prompt or "Masked image edit",
            result_url=edit_payload["edited_image_url"],
            metadata={
                "workflow": edit_payload.get("workflow"),
                "provider": "openai" if edit_payload.get("workflow") == "openai-image-edit-mask" else selected_inpaint_provider(),
                "source_image_url": edit_payload.get("image_url"),
                "context_prompt": context_prompt,
                "mask": edit_payload.get("mask"),
                "target_color": edit_payload.get("target_color"),
            },
        )
    except Exception:
        app.logger.exception("Failed to record masked image edit history")


def build_image_edit_payload(*, image_url: str, micro_prompt: str, mask: dict, context_prompt: str | None = None) -> dict:
    color_request = detect_color_recolor_request(micro_prompt)
    if color_request:
        color_prompt = (
            f"Change only the selected object's color to {color_request.target_name}. "
            "Preserve the exact same object shape, size, position, texture, lighting, shadows, and background. "
            "Do not add, remove, duplicate, resize, or redraw the object."
        )
        inpaint_payload = materialize_openai_image_edit(
            image_url=image_url,
            mask=mask,
            micro_prompt=color_prompt,
            context_prompt=context_prompt,
        )
        return {
            "image_url": image_url,
            "micro_prompt": micro_prompt,
            "mask": mask,
            "target_color": color_request.target_name,
            "workflow": "openai-image-edit-mask",
            **inpaint_payload,
        }

    provider = selected_inpaint_provider()
    if provider == "fal":
        inpaint_payload = materialize_flux_inpaint_edit(
            image_url=image_url,
            mask=mask,
            micro_prompt=micro_prompt,
            context_prompt=context_prompt,
        )
        workflow = "flux-inpainting-mask"
    elif provider == "openai":
        inpaint_payload = materialize_openai_image_edit(
            image_url=image_url,
            mask=mask,
            micro_prompt=micro_prompt,
            context_prompt=context_prompt,
        )
        workflow = "openai-image-edit-mask"
    else:
        raise ValueError("Unsupported INPAINT_PROVIDER. Use 'openai' or 'fal'.")
    return {
        "image_url": image_url,
        "micro_prompt": micro_prompt,
        "mask": mask,
        "workflow": workflow,
        **inpaint_payload,
    }


def prune_image_edit_jobs(now: float | None = None) -> None:
    now = now or time.time()
    with IMAGE_EDIT_JOBS_LOCK:
        stale_ids = [
            job_id
            for job_id, job in IMAGE_EDIT_JOBS.items()
            if now - float(job.get("created_at", now)) > IMAGE_EDIT_JOB_TTL_SECONDS
        ]
        for job_id in stale_ids:
            IMAGE_EDIT_JOBS.pop(job_id, None)


def start_image_edit_job(
    *,
    image_url: str,
    micro_prompt: str,
    mask: dict,
    context_prompt: str | None = None,
    user_id: str | None = None,
) -> str:
    prune_image_edit_jobs()
    job_id = f"edit_{uuid.uuid4().hex}"
    with IMAGE_EDIT_JOBS_LOCK:
        IMAGE_EDIT_JOBS[job_id] = {
            "status": "queued",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    def worker() -> None:
        with IMAGE_EDIT_JOBS_LOCK:
            if job_id in IMAGE_EDIT_JOBS:
                IMAGE_EDIT_JOBS[job_id]["status"] = "running"
                IMAGE_EDIT_JOBS[job_id]["updated_at"] = time.time()
        try:
            with app.app_context():
                result = build_image_edit_payload(
                    image_url=image_url,
                    micro_prompt=micro_prompt,
                    mask=mask,
                    context_prompt=context_prompt,
                )
                record_masked_edit_history(user_id=user_id, edit_payload=result, context_prompt=context_prompt)
            with IMAGE_EDIT_JOBS_LOCK:
                if job_id in IMAGE_EDIT_JOBS:
                    IMAGE_EDIT_JOBS[job_id].update(
                        {"status": "completed", "result": result, "error": None, "updated_at": time.time()}
                    )
        except Exception as exc:
            app.logger.exception("Background masked image edit failed")
            with IMAGE_EDIT_JOBS_LOCK:
                if job_id in IMAGE_EDIT_JOBS:
                    IMAGE_EDIT_JOBS[job_id].update(
                        {"status": "failed", "result": None, "error": str(exc), "updated_at": time.time()}
                    )

    threading.Thread(target=worker, daemon=True).start()
    return job_id


@app.post("/image/edit-region")
def edit_image_region():
    payload = request.get_json(silent=True) or request.form
    image_url = (payload.get("image_url") or "").strip()
    micro_prompt = (payload.get("micro_prompt") or payload.get("prompt") or "").strip()
    context_prompt = (payload.get("context_prompt") or "").strip() or None
    mask = payload.get("mask") or {}

    if not image_url:
        return jsonify({"error": "Image URL is required before editing a masked region."}), 400
    if not micro_prompt:
        return jsonify({"error": "Please enter a micro-prompt for the masked region."}), 400
    if not isinstance(mask, dict):
        return jsonify({"error": "A mask box is required before editing a region."}), 400

    _user, credit_error = require_and_spend_credit("image")
    if credit_error:
        return credit_error

    try:
        run_async = parse_bool(os.getenv("IMAGE_EDIT_ASYNC", "true")) and not parse_bool(payload.get("sync"))
        if run_async:
            job_id = start_image_edit_job(
                image_url=image_url,
                micro_prompt=micro_prompt,
                mask=mask,
                context_prompt=context_prompt,
                user_id=current_user_id(),
            )
            return jsonify(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "status_url": url_for("image_edit_region_status", job_id=job_id),
                    "workflow": "masked-image-edit-async",
                }
            ), 202

        edit_payload = build_image_edit_payload(
            image_url=image_url,
            micro_prompt=micro_prompt,
            mask=mask,
            context_prompt=context_prompt,
        )
        record_masked_edit_history(user_id=current_user_id(), edit_payload=edit_payload, context_prompt=context_prompt)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ImageEditError as exc:
        return jsonify({"error": str(exc)}), 422
    except FluxInpaintError as exc:
        return jsonify({"error": str(exc), "workflow": "flux-inpainting-mask"}), 502
    except OpenAIImageEditError as exc:
        return jsonify({"error": str(exc), "workflow": "openai-image-edit-mask"}), 502
    except Exception as exc:
        app.logger.exception("Unexpected error while editing masked image region")
        return jsonify({"error": "Masked image edit failed.", "detail": str(exc)}), 500

    return jsonify(edit_payload)


@app.get("/image/edit-region/status/<job_id>")
def image_edit_region_status(job_id: str):
    prune_image_edit_jobs()
    with IMAGE_EDIT_JOBS_LOCK:
        job = IMAGE_EDIT_JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Image edit job not found."}), 404
        payload = {
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "result": job.get("result"),
            "error": job.get("error"),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
        }
    return jsonify(payload)


@app.post("/generate")
def generate():
    prompt = request.form.get("prompt", "").strip()
    selected_ratio = request.form.get("aspect_ratio", "1024x1024")

    if selected_ratio not in SUPPORTED_ASPECT_RATIOS:
        selected_ratio = "1024x1024"

    if not prompt:
        return render_index(
            selected_ratio=selected_ratio,
            prompt=prompt,
            error="Please enter a prompt before generating an image.",
        ), 400

    width, height = SUPPORTED_ASPECT_RATIOS[selected_ratio]

    _user, credit_error = require_and_spend_credit("image")
    if credit_error:
        return render_index(
            selected_ratio=selected_ratio,
            prompt=prompt,
            error=credit_error[0].get_json().get("error", "Not enough image credits."),
        ), credit_error[1]

    # Build the Pollinations URL and let the browser load the image directly.
    # This avoids the Flask request hanging when WSL cannot resolve/reach
    # image.pollinations.ai server-side. It also keeps the UI responsive: the
    # page returns immediately, then the image itself loads in the browser.
    image_url = build_pollinations_url(
        prompt,
        model_choice=DEFAULT_MODEL,
        width=width,
        height=height,
        seed=stable_generation_seed(prompt, selected_ratio),
    )

    db = current_db()
    user_id = current_user_id()
    if db is not None and user_id:
        try:
            record_generation_history(
                db,
                user_id=user_id,
                media_type="image",
                prompt=prompt,
                result_url=image_url,
                metadata={"aspect_ratio": selected_ratio, "width": width, "height": height, "provider": "pollinations"},
            )
        except Exception:
            app.logger.exception("Failed to record image generation history")

    return render_index(
        selected_ratio=selected_ratio,
        prompt=prompt,
        image_url=image_url,
        download_url=image_url,
    )


@app.post("/video/storyboard")
def create_video_storyboard():
    payload = request.get_json(silent=True) or request.form
    prompt = (payload.get("prompt") or "").strip()
    selected_ratio = payload.get("aspect_ratio") or DEFAULT_VIDEO_ASPECT_RATIO

    if selected_ratio not in SUPPORTED_VIDEO_ASPECT_RATIOS:
        selected_ratio = DEFAULT_VIDEO_ASPECT_RATIO
    if not prompt:
        return jsonify({"error": "Please enter a prompt before creating a storyboard."}), 400

    frame_width, frame_height = VIDEO_START_FRAME_SIZES[selected_ratio]
    try:
        frames = build_storyboard_frames(
            prompt,
            aspect_ratio=selected_ratio,
            width=frame_width,
            height=frame_height,
            model_choice=DEFAULT_MODEL,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Unexpected error while creating video storyboard")
        return jsonify({"error": "Storyboard generation failed.", "detail": str(exc)}), 500

    return jsonify(
        {
            "prompt": prompt,
            "optimized_prompt": frames[0].prompt if frames else prompt,
            "aspect_ratio": selected_ratio,
            "frames": [storyboard_frame_payload(frame) for frame in frames],
            "workflow": "pollinations-three-frame-storyboard-before-i2v",
        }
    )


@app.post("/video/storyboard/frame")
def regenerate_video_storyboard_frame():
    payload = request.get_json(silent=True) or request.form
    prompt = (payload.get("prompt") or "").strip()
    base_prompt = (payload.get("base_prompt") or "").strip() or None
    selected_ratio = payload.get("aspect_ratio") or DEFAULT_VIDEO_ASPECT_RATIO
    stage = payload.get("stage") or "start"

    if selected_ratio not in SUPPORTED_VIDEO_ASPECT_RATIOS:
        selected_ratio = DEFAULT_VIDEO_ASPECT_RATIO
    if not prompt:
        return jsonify({"error": "Please enter a prompt before regenerating a storyboard frame."}), 400

    frame_width, frame_height = VIDEO_START_FRAME_SIZES[selected_ratio]
    try:
        frame = regenerate_storyboard_frame(
            prompt,
            stage=stage,
            aspect_ratio=selected_ratio,
            width=frame_width,
            height=frame_height,
            model_choice=DEFAULT_MODEL,
            base_prompt=base_prompt,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Unexpected error while regenerating storyboard frame")
        return jsonify({"error": "Storyboard frame regeneration failed.", "detail": str(exc)}), 500

    return jsonify({"frame": storyboard_frame_payload(frame), "workflow": "pollinations-storyboard-frame-regenerated"})


@app.post("/video/start")
def start_video_generation():
    payload = request.get_json(silent=True) or request.form
    prompt = (payload.get("prompt") or "").strip()
    selected_ratio = payload.get("aspect_ratio") or DEFAULT_VIDEO_ASPECT_RATIO
    generate_audio = parse_bool(payload.get("generate_audio"))

    if selected_ratio not in SUPPORTED_VIDEO_ASPECT_RATIOS:
        selected_ratio = DEFAULT_VIDEO_ASPECT_RATIO

    if not prompt:
        return jsonify({"error": "Please enter a prompt before generating a video."}), 400

    _user, credit_error = require_and_spend_credit("video")
    if credit_error:
        return credit_error
    user_id = str(_user.get("_id")) if _user else current_user_id()

    frame_width, frame_height = VIDEO_START_FRAME_SIZES[selected_ratio]
    storyboard_start_frame_url = (payload.get("storyboard_start_frame_url") or payload.get("start_frame_url") or "").strip()
    storyboard_optimized_prompt = (payload.get("optimized_prompt") or prompt).strip()

    if storyboard_start_frame_url:
        first_frame = FirstFrameResult(
            original_prompt=prompt,
            optimized_prompt=storyboard_optimized_prompt,
            start_frame_url=storyboard_start_frame_url,
            critique=VisionCritique(
                approved=True,
                confidence=1.0,
                reason="User-approved storyboard start frame selected before paid I2V submission.",
                improvements=[],
                raw_response="",
            ),
            attempts=1,
            width=frame_width,
            height=frame_height,
            seed=0,
        )
    else:
        try:
            first_frame = orchestrate_video_first_frame(
                prompt,
                aspect_ratio=selected_ratio,
                width=frame_width,
                height=frame_height,
                model_choice=DEFAULT_MODEL,
                max_attempts=max(1, int(os.environ.get("VIDEO_ORCHESTRATOR_MAX_ATTEMPTS", "1"))),
            )
        except VideoOrchestrationError as exc:
            return jsonify({"error": str(exc), "workflow": "vision-gated-i2v-blocked"}), 422
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            app.logger.exception("Unexpected error while preparing video first frame")
            return jsonify(
                {
                    "error": "Video generation failed before the paid I2V job was submitted.",
                    "detail": str(exc),
                    "workflow": "vision-gated-i2v-blocked",
                }
            ), 500

    try:
        job = submit_video_job(
            first_frame.optimized_prompt,
            aspect_ratio=selected_ratio,
            duration=DEFAULT_VIDEO_DURATION,
            resolution=DEFAULT_VIDEO_RESOLUTION,
            generate_audio=generate_audio,
            first_frame_url=first_frame.start_frame_url,
            timeout=max(5, int(os.environ.get("VIDEO_SUBMIT_TIMEOUT", "8"))),
        )
    except (OpenRouterVideoError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        app.logger.exception("Unexpected error while submitting OpenRouter video job")
        return jsonify(
            {
                "error": "Video generation failed while submitting the paid I2V job.",
                "detail": str(exc),
            }
        ), 500

    job_id_for_response = job.get("id")
    if job_id_for_response:
        remember_video_job(
            str(job_id_for_response),
            {
                "user_id": user_id,
                "prompt": prompt,
                "optimized_prompt": first_frame.optimized_prompt,
                "aspect_ratio": selected_ratio,
                "model": OPENROUTER_VIDEO_MODEL,
            },
        )

    return jsonify(
        {
            "id": job_id_for_response,
            "polling_url": job.get("polling_url"),
            "status": job.get("status", "pending"),
            "model": OPENROUTER_VIDEO_MODEL,
            "start_frame_url": first_frame.start_frame_url,
            "optimized_prompt": first_frame.optimized_prompt,
            "vision_critique": first_frame.critique.to_dict(),
            "frame_attempts": first_frame.attempts,
            "frame_seed": first_frame.seed,
            "generate_audio": generate_audio,
            "workflow": "pollinations-vision-gated-start-frame-to-openrouter-i2v",
        }
    ), 202


@app.get("/video/status/<path:job_id>")
def video_generation_status(job_id: str):
    try:
        status_payload = get_video_status(job_id)
    except (OpenRouterVideoError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        app.logger.exception("Unexpected error while checking OpenRouter video status")
        return jsonify(
            {
                "error": "Video status lookup failed.",
                "detail": str(exc),
            }
        ), 500

    status = status_payload.get("status", "unknown")
    video_url = extract_video_url(status_payload)
    job_id_for_response = status_payload.get("id", job_id)
    if status == "completed" and (
        not video_url or video_url.startswith("https://openrouter.ai/api/") or video_url.startswith("/api/")
    ):
        video_url = url_for("video_content", job_id=job_id_for_response)
    if status == "completed":
        record_completed_video_history(str(job_id_for_response), video_url)
    return jsonify(
        {
            "id": job_id_for_response,
            "status": status,
            "video_url": video_url,
            "error": status_payload.get("error"),
            "usage": status_payload.get("usage"),
        }
    )


@app.get("/video/content/<path:job_id>")
def video_content(job_id: str):
    try:
        content, content_type = get_video_content(job_id)
    except (OpenRouterVideoError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        app.logger.exception("Unexpected error while proxying OpenRouter video content")
        return jsonify({"error": "Video content download failed.", "detail": str(exc)}), 500
    return Response(content, mimetype=content_type)


@app.get("/download/<path:filename>")
def download_image(filename: str):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
