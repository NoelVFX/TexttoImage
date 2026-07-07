from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

import requests
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, url_for, has_request_context, session, redirect

from AuthService import (
    authenticate_user,
    create_user,
    ensure_auth_indexes,
    get_user_by_id,
    list_generation_history,
    record_generation_history,
    serialize_user,
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
from TexttoImage import DEFAULT_MODEL, SUPPORTED_ASPECT_RATIOS, build_pollinations_url

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
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

VIDEO_START_FRAME_SIZES = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
}


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


def stable_generation_seed(prompt: str, aspect_ratio: str) -> int:
    digest = hashlib.sha256(f"{aspect_ratio}|{prompt.strip()}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def download_image_bytes(image_url: str, *, timeout: int = 45) -> tuple[bytes, str]:
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


def materialize_storyboard_frame(frame, *, timeout: int = 30) -> str:
    """Download a Pollinations storyboard frame and expose it as a stable app-served image URL.

    Some I2V providers reject dynamic image-generator URLs with errors like
    "failed to process the file". Serving a normal static image file from the
    app gives OpenRouter/Wan a direct, stable image URL to fetch.
    """
    response = requests.get(frame.url, timeout=timeout)
    content_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0].lower()
    if response.status_code != 200:
        raise RuntimeError(f"Storyboard image download returned HTTP {response.status_code}: {response.text[:200]}")
    if not content_type.startswith("image/"):
        raise RuntimeError(f"Storyboard image download returned non-image content ({content_type}): {response.text[:200]}")

    suffix = STORYBOARD_IMAGE_EXTENSIONS.get(content_type, ".jpg")
    digest = hashlib.sha256(f"{frame.stage}|{frame.seed}|{frame.url}".encode("utf-8")).hexdigest()[:16]
    filename = f"storyboard-{frame.stage}-{digest}{suffix}"
    output_path = GENERATED_DIR / filename
    output_path.write_bytes(response.content)
    return public_generated_url(filename)


def storyboard_frame_payload(frame) -> dict:
    payload = frame.to_dict()
    payload["source_url"] = frame.url
    payload["url"] = materialize_storyboard_frame(frame)
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


@app.get("/login")
def login_page():
    return render_index(show_login=True)


def google_oauth_config() -> tuple[str | None, str | None]:
    return (os.getenv("GOOGLE_CLIENT_ID"), os.getenv("GOOGLE_CLIENT_SECRET"))


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
    redirect_uri = url_for("auth_google_callback", _external=True)
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
            "redirect_uri": url_for("auth_google_callback", _external=True),
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


def start_image_edit_job(*, image_url: str, micro_prompt: str, mask: dict, context_prompt: str | None = None) -> str:
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

    try:
        run_async = parse_bool(os.getenv("IMAGE_EDIT_ASYNC", "true")) and not parse_bool(payload.get("sync"))
        if run_async:
            job_id = start_image_edit_job(
                image_url=image_url,
                micro_prompt=micro_prompt,
                mask=mask,
                context_prompt=context_prompt,
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

    return jsonify(
        {
            "id": job.get("id"),
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
