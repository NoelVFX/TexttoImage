from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_from_directory, url_for

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
from OrchestratedVideo import VideoOrchestrationError, orchestrate_video_first_frame
from PromptRewrite import PromptRewriteError, rewrite_prompt
from TexttoImage import DEFAULT_MODEL, SUPPORTED_ASPECT_RATIOS, build_pollinations_url

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

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
    )

    return render_index(
        selected_ratio=selected_ratio,
        prompt=prompt,
        image_url=image_url,
        download_url=image_url,
    )


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
