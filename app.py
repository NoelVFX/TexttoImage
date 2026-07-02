from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from OpenRouterVideo import (
    DEFAULT_VIDEO_ASPECT_RATIO,
    DEFAULT_VIDEO_DURATION,
    DEFAULT_VIDEO_RESOLUTION,
    SUPPORTED_VIDEO_ASPECT_RATIOS,
    OpenRouterVideoError,
    extract_video_url,
    get_video_status,
    submit_video_job,
)
from OrchestratedVideo import VideoOrchestrationError, orchestrate_video_first_frame
from TexttoImage import DEFAULT_MODEL, SUPPORTED_ASPECT_RATIOS, build_pollinations_url

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

VIDEO_START_FRAME_SIZES = {
    "1:1": (720, 720),
    "16:9": (1280, 720),
}


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
    }
    context.update(overrides)
    return render_template("index.html", **context)


@app.get("/")
def index():
    return render_index()


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
        )
    except VideoOrchestrationError as exc:
        return jsonify({"error": str(exc), "workflow": "vision-gated-i2v-blocked"}), 422
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        job = submit_video_job(
            first_frame.optimized_prompt,
            aspect_ratio=selected_ratio,
            duration=DEFAULT_VIDEO_DURATION,
            resolution=DEFAULT_VIDEO_RESOLUTION,
            generate_audio=False,
            first_frame_url=first_frame.start_frame_url,
        )
    except (OpenRouterVideoError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify(
        {
            "id": job.get("id"),
            "polling_url": job.get("polling_url"),
            "status": job.get("status", "pending"),
            "model": "bytedance/seedance-2.0-fast",
            "start_frame_url": first_frame.start_frame_url,
            "optimized_prompt": first_frame.optimized_prompt,
            "vision_critique": first_frame.critique.to_dict(),
            "frame_attempts": first_frame.attempts,
            "frame_seed": first_frame.seed,
            "workflow": "pollinations-vision-gated-start-frame-to-openrouter-i2v",
        }
    ), 202


@app.get("/video/status/<path:job_id>")
def video_generation_status(job_id: str):
    try:
        status_payload = get_video_status(job_id)
    except (OpenRouterVideoError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 502

    status = status_payload.get("status", "unknown")
    video_url = extract_video_url(status_payload)
    return jsonify(
        {
            "id": status_payload.get("id", job_id),
            "status": status,
            "video_url": video_url,
            "error": status_payload.get("error"),
            "usage": status_payload.get("usage"),
        }
    )


@app.get("/download/<path:filename>")
def download_image(filename: str):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=True)


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
