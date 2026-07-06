import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app
from TexttoImage import build_pollinations_url


class MaskedImageEditTests(unittest.TestCase):
    def test_normalise_mask_box_clamps_and_rounds_values(self):
        from ImageEdit import normalise_mask_box

        box = normalise_mask_box({"x": -4, "y": 10.6, "width": 120.2, "height": 0}, image_width=300, image_height=200)

        self.assertEqual(box, {"x": 0, "y": 11, "width": 120, "height": 1})

    def test_build_masked_region_edit_returns_patch_url_and_preserves_mask(self):
        from ImageEdit import build_masked_region_edit

        result = build_masked_region_edit(
            image_url="https://example.com/original.jpg",
            micro_prompt="open this door",
            mask={"x": 20, "y": 30, "width": 180, "height": 120},
            context_prompt="a blue house with a closed red door",
        )

        self.assertEqual(result["mask"]["width"], 180)
        self.assertEqual(result["mask"]["height"], 120)
        self.assertIn("open%20this%20door", result["patch_url"])
        self.assertIn("background%20unchanged", result["patch_url"])
        self.assertIn("width=180", result["patch_url"])
        self.assertIn("height=120", result["patch_url"])
        self.assertIn("same object scale", result["patch_prompt"])
        self.assertIn("match surrounding grass", result["patch_prompt"])
        self.assertIn("preserve original background", result["patch_prompt"])


class StoryboardGenerationTests(unittest.TestCase):
    def test_build_storyboard_frames_returns_start_middle_end_frames(self):
        from Storyboard import build_storyboard_frames

        frames = build_storyboard_frames(
            "a monkey holding colorful eggs",
            aspect_ratio="16:9",
            width=1280,
            height=720,
            prompt_optimizer=lambda prompt, *, aspect_ratio: "cinematic monkey with colorful eggs",
        )

        self.assertEqual([frame.stage for frame in frames], ["start", "middle", "end"])
        self.assertEqual([frame.label for frame in frames], ["Start", "Middle", "End"])
        self.assertEqual(len({frame.seed for frame in frames}), 3)
        self.assertTrue(all("width=1280" in frame.url for frame in frames))
        self.assertTrue(all("height=720" in frame.url for frame in frames))
        self.assertIn("opening", frames[0].prompt.lower())
        self.assertIn("midpoint", frames[1].prompt.lower())
        self.assertIn("ending", frames[2].prompt.lower())

    def test_regenerate_storyboard_frame_uses_custom_frame_prompt(self):
        from Storyboard import regenerate_storyboard_frame

        frame = regenerate_storyboard_frame(
            "make the monkey jump higher",
            stage="middle",
            aspect_ratio="9:16",
            width=720,
            height=1280,
        )

        self.assertEqual(frame.stage, "middle")
        self.assertEqual(frame.label, "Middle")
        self.assertIn("make the monkey jump higher", frame.prompt)
        self.assertIn("width=720", frame.url)
        self.assertIn("height=1280", frame.url)



class PollinationsUrlTests(unittest.TestCase):
    def test_build_pollinations_url_includes_seed_when_provided(self):
        url = build_pollinations_url("gold coins", width=720, height=720, seed=12345)

        self.assertIn("seed=12345", url)
        self.assertIn("width=720", url)
        self.assertIn("height=720", url)

    def test_build_pollinations_url_includes_token_when_configured(self):
        with patch.dict(os.environ, {"POLLINATIONS_TOKEN": "token_123"}, clear=False):
            url = build_pollinations_url("gold coins", width=720, height=720, seed=12345)

        self.assertIn("token=token_123", url)

    @patch("TexttoImage.requests.get")
    def test_generate_pollinations_image_bytes_retries_fallback_model_after_queue_message(self, mock_get):
        from TexttoImage import generate_pollinations_image_bytes

        queued = Mock()
        queued.status_code = 429
        queued.headers = {"content-type": "text/plain"}
        queued.text = "Too many requests, 1 request queued. Get unlimited access at https://auth.pollinations.ai"
        image = Mock()
        image.status_code = 200
        image.headers = {"content-type": "image/jpeg"}
        image.content = b"fake image bytes"
        image.text = ""
        mock_get.side_effect = [queued, image]

        with patch.dict(os.environ, {"POLLINATIONS_TOKEN": ""}, clear=False):
            content, content_type = generate_pollinations_image_bytes(
                "gold coins",
                model_choice="gpt-image-large",
                width=720,
                height=720,
            )

        self.assertEqual(content, b"fake image bytes")
        self.assertEqual(content_type, "image/jpeg")
        self.assertEqual(mock_get.call_count, 2)
        self.assertIn("model=turbo", mock_get.call_args_list[1].args[0])


class PromptRewriteTests(unittest.TestCase):
    def test_build_prompt_rewrite_command_uses_hermes_quiet_one_shot(self):
        from PromptRewrite import build_prompt_rewrite_command

        command = build_prompt_rewrite_command(
            "A castle on a mountain",
            media_type="image",
            aspect_ratio="1792x1024",
            provider="openrouter",
            model="openai/gpt-4o-mini",
        )

        self.assertEqual(command[:3], ["hermes", "chat", "-Q"])
        self.assertIn("--ignore-rules", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--source", command)
        self.assertIn("tool", command)
        self.assertIn("--max-turns", command)
        self.assertIn("1", command)
        self.assertIn("--provider", command)
        self.assertIn("openrouter", command)
        self.assertIn("-m", command)
        self.assertIn("openai/gpt-4o-mini", command)
        self.assertEqual(command[-2], "-q")
        self.assertIn("lighting", command[-1].lower())
        self.assertIn("camera", command[-1].lower())
        self.assertIn("A castle on a mountain", command[-1])

    def test_rewrite_prompt_returns_clean_ai_text(self):
        from PromptRewrite import rewrite_prompt

        def fake_runner(command, **kwargs):
            class Result:
                returncode = 0
                stdout = "```\nCinematic wide shot of a castle on a mountain, golden hour lighting, detailed stonework, dramatic clouds, ultra high quality.\n```"
                stderr = ""

            return Result()

        rewritten = rewrite_prompt(
            "castle",
            media_type="image",
            aspect_ratio="1792x1024",
            runner=fake_runner,
        )

        self.assertIn("golden hour lighting", rewritten)
        self.assertNotIn("```", rewritten)

    def test_rewrite_prompt_removes_tirith_scanner_warning(self):
        from PromptRewrite import rewrite_prompt

        def fake_runner(command, **kwargs):
            class Result:
                returncode = 0
                stdout = (
                    "⚠ tirith security scanner enabled but not available — command scanning will use pattern matching only\n"
                    "Cinematic close-up of a monkey holding colorful eggs in a lush jungle, warm golden light, low-angle camera, vibrant playful mood, crisp high-detail textures."
                )
                stderr = ""

            return Result()

        rewritten = rewrite_prompt(
            "monkey with eggs",
            media_type="image",
            aspect_ratio="1024x1024",
            runner=fake_runner,
        )

        self.assertNotIn("tirith", rewritten.lower())
        self.assertNotIn("security scanner", rewritten.lower())
        self.assertTrue(rewritten.startswith("Cinematic close-up"))

    def test_rewrite_prompt_limits_overly_long_output(self):
        from PromptRewrite import rewrite_prompt

        long_prompt = " ".join(f"word{i}" for i in range(90))

        def fake_runner(command, **kwargs):
            class Result:
                returncode = 0
                stdout = long_prompt
                stderr = ""

            return Result()

        rewritten = rewrite_prompt(
            "short idea",
            media_type="video",
            aspect_ratio="16:9",
            runner=fake_runner,
        )

        self.assertLessEqual(len(rewritten.split()), 45)


class HermesReviewBackendTests(unittest.TestCase):
    def test_build_hermes_review_command_attaches_image_and_forces_json_friendly_mode(self):
        from OrchestratedVideo import build_hermes_review_command

        command = build_hermes_review_command(
            image_path="/tmp/storyboard.jpg",
            prompt="review prompt",
            provider="openrouter",
            model="openai/gpt-4o-mini",
        )

        self.assertEqual(command[:3], ["hermes", "chat", "-Q"])
        self.assertIn("--ignore-rules", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--source", command)
        self.assertIn("tool", command)
        self.assertIn("--max-turns", command)
        self.assertIn("1", command)
        self.assertIn("--image", command)
        self.assertIn("/tmp/storyboard.jpg", command)
        self.assertIn("--provider", command)
        self.assertIn("openrouter", command)
        self.assertIn("-m", command)
        self.assertIn("openai/gpt-4o-mini", command)
        self.assertEqual(command[-2:], ["-q", "review prompt"])

    def test_build_hermes_review_command_defaults_to_openrouter_vision_model(self):
        from OrchestratedVideo import build_hermes_review_command

        command = build_hermes_review_command(
            image_path="/tmp/storyboard.jpg",
            prompt="review prompt",
        )

        self.assertIn("--provider", command)
        self.assertIn("openrouter", command)
        self.assertIn("-m", command)
        self.assertIn("openai/gpt-4o-mini", command)

    def test_critique_storyboard_image_uses_hermes_cli_and_parses_json(self):
        from OrchestratedVideo import critique_storyboard_image

        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))

            class Result:
                returncode = 0
                stdout = json.dumps(
                    {
                        "approved": True,
                        "confidence": 0.88,
                        "reason": "Matches the shiny coin UI prompt with no visible stretching.",
                        "improvements": [],
                    }
                )
                stderr = ""

            return Result()

        image_bytes = b"\xff\xd8" + b"x" * 4096
        with tempfile.TemporaryDirectory() as tmpdir:
            critique = critique_storyboard_image(
                image_bytes,
                "image/jpeg",
                user_intent="shiny gold coins",
                optimized_prompt="crisp shiny gold coin game UI",
                aspect_ratio="1:1",
                reviewer="hermes",
                runner=fake_runner,
                temp_dir=Path(tmpdir),
            )

        self.assertTrue(critique.approved)
        self.assertEqual(critique.confidence, 0.88)
        self.assertIn("shiny coin", critique.reason)
        self.assertEqual(len(calls), 1)
        command, kwargs = calls[0]
        self.assertIn("hermes", command[0])
        self.assertIn("--image", command)
        image_path = Path(command[command.index("--image") + 1])
        self.assertEqual(image_path.suffix, ".jpg")
        self.assertEqual(kwargs["timeout"], 30)
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])

    def test_critique_storyboard_image_soft_approves_when_hermes_returns_invalid_json(self):
        from OrchestratedVideo import critique_storyboard_image

        def fake_runner(_command, **_kwargs):
            class Result:
                returncode = 0
                stdout = "I think it looks fine, but this is not JSON."
                stderr = ""

            return Result()

        with patch.dict(os.environ, {"VIDEO_ORCHESTRATOR_STRICT_REVIEW": "false", "VIDEO_ORCHESTRATOR_SOFT_REVIEW_FAILURES": "true"}, clear=False):
            critique = critique_storyboard_image(
                b"\xff\xd8" + b"x" * 4096,
                "image/jpeg",
                user_intent="shiny gold coins",
                optimized_prompt="crisp shiny gold coin game UI",
                aspect_ratio="1:1",
                reviewer="hermes",
                runner=fake_runner,
            )

        self.assertTrue(critique.approved)
        self.assertIn("unreadable", critique.reason.lower())

    def test_critique_storyboard_image_blocks_invalid_json_when_strict_review_enabled(self):
        from OrchestratedVideo import critique_storyboard_image

        def fake_runner(_command, **_kwargs):
            class Result:
                returncode = 0
                stdout = "I think it looks fine, but this is not JSON."
                stderr = ""

            return Result()

        with patch.dict(os.environ, {"VIDEO_ORCHESTRATOR_STRICT_REVIEW": "true"}, clear=False):
            critique = critique_storyboard_image(
                b"\xff\xd8" + b"x" * 4096,
                "image/jpeg",
                user_intent="shiny gold coins",
                optimized_prompt="crisp shiny gold coin game UI",
                aspect_ratio="1:1",
                reviewer="hermes",
                runner=fake_runner,
            )

        self.assertFalse(critique.approved)
        self.assertIn("strict review", critique.reason.lower())

    def test_critique_storyboard_image_soft_approves_low_confidence_rejection(self):
        from OrchestratedVideo import critique_storyboard_image

        def fake_runner(_command, **_kwargs):
            class Result:
                returncode = 0
                stdout = json.dumps(
                    {
                        "approved": False,
                        "confidence": 0.42,
                        "reason": "Minor style mismatch but main subject is present.",
                        "improvements": ["Make it shinier"],
                    }
                )
                stderr = ""

            return Result()

        with patch.dict(os.environ, {"VIDEO_ORCHESTRATOR_STRICT_REVIEW": "false", "VIDEO_ORCHESTRATOR_REJECTION_CONFIDENCE_THRESHOLD": "0.85"}, clear=False):
            critique = critique_storyboard_image(
                b"\xff\xd8" + b"x" * 4096,
                "image/jpeg",
                user_intent="shiny gold coins",
                optimized_prompt="crisp shiny gold coin game UI",
                aspect_ratio="1:1",
                reviewer="hermes",
                runner=fake_runner,
            )

        self.assertTrue(critique.approved)
        self.assertIn("low-confidence", critique.reason.lower())

    def test_critique_storyboard_image_blocks_high_confidence_rejection(self):
        from OrchestratedVideo import critique_storyboard_image

        def fake_runner(_command, **_kwargs):
            class Result:
                returncode = 0
                stdout = json.dumps(
                    {
                        "approved": False,
                        "confidence": 0.95,
                        "reason": "Wrong main subject and blank frame.",
                        "improvements": ["Regenerate frame"],
                    }
                )
                stderr = ""

            return Result()

        with patch.dict(os.environ, {"VIDEO_ORCHESTRATOR_STRICT_REVIEW": "false", "VIDEO_ORCHESTRATOR_REJECTION_CONFIDENCE_THRESHOLD": "0.85"}, clear=False):
            critique = critique_storyboard_image(
                b"\xff\xd8" + b"x" * 4096,
                "image/jpeg",
                user_intent="shiny gold coins",
                optimized_prompt="crisp shiny gold coin game UI",
                aspect_ratio="1:1",
                reviewer="hermes",
                runner=fake_runner,
            )

        self.assertFalse(critique.approved)
        self.assertIn("wrong main subject", critique.reason.lower())
    def test_critique_storyboard_image_can_approve_structural_frame_when_hermes_times_out_if_enabled(self):
        from OrchestratedVideo import critique_storyboard_image

        def slow_runner(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 12))

        with patch.dict(os.environ, {"VIDEO_ORCHESTRATOR_ALLOW_REVIEW_TIMEOUT": "true"}, clear=False):
            critique = critique_storyboard_image(
                b"\xff\xd8" + b"x" * 4096,
                "image/jpeg",
                user_intent="neon cyberpunk city in rain",
                optimized_prompt="cinematic neon cyberpunk city first frame",
                aspect_ratio="16:9",
                reviewer="hermes",
                runner=slow_runner,
            )

        self.assertTrue(critique.approved)
        self.assertLess(critique.confidence, 0.5)
        self.assertIn("timed out", critique.reason.lower())

    def test_critique_storyboard_image_uses_env_timeout_for_hermes_cli(self):
        from OrchestratedVideo import critique_storyboard_image

        calls = []

        def fake_runner(command, **kwargs):
            calls.append((command, kwargs))

            class Result:
                returncode = 0
                stdout = json.dumps({"approved": True, "confidence": 0.8, "reason": "ok", "improvements": []})
                stderr = ""

            return Result()

        with patch.dict(os.environ, {"HERMES_REVIEW_TIMEOUT": "24"}, clear=False):
            critique = critique_storyboard_image(
                b"\xff\xd8" + b"x" * 4096,
                "image/jpeg",
                user_intent="neon cyberpunk city in rain",
                optimized_prompt="cinematic neon cyberpunk city first frame",
                aspect_ratio="16:9",
                reviewer="hermes",
                runner=fake_runner,
            )

        self.assertTrue(critique.approved)
        self.assertEqual(calls[0][1]["timeout"], 24)


class VideoOrchestrationRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_get_generate_renders_index_instead_of_html_method_error(self):
        response = self.client.get("/generate")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Generate images and videos from prompts", response.data)

    def test_index_includes_session_media_library(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Session media library", response.data)
        self.assertIn(b"session-library-grid", response.data)
        self.assertIn(b"sessionStorage", response.data)
        self.assertIn(b"addLibraryItem", response.data)

    def test_generated_image_result_includes_masked_edit_controls(self):
        response = self.client.post("/generate", data={"prompt": "a blue house", "aspect_ratio": "1024x1024"})

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Draw mask box", response.data)
        self.assertIn(b"image-edit-workspace", response.data)
        self.assertIn(b"Apply masked edit", response.data)
        self.assertIn(b"applyMaskedImageEdit", response.data)
        self.assertIn(b"clearMaskedEditSelection", response.data)
        self.assertIn(b"imageMaskBox.classList.add('hidden')", response.data)

    @patch("app.build_masked_region_edit")
    def test_image_edit_region_endpoint_returns_patch_payload(self, mock_edit):
        mock_edit.return_value = {
            "patch_url": "https://image.pollinations.ai/p/open-door?width=120&height=80",
            "mask": {"x": 10, "y": 20, "width": 120, "height": 80},
            "image_url": "https://example.com/original.jpg",
            "micro_prompt": "open this door",
        }

        response = self.client.post(
            "/image/edit-region",
            json={
                "image_url": "https://example.com/original.jpg",
                "micro_prompt": "open this door",
                "mask": {"x": 10, "y": 20, "width": 120, "height": 80},
                "context_prompt": "a blue house",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["patch_url"], mock_edit.return_value["patch_url"])
        self.assertEqual(payload["mask"]["width"], 120)
        mock_edit.assert_called_once()

    def test_image_edit_region_endpoint_requires_micro_prompt(self):
        response = self.client.post(
            "/image/edit-region",
            json={"image_url": "https://example.com/original.jpg", "micro_prompt": "", "mask": {"width": 1, "height": 1}},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("micro-prompt", response.get_json()["error"])

    def test_index_includes_video_audio_toggle(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'name="generate_audio"', response.data)
        self.assertIn(b"Add AI-generated audio", response.data)

    def test_index_includes_prompt_rewrite_buttons(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Rewrite image prompt", response.data)
        self.assertIn(b"Rewrite video prompt", response.data)
        self.assertIn(b"rewritePrompt", response.data)

    def test_index_includes_storyboard_grid_controls(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Create 3-frame storyboard", response.data)
        self.assertIn(b"storyboard-grid", response.data)
        self.assertIn(b"regenerateStoryboardFrame", response.data)
        self.assertIn(b"Generate video from storyboard", response.data)
        self.assertNotIn(b"Free Pollinations start frame", response.data)
        self.assertIn(b"resetVideoResult", response.data)
        self.assertIn(b"videoEl.removeAttribute('src')", response.data)
        self.assertIn(b"videoDownload.classList.add('hidden')", response.data)

    @patch("app.materialize_storyboard_frame")
    @patch("app.build_storyboard_frames")
    def test_video_storyboard_endpoint_returns_three_app_served_frames_before_i2v(self, mock_build, mock_materialize):
        from Storyboard import StoryboardFrame

        mock_build.return_value = [
            StoryboardFrame(stage="start", label="Start", prompt="opening frame", url="https://img/start", seed=1),
            StoryboardFrame(stage="middle", label="Middle", prompt="midpoint frame", url="https://img/middle", seed=2),
            StoryboardFrame(stage="end", label="End", prompt="ending frame", url="https://img/end", seed=3),
        ]
        mock_materialize.side_effect = lambda frame: f"https://app.test/storyboard-{frame.stage}.jpg"

        response = self.client.post(
            "/video/storyboard",
            json={"prompt": "monkey with colorful eggs", "aspect_ratio": "16:9"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["workflow"], "pollinations-three-frame-storyboard-before-i2v")
        self.assertEqual([frame["stage"] for frame in payload["frames"]], ["start", "middle", "end"])
        self.assertEqual(payload["frames"][0]["url"], "https://app.test/storyboard-start.jpg")
        self.assertEqual(payload["frames"][0]["source_url"], "https://img/start")
        self.assertEqual(mock_materialize.call_count, 3)
        mock_build.assert_called_once()

    @patch("app.materialize_storyboard_frame")
    @patch("app.regenerate_storyboard_frame")
    def test_video_storyboard_frame_endpoint_regenerates_one_app_served_frame(self, mock_regenerate, mock_materialize):
        from Storyboard import StoryboardFrame

        mock_regenerate.return_value = StoryboardFrame(
            stage="middle",
            label="Middle",
            prompt="monkey jumps higher",
            url="https://img/middle-new",
            seed=9,
        )
        mock_materialize.return_value = "https://app.test/storyboard-middle-new.jpg"

        response = self.client.post(
            "/video/storyboard/frame",
            json={"prompt": "monkey jumps higher", "stage": "middle", "aspect_ratio": "16:9"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["frame"]["stage"], "middle")
        self.assertEqual(payload["frame"]["url"], "https://app.test/storyboard-middle-new.jpg")
        self.assertEqual(payload["frame"]["source_url"], "https://img/middle-new")
        mock_materialize.assert_called_once()
        mock_regenerate.assert_called_once()

    @patch("app.submit_video_job")
    @patch("app.orchestrate_video_first_frame")
    def test_start_video_generation_uses_approved_storyboard_start_frame_without_regenerating(
        self, mock_orchestrate, mock_submit
    ):
        mock_submit.return_value = {"id": "job_story", "polling_url": "https://openrouter.ai/jobs/job_story", "status": "pending"}

        response = self.client.post(
            "/video/start",
            json={
                "prompt": "monkey with colorful eggs",
                "optimized_prompt": "cinematic monkey storyboard",
                "aspect_ratio": "16:9",
                "storyboard_start_frame_url": "https://img/start-approved",
            },
        )

        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertEqual(payload["start_frame_url"], "https://img/start-approved")
        mock_orchestrate.assert_not_called()
        submitted_args, submitted_kwargs = mock_submit.call_args
        self.assertEqual(submitted_args[0], "cinematic monkey storyboard")
        self.assertEqual(submitted_kwargs["first_frame_url"], "https://img/start-approved")

    @patch("app.rewrite_prompt")
    def test_prompt_rewrite_endpoint_returns_rewritten_prompt(self, mock_rewrite):
        mock_rewrite.return_value = "Cinematic macro shot of gold coins, warm studio lighting, crisp details."

        response = self.client.post(
            "/prompt/rewrite",
            json={"prompt": "gold coins", "media_type": "image", "aspect_ratio": "1024x1024"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["original_prompt"], "gold coins")
        self.assertEqual(payload["rewritten_prompt"], mock_rewrite.return_value)
        self.assertEqual(payload["media_type"], "image")
        mock_rewrite.assert_called_once_with("gold coins", media_type="image", aspect_ratio="1024x1024")

    def test_prompt_rewrite_endpoint_requires_prompt(self):
        response = self.client.post("/prompt/rewrite", json={"prompt": "", "media_type": "video"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Please enter a prompt", response.get_json()["error"])

    @patch("app.submit_video_job")
    @patch("app.orchestrate_video_first_frame")
    def test_start_video_generation_does_not_submit_paid_i2v_when_vision_rejects(
        self, mock_orchestrate, mock_submit
    ):
        from OrchestratedVideo import VideoOrchestrationError

        mock_orchestrate.side_effect = VideoOrchestrationError(
            "Vision Agent rejected the free storyboard frame: missing shiny coins"
        )

        response = self.client.post(
            "/video/start",
            json={"prompt": "A vibrant 2D game asset UI design featuring shiny gold coins", "aspect_ratio": "1:1"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("Vision Agent rejected", response.get_json()["error"])
        mock_submit.assert_not_called()

    @patch("app.orchestrate_video_first_frame")
    def test_start_video_generation_returns_json_when_orchestrator_crashes(self, mock_orchestrate):
        mock_orchestrate.side_effect = RuntimeError("Pollinations timeout returned an HTML gateway page")

        response = self.client.post(
            "/video/start",
            json={"prompt": "gold coins", "aspect_ratio": "1:1"},
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.content_type.split(";", 1)[0], "application/json")
        payload = response.get_json()
        self.assertIn("Video generation failed before the paid I2V job was submitted", payload["error"])
        self.assertIn("Pollinations timeout", payload["detail"])

    @patch("app.get_video_status")
    def test_video_status_returns_json_when_status_lookup_crashes(self, mock_status):
        mock_status.side_effect = RuntimeError("OpenRouter returned an HTML error page")

        response = self.client.get("/video/status/job_123")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.content_type.split(";", 1)[0], "application/json")
        payload = response.get_json()
        self.assertIn("Video status lookup failed", payload["error"])
        self.assertIn("OpenRouter returned", payload["detail"])

    @patch("app.get_video_status")
    def test_completed_video_status_uses_same_origin_proxy_for_openrouter_api_content_url(self, mock_status):
        mock_status.return_value = {
            "id": "job_123",
            "status": "completed",
            "unsigned_urls": [],
        }

        response = self.client.get("/video/status/job_123")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["video_url"], "/video/content/job_123")

    @patch("app.get_video_content")
    def test_video_content_proxies_openrouter_bytes_with_video_content_type(self, mock_content):
        mock_content.return_value = (b"fake mp4 bytes", "video/mp4")

        response = self.client.get("/video/content/job_123")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type.split(";", 1)[0], "video/mp4")
        self.assertEqual(response.data, b"fake mp4 bytes")

    @patch("app.submit_video_job")
    @patch("app.orchestrate_video_first_frame")
    def test_start_video_generation_submits_paid_i2v_only_after_approved_frame(
        self, mock_orchestrate, mock_submit
    ):
        from OrchestratedVideo import FirstFrameResult, VisionCritique

        critique = VisionCritique(
            approved=True,
            confidence=0.92,
            reason="Composition matches prompt and has no visible distortion.",
            improvements=[],
            raw_response="{}",
        )
        mock_orchestrate.return_value = FirstFrameResult(
            original_prompt="gold coins",
            optimized_prompt="A crisp 2D game UI asset with shiny gold coins",
            start_frame_url="https://image.pollinations.ai/p/approved-frame?seed=7",
            critique=critique,
            attempts=1,
            width=720,
            height=720,
            seed=7,
        )
        mock_submit.return_value = {"id": "job_123", "polling_url": "https://openrouter.ai/jobs/job_123", "status": "pending"}

        response = self.client.post(
            "/video/start",
            json={"prompt": "gold coins", "aspect_ratio": "1:1"},
        )

        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertEqual(payload["id"], "job_123")
        self.assertEqual(payload["start_frame_url"], "https://image.pollinations.ai/p/approved-frame?seed=7")
        self.assertEqual(payload["optimized_prompt"], "A crisp 2D game UI asset with shiny gold coins")
        self.assertEqual(payload["model"], "alibaba/wan-2.6")
        self.assertEqual(payload["vision_critique"]["approved"], True)
        self.assertEqual(payload["workflow"], "pollinations-vision-gated-start-frame-to-openrouter-i2v")
        mock_submit.assert_called_once()
        submitted_args, submitted_kwargs = mock_submit.call_args
        self.assertEqual(submitted_args[0], "A crisp 2D game UI asset with shiny gold coins")
        self.assertEqual(submitted_kwargs["first_frame_url"], "https://image.pollinations.ai/p/approved-frame?seed=7")
        self.assertFalse(submitted_kwargs["generate_audio"])

    @patch("app.submit_video_job")
    @patch("app.orchestrate_video_first_frame")
    def test_start_video_generation_can_request_generated_audio(
        self, mock_orchestrate, mock_submit
    ):
        from OrchestratedVideo import FirstFrameResult, VisionCritique

        critique = VisionCritique(
            approved=True,
            confidence=0.92,
            reason="Composition matches prompt and has no visible distortion.",
            improvements=[],
            raw_response="{}",
        )
        mock_orchestrate.return_value = FirstFrameResult(
            original_prompt="waves on a beach",
            optimized_prompt="A cinematic first frame of waves on a beach",
            start_frame_url="https://image.pollinations.ai/p/approved-frame?seed=8",
            critique=critique,
            attempts=1,
            width=1280,
            height=720,
            seed=8,
        )
        mock_submit.return_value = {"id": "job_audio", "polling_url": "https://openrouter.ai/jobs/job_audio", "status": "pending"}

        response = self.client.post(
            "/video/start",
            json={"prompt": "waves on a beach", "aspect_ratio": "16:9", "generate_audio": True},
        )

        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertTrue(payload["generate_audio"])
        _submitted_args, submitted_kwargs = mock_submit.call_args
        self.assertTrue(submitted_kwargs["generate_audio"])


if __name__ == "__main__":
    unittest.main()
