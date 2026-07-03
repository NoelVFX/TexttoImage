import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import app
from TexttoImage import build_pollinations_url


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


if __name__ == "__main__":
    unittest.main()
