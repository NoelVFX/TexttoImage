import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from TexttoImage import build_pollinations_url


class PollinationsUrlTests(unittest.TestCase):
    def test_build_pollinations_url_includes_seed_when_provided(self):
        url = build_pollinations_url("gold coins", width=720, height=720, seed=12345)

        self.assertIn("seed=12345", url)
        self.assertIn("width=720", url)
        self.assertIn("height=720", url)


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
        self.assertEqual(kwargs["timeout"], 120)
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])

    def test_critique_storyboard_image_blocks_when_hermes_returns_invalid_json(self):
        from OrchestratedVideo import critique_storyboard_image

        def fake_runner(_command, **_kwargs):
            class Result:
                returncode = 0
                stdout = "I think it looks fine, but this is not JSON."
                stderr = ""

            return Result()

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
        self.assertIn("unreadable", critique.reason.lower())


class VideoOrchestrationRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_get_generate_renders_index_instead_of_html_method_error(self):
        response = self.client.get("/generate")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Generate images and videos from prompts", response.data)

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
        self.assertEqual(payload["vision_critique"]["approved"], True)
        self.assertEqual(payload["workflow"], "pollinations-vision-gated-start-frame-to-openrouter-i2v")
        mock_submit.assert_called_once()
        submitted_args, submitted_kwargs = mock_submit.call_args
        self.assertEqual(submitted_args[0], "A crisp 2D game UI asset with shiny gold coins")
        self.assertEqual(submitted_kwargs["first_frame_url"], "https://image.pollinations.ai/p/approved-frame?seed=7")


if __name__ == "__main__":
    unittest.main()
