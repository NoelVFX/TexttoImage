from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from OrchestratedVideo import optimize_prompt_for_storyboard, stable_seed
from TexttoImage import DEFAULT_MODEL, build_pollinations_url


STORYBOARD_STAGES = (
    ("start", "Start", "Opening frame / beginning of the scene"),
    ("middle", "Middle", "Midpoint frame / main action beat"),
    ("end", "End", "Ending frame / resolved final beat"),
)
STAGE_LABELS = {stage: label for stage, label, _description in STORYBOARD_STAGES}
STAGE_DESCRIPTIONS = {stage: description for stage, _label, description in STORYBOARD_STAGES}


@dataclass(frozen=True)
class StoryboardFrame:
    stage: str
    label: str
    prompt: str
    url: str
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def normalise_stage(stage: str | None) -> str:
    selected = (stage or "start").strip().lower()
    return selected if selected in STAGE_LABELS else "start"


def build_stage_prompt(optimized_prompt: str, *, stage: str, user_direction: str | None = None) -> str:
    selected_stage = normalise_stage(stage)
    direction = (user_direction or "").strip()
    stage_detail = STAGE_DESCRIPTIONS[selected_stage]
    if direction:
        return (
            f"{stage_detail}: {direction}. Based on the video concept: {optimized_prompt}. "
            "Consistent character identity, composition, cinematic lighting, coherent visual continuity."
        )
    return (
        f"{stage_detail}: {optimized_prompt}. Consistent character identity, composition, cinematic lighting, "
        "clear visual continuity, high-quality static storyboard frame."
    )


def build_storyboard_frame(
    optimized_prompt: str,
    *,
    stage: str,
    aspect_ratio: str,
    width: int,
    height: int,
    model_choice: str = DEFAULT_MODEL,
    user_direction: str | None = None,
) -> StoryboardFrame:
    selected_stage = normalise_stage(stage)
    frame_prompt = build_stage_prompt(optimized_prompt, stage=selected_stage, user_direction=user_direction)
    seed = stable_seed(f"storyboard|{frame_prompt}|{aspect_ratio}|{selected_stage}")
    url = build_pollinations_url(
        frame_prompt,
        model_choice=model_choice,
        width=width,
        height=height,
        seed=seed,
    )
    return StoryboardFrame(
        stage=selected_stage,
        label=STAGE_LABELS[selected_stage],
        prompt=frame_prompt,
        url=url,
        seed=seed,
    )


def build_storyboard_frames(
    user_prompt: str,
    *,
    aspect_ratio: str,
    width: int,
    height: int,
    model_choice: str = DEFAULT_MODEL,
    prompt_optimizer: Callable[..., str] = optimize_prompt_for_storyboard,
) -> list[StoryboardFrame]:
    if not user_prompt or not user_prompt.strip():
        raise ValueError("Prompt is required.")
    optimized_prompt = prompt_optimizer(user_prompt.strip(), aspect_ratio=aspect_ratio)
    return [
        build_storyboard_frame(
            optimized_prompt,
            stage=stage,
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            model_choice=model_choice,
        )
        for stage, _label, _description in STORYBOARD_STAGES
    ]


def regenerate_storyboard_frame(
    frame_prompt: str,
    *,
    stage: str,
    aspect_ratio: str,
    width: int,
    height: int,
    model_choice: str = DEFAULT_MODEL,
    base_prompt: str | None = None,
) -> StoryboardFrame:
    if not frame_prompt or not frame_prompt.strip():
        raise ValueError("Prompt is required.")
    base = (base_prompt or frame_prompt).strip()
    return build_storyboard_frame(
        base,
        stage=stage,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
        model_choice=model_choice,
        user_direction=frame_prompt.strip(),
    )
