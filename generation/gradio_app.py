import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from generation.generate_post import OPENAI_MODEL_OPTIONS, generate_post
from generation.brand_checker import check_brand_consistency
from generation.refiner import refine_post

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COHERE_MODEL_OPTIONS = [
    "command-a-03-2025",
    "command-r7b-12-2024",
    "command-r-plus-08-2024",
    "command-r-08-2024",
]

# Load env vars from common local files so OPENAI_API_KEY is available in UI runs.
load_dotenv(PROJECT_ROOT / ".ENV")
load_dotenv(PROJECT_ROOT / ".env")


def _build_config(
    model: str,
    custom_model: Optional[str],
    cohere_model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    timeout: float,
) -> Dict[str, Any]:
    selected_model = (custom_model or model or "").strip()
    config: Dict[str, Any] = {
        "model": selected_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "retries": retries,
        "timeout": timeout,
        "cohere_model": cohere_model,
    }
    return config


def run_generation(
    topic: str,
    post_type: str,
    business_objective: str,
    model: str,
    custom_model: Optional[str],
    cohere_model: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    timeout: float,
) -> Tuple[str, str]:
    steps: List[str] = []
    topic = (topic or "").strip()
    business_objective = (business_objective or "").strip()

    if not topic:
        return "Validation failed: Topic is required.", ""
    if not business_objective:
        return "Validation failed: Business objective is required.", ""
    if not (custom_model or model):
        return "Validation failed: Model selection is required.", ""

    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_key:
        return "Validation failed: OPENAI_API_KEY not found in environment.", ""
    has_cohere_key = bool(os.getenv("COHERE_API_KEY"))
    if not has_cohere_key:
        return "Validation failed: COHERE_API_KEY not found in environment.", ""

    try:
        steps.append("1. Inputs validated")
        config = _build_config(
            model=model,
            custom_model=custom_model,
            cohere_model=cohere_model,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            timeout=timeout,
        )
        steps.append("2. Generating candidate drafts with OpenAI")
        draft_post, metadata = generate_post(
            topic=topic,
            post_type=post_type,
            business_objective=business_objective,
            config=config,
        )
        selected_angle = (
            metadata.get("candidate_generation", {}).get("selected_angle")
            if isinstance(metadata, dict)
            else None
        )
        if selected_angle:
            steps.append(f"3. Cohere selected best draft angle: {selected_angle}")
        else:
            steps.append("3. Cohere evaluated and selected best draft")

        steps.append("4. Running first refinement pass")
        refined_post, refinement_metadata = refine_post(
            draft_post=draft_post,
            topic=topic,
            post_type=post_type,
            business_objective=business_objective,
            config=config,
        )
        if not refined_post:
            refined_post = draft_post

        steps.append("5. Running initial brand consistency check")
        initial_brand_result, initial_brand_metadata = check_brand_consistency(
            post=refined_post,
            config=config,
        )

        steps.append("6. Refining again with brand checker feedback")
        feedback_refined_post, feedback_refinement_metadata = refine_post(
            draft_post=refined_post,
            topic=topic,
            post_type=post_type,
            business_objective=business_objective,
            config=config,
            brand_feedback_summary=initial_brand_result.get("feedback_summary", ""),
            brand_score=int(initial_brand_result.get("score", 0)),
        )
        final_post = feedback_refined_post or refined_post

        steps.append("7. Running final brand consistency check")
        final_brand_result, final_brand_metadata = check_brand_consistency(
            post=final_post,
            config=config,
        )

        metadata["refinement"] = {
            "initial": refinement_metadata,
            "feedback_driven": feedback_refinement_metadata,
        }
        metadata["brand_check"] = {
            "initial": {
                "result": initial_brand_result,
                "metadata": initial_brand_metadata,
            },
            "final": {
                "result": final_brand_result,
                "metadata": final_brand_metadata,
            },
        }
        final_score = final_brand_result.get("score", 0)
        steps.append(f"8. Final post ready (brand score: {final_score}/100)")

        return "\n".join(steps), final_post
    except Exception as exc:
        steps.append(f"Failed: {exc}")
        return "\n".join(steps), ""


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Personal Brand AI Content Generator") as demo:
        gr.Markdown("## Personal Brand AI Content Generator")
        gr.Markdown("Generate SME-focused LinkedIn posts and inspect generation metadata.")

        with gr.Row():
            with gr.Column(scale=2):
                topic = gr.Textbox(
                    label="Topic",
                    placeholder="e.g., Why SME teams fail at AI adoption after pilot success",
                )
                post_type = gr.Dropdown(
                    label="Post Type",
                    choices=["thought_leadership", "educational", "trend_commentary"],
                    value="thought_leadership",
                )
                business_objective = gr.Textbox(
                    label="Business Objective",
                    placeholder="e.g., Build authority with SME founders and generate inbound leads",
                )

                with gr.Accordion("Generation Settings", open=False):
                    model = gr.Dropdown(
                        label="Model",
                        choices=OPENAI_MODEL_OPTIONS,
                        value=OPENAI_MODEL_OPTIONS[0],
                    )
                    custom_model = gr.Textbox(
                        label="Custom Model (optional override)",
                        placeholder="e.g., gpt-5-mini",
                    )
                    cohere_model = gr.Dropdown(
                        label="Cohere Evaluator Model",
                        choices=COHERE_MODEL_OPTIONS,
                        value=COHERE_MODEL_OPTIONS[0],
                    )
                    temperature = gr.Slider(
                        label="Temperature", minimum=0.0, maximum=1.5, step=0.1, value=0.7
                    )
                    max_tokens = gr.Slider(
                        label="Max Tokens", minimum=100, maximum=2000, step=50, value=500
                    )
                    retries = gr.Slider(label="Retries", minimum=1, maximum=6, step=1, value=3)
                    timeout = gr.Slider(
                        label="Timeout (seconds)", minimum=10, maximum=180, step=5, value=60
                    )
                generate_btn = gr.Button("Generate Post", variant="primary")

            with gr.Column(scale=3):
                steps_output = gr.Textbox(
                    label="Generation Steps",
                    lines=10,
                )
                final_post_output = gr.Textbox(
                    label="Final Post",
                    lines=18,
                )

        generate_btn.click(
            fn=run_generation,
            inputs=[
                topic,
                post_type,
                business_objective,
                model,
                custom_model,
                cohere_model,
                temperature,
                max_tokens,
                retries,
                timeout,
            ],
            outputs=[
                steps_output,
                final_post_output,
            ],
        )

    return demo


def main() -> None:
    demo = build_interface()
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    try:
        demo.launch(server_name="127.0.0.1", server_port=preferred_port)
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            fallback_port = sock.getsockname()[1]
        demo.launch(server_name="127.0.0.1", server_port=fallback_port)


if __name__ == "__main__":
    main()
