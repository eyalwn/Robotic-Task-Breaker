"""Reusable sim run-loop: ties LLMAgent, success checker, and frame saver."""
from __future__ import annotations

import logging
import time
from core.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


def run_sim_agent(
    agent: LLMAgent,
    camera,
    success_checker,
    frame_saver,
    max_loops: int = 100,
    max_seconds: float = 300,
) -> dict:
    """Run the agent loop with bounded iterations and between-turn time.

    Note:
        max_seconds is checked only before each outer LLM turn. A single tool call
        inside `agent.main_loop_content()` may run longer than max_seconds.

    Returns a summary dict with keys:
        loops_executed, elapsed_seconds, exit_reason, last_frame_path
    """
    start = time.monotonic()
    loops = 0
    exit_reason = "max_loops"
    last_frame_path: str | None = None
    if max_loops == 1:
        logger.info("Single-turn mode enabled (max_loops=1); multi-turn retries are opt-in.")

    try:
        for loops in range(1, max_loops + 1):
            elapsed = time.monotonic() - start
            if elapsed >= max_seconds:
                exit_reason = "max_seconds"
                loops -= 1  # this iteration didn't actually run
                break

            jpeg_bytes = camera.capture_image()
            agent.main_loop_content()

            saved = frame_saver.save(jpeg_bytes, loops)
            if saved is not None:
                last_frame_path = saved

            if success_checker.check():
                proof = frame_saver.save_success(jpeg_bytes)
                last_frame_path = proof
                exit_reason = "success"
                logger.info("Success detected at loop %d", loops)
                break

            if agent.task is None:
                exit_reason = "finish_task"
                break
        else:
            exit_reason = "max_loops"

    except Exception:
        exit_reason = "error"
        logger.exception("Runner loop error")

    elapsed = time.monotonic() - start
    logger.info(
        "Run finished — turns=%d elapsed=%.1fs reason=%s last_frame=%s",
        loops,
        elapsed,
        exit_reason,
        last_frame_path,
    )
    return {
        "loops_executed": loops,
        "elapsed_seconds": round(elapsed, 3),
        "exit_reason": exit_reason,
        "last_frame_path": last_frame_path,
    }
