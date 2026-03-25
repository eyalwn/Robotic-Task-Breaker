"""LangChain tool factories for arm control in ManiSkill sim.

Tools wrap a ManiSkillArm adapter; they can be used with any adapter that
exposes move_ee(dx, dy, dz) and set_gripper(open) methods.
"""
from __future__ import annotations
from langchain_core.tools import tool


def create_move_ee(arm):
	"""Return a move_ee LangChain tool bound to *arm*."""

	@tool
	def move_ee(dx: float, dy: float, dz: float) -> str:
		"""Move the end-effector by a delta (in metres) along x, y, z axes.

		Use small steps (0.05–0.2 m). Positive z is up, positive x is forward.
		"""
		arm.move_ee(float(dx), float(dy), float(dz))
		return f"Moved EE by ({dx:.3f}, {dy:.3f}, {dz:.3f}) m."

	return move_ee


def create_set_gripper(arm):
	"""Return a set_gripper LangChain tool bound to *arm*."""

	@tool
	def set_gripper(state: str) -> str:
		"""Open or close the gripper. state must be 'open' or 'close'."""
		if state not in ("open", "close"):
			return f"Error: state must be 'open' or 'close', got {state!r}."
		arm.set_gripper(state)
		past = "opened" if state == "open" else "closed"
		return f"Gripper {past}."

	return set_gripper
