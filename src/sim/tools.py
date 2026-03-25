"""LangChain tool factories for the ManiSkill sim drive backend.

Mirrors the signatures of robots.XLeRobot.tools so that LLMAgent
works unchanged with either backend.
"""
from __future__ import annotations
from langchain_core.tools import tool


def create_move_forward(drive):
	@tool
	def move_forward(distance_meters: float) -> str:
		"""Drives the robot forward (or backward) for a specific distance."""
		drive.move_linear(float(distance_meters))
		direction = "forward" if distance_meters >= 0 else "backward"
		return f"Moved {direction} {abs(distance_meters):.2f} meters."
	return move_forward


def create_turn_left(drive):
	@tool
	def turn_left(angle_degrees: float) -> str:
		"""Turns the robot left by angle in degrees."""
		drive.turn_in_place(abs(float(angle_degrees)))
		return f"Turned left by {abs(angle_degrees):.1f} degrees."
	return turn_left


def create_turn_right(drive):
	@tool
	def turn_right(angle_degrees: float) -> str:
		"""Turns the robot right by angle in degrees."""
		drive.turn_in_place(-abs(float(angle_degrees)))
		return f"Turned right by {abs(angle_degrees):.1f} degrees."
	return turn_right
