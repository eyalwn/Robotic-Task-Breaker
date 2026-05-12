"""
Earth Rover basic movement example demonstrating essential navigation capabilities.
"""
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from robots.EarthRover.earth_rover_llm_agent import EarthRoverAgent
from robots.EarthRover.tools import \
    move_forward, \
    move_backward, \
    move_forward_max_speed, \
    turn_right_forward_rotation, \
    turn_left_forward_rotation, \
    turn_right_backward_rotation, \
    turn_left_backward_rotation

# init Earth Rover agent
agent = EarthRoverAgent(
    model="google_genai:gemini-3-flash-preview",
    tools=[
        move_forward,
        move_backward,
        move_forward_max_speed,
        turn_right_forward_rotation,
        turn_left_forward_rotation,
        turn_right_backward_rotation,
        turn_left_backward_rotation,
    ],
    history_len=5,
    camera_fov=120,
    use_location_visualizer=True,
)

# Put here your waypoint coordinates
agent.waypoints = [
    (52.22018705359169, 21.121002790561878),
    (52.21051416289311, 21.12417188863794),
    (52.20248417804582, 21.11926649920412),
    ]

agent.go()
