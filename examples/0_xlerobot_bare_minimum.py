"""
The simplest example of agent that can drive XLeRobot.
"""
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.camera import RobotCamera
from robots.XLeRobot.xlerobot_llm_agent import XLeRobotAgent
from robots.XLeRobot.tools import create_move_forward, create_turn_right, create_turn_left
from robots.XLeRobot.servo_controls import ServoController

# set up main camera
main_camera = RobotCamera("/dev/camera_center") # camera usb port Eg: /dev/video0

#set up servo controller
right_arm_wheel_usb = "/dev/arm_right"    # provide your right arm usb port. Eg: /dev/ttyACM1
servo_controller = ServoController(right_arm_wheel_usb=right_arm_wheel_usb)

#set up tools
move_forward = create_move_forward(servo_controller)
turn_left = create_turn_left(servo_controller)
turn_right = create_turn_right(servo_controller)

# init agent
agent = XLeRobotAgent(
    model="google_genai:gemini-3-flash-preview",
    tools=[
        move_forward,
        turn_left,
        turn_right,
    ],
    main_camera=main_camera,
    servo_controller=servo_controller,
)

agent.task = "Approach a human."

agent.go()