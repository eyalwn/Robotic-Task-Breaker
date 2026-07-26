"""XLeRobot voice demo: wakeword + mic + basic movement.

Device paths via ROBOCREW_CAMERA, ROBOCREW_USB_RIGHT.
"""
import os

from core.camera import RobotCamera
from core.tools import finish_task
from robots.XLeRobot.servo_controls import ServoController
from robots.XLeRobot.tools import create_move_forward, create_turn_left, create_turn_right
from robots.XLeRobot.xlerobot_llm_agent import XLeRobotAgent
from utils import configure_logging, load_env

load_env()
configure_logging()

main_camera = RobotCamera(os.environ.get("ROBOCREW_CAMERA", "/dev/camera_center"))
servo_controller = ServoController(
    right_arm_wheel_usb=os.environ.get("ROBOCREW_USB_RIGHT", "/dev/arm_right"),
)

move_forward = create_move_forward(servo_controller)
turn_left = create_turn_left(servo_controller)
turn_right = create_turn_right(servo_controller)

agent = XLeRobotAgent(
    model="google_genai:gemini-3-flash-preview",
    tools=[move_forward, turn_left, turn_right, finish_task],
    main_camera=main_camera,
    servo_controller=servo_controller,
    sounddevice_index_or_alias="mic_main",
    wakeword="Bob",
)

agent.go()
print("Listening for your instructions...")
