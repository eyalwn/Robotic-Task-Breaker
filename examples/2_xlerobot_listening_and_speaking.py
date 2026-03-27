"""
The simplest example of agent that can listen and speak, aside from driving XLeRobot.

Device paths are read from env vars (set these before running on HW):
  ROBOCREW_CAMERA     camera device  (default: /dev/camera_center)
  ROBOCREW_USB_RIGHT  right-arm/wheel USB port (default: /dev/arm_right)
"""
import os
import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from core.camera import RobotCamera
from robots.XLeRobot.xlerobot_LLM_agent import XLeRobotAgent
from robots.XLeRobot.tools import create_move_forward, create_turn_right, create_turn_left
from robots.XLeRobot.servo_controls import ServoControler
from core.tools import finish_task

# set up main camera
main_camera = RobotCamera(os.environ.get("ROBOCREW_CAMERA", "/dev/camera_center"))

# set up servo controler
right_arm_wheel_usb = os.environ.get("ROBOCREW_USB_RIGHT", "/dev/arm_right")
servo_controler = ServoControler(right_arm_wheel_usb=right_arm_wheel_usb)

# set up tools
move_forward = create_move_forward(servo_controler)
turn_left = create_turn_left(servo_controler)
turn_right = create_turn_right(servo_controler)

# init agent
agent = XLeRobotAgent(
    model="google_genai:gemini-3-flash-preview",
    tools=[
        move_forward,
        turn_left,
        turn_right,
        finish_task,
    ],
    
    main_camera=main_camera,
    servo_controler=servo_controler,
    sounddevice_index_or_alias="mic_main",    # provide your microphone device index.
    wakeword="Bob",         # set custom wakeword (default is "robot").
    #tts=True,               # uncomment for text-to-speech, to make robot speak (work in progress).
)

agent.go()

print("Listening for your instructions...")