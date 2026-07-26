"""XLeRobot arm VLA demo (grab notebook, give to human).

Run VLA server first:
  python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080

Device paths via ROBOCREW_CAMERA, ROBOCREW_CAMERA_RIGHT, ROBOCREW_USB_*.
"""
import os
from pathlib import Path

from core.camera import RobotCamera
from core.tools import save_checkpoint
from robots.XLeRobot.servo_controls import ServoController
from robots.XLeRobot.tools import (
    create_go_to_normal_mode,
    create_go_to_precision_mode,
    create_look_around,
    create_move_backward,
    create_move_forward,
    create_strafe_left,
    create_strafe_right,
    create_turn_left,
    create_turn_right,
    create_vla_single_arm_manipulation,
)
from robots.XLeRobot.xlerobot_llm_agent import XLeRobotAgent
from utils import configure_logging, load_env

load_env()
configure_logging()

prompt_path = Path(__file__).parent.parent / "src/robots/XLeRobot/xlerobot.prompt"
with open(prompt_path, encoding="utf-8") as f:
    system_prompt = f.read()

main_camera = RobotCamera(os.environ.get("ROBOCREW_CAMERA", "/dev/camera_center"))
right_arm_wheel_usb = os.environ.get("ROBOCREW_USB_RIGHT", "/dev/arm_right")
left_arm_head_usb = os.environ.get("ROBOCREW_USB_LEFT", "/dev/arm_left")
servo_controller = ServoController(right_arm_wheel_usb, left_arm_head_usb)

move_forward = create_move_forward(servo_controller)
move_backward = create_move_backward(servo_controller)
turn_left = create_turn_left(servo_controller)
turn_right = create_turn_right(servo_controller)
strafe_left = create_strafe_left(servo_controller)
strafe_right = create_strafe_right(servo_controller)
look_around = create_look_around(servo_controller, main_camera)
go_to_precision_mode = create_go_to_precision_mode(servo_controller)
go_to_normal_mode = create_go_to_normal_mode(servo_controller)

camera_config = {
    "main": {"index_or_path": os.environ.get("ROBOCREW_CAMERA", "/dev/camera_center")},
    "right_arm": {
        "index_or_path": os.environ.get("ROBOCREW_CAMERA_RIGHT", "/dev/camera_right"),
    },
}

pick_up_notebook = create_vla_single_arm_manipulation(
    tool_name="Grab_a_notebook",
    tool_description=(
        "Manipulation tool to grab a notebook from the table and put it to your basket. "
        "Use the tool only when you are very very close to table with a notebook, "
        "and look straingt on it."
    ),
    task_prompt="Grab a notebook.",
    server_address="0.0.0.0:8080",
    policy_name="Grigorij/act_right-arm-grab-notebook-2",
    policy_type="act",
    arm_port=right_arm_wheel_usb,
    servo_controller=servo_controller,
    camera_config=camera_config,
    main_camera_object=main_camera,
    policy_device="cpu",
)

give_notebook = create_vla_single_arm_manipulation(
    tool_name="Give_a_notebook_to_a_human",
    tool_description=(
        "Manipulation tool to take a notebook from your basket and give it to human. "
        "Use the tool only when you are close to the human (base of human is below green line), "
        "and look straingt on him."
    ),
    task_prompt="Grab a notebook and give it to a human.",
    server_address="0.0.0.0:8080",
    policy_name="Grigorij/act_right_arm_give_notebook",
    policy_type="act",
    arm_port=right_arm_wheel_usb,
    servo_controller=servo_controller,
    camera_config=camera_config,
    main_camera_object=main_camera,
    policy_device="cpu",
    execution_time=45,
)

agent = XLeRobotAgent(
    model="google_genai:gemini-3-flash-preview",
    system_prompt=system_prompt,
    tools=[
        move_forward,
        move_backward,
        strafe_left,
        strafe_right,
        turn_left,
        turn_right,
        look_around,
        pick_up_notebook,
        give_notebook,
        go_to_precision_mode,
        go_to_normal_mode,
        save_checkpoint,
    ],
    history_len=8,
    main_camera=main_camera,
    camera_fov=90,
    servo_controller=servo_controller,
)

agent.task = (
    "Approach blue notebook, grab it from the table and give it to human. "
    "Do not approach human until you grabbed a notebook."
)
agent.go()
