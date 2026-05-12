"""XLeRobot-specific agent entry point; delegates all behaviour to the base LLMAgent."""
from pathlib import Path

from core.llm_agent import LLMAgent


class XLeRobotAgent(LLMAgent):
	"""XLeRobot specific LLM agent that inherits from base LLMAgent."""
	def __init__(
		self,
		model,
		tools,
		system_prompt=None,
		camera_fov=90,
		history_len=None,
		use_memory=False,
		main_camera=None,
		sounddevice_index_or_alias=None,
		servo_controller=None,
		wakeword=None,
		tts=False,
		lidar_usb_port=None,
	):
		if system_prompt is None:
			prompt_path = Path(__file__).parent / "xlerobot.prompt"
			with open(prompt_path, "r") as f:
				system_prompt = f.read()

		super().__init__(
			model=model,
			tools=tools,
			main_camera=main_camera,
			system_prompt=system_prompt,
			camera_fov=camera_fov,
			sounddevice_index_or_alias=sounddevice_index_or_alias,
			servo_controller=servo_controller,
			wakeword=wakeword,
			tts=tts,
			history_len=history_len,
			use_memory=use_memory,
			lidar_usb_port=lidar_usb_port
		)

	# No new features or methods; inherits all behavior from LLMAgent
