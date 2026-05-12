"""Vision-language agent loop: captures camera/lidar input, calls an LLM, and executes tool responses."""
from core.capture_compat import capture_main_view
from core.peripherals import build_peripherals
from dotenv import find_dotenv, load_dotenv
import time
import base64
import logging
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)

_TOOL_FINISH_TASK = "finish_task"
_TOOL_SAVE_CHECKPOINT = "save_checkpoint"
_TOOL_PRECISION_MODE = "go_to_precision_mode"
_TOOL_NORMAL_MODE = "go_to_normal_mode"


class LLMAgent():
    """Robot agent that drives a perception-action loop using a bound LLM and a set of LangChain tools."""

    def __init__(
            self,
            model,
            tools,
            main_camera,
            system_prompt=None,
            camera_fov=90,
            sounddevice_index_or_alias=None,
            servo_controller=None,
            wakeword="robot",
            tts=False,
            history_len=None,
            use_memory=False,
            lidar_usb_port=None,
        ):
        """
        model: name of the model to use
        tools: list of langchain tools
        system_prompt: custom system prompt - optional
        main_camera: provide your robot front camera object.
        camera_fov: field of view (degrees) of your main camera.
        sounddevice_index: provide sounddevice index of your microphone if you want robot to hear.
        wakeword: custom wakeword hearing which robot will set your sentence as a task o do.
        history_len: if you want agent to have messages history cuttof, provide number of newest request-response pairs to keep.
        use_memory: set to True to enable long-term memory (requires sqlite3).
        tts: set to True to enable text-to-speech (robot can speak).
        lidar_usb_port: provide usb port of your lidar if you want robot to support your navigation with lidar.
        """
        system_prompt = system_prompt or ""

        self.tts = tts
        self.task = None
        self.sounddevice_index_or_alias = sounddevice_index_or_alias
        self.navigation_mode = "normal"  # or "precision"

        system_prompt, ph = build_peripherals(
            tools=tools,
            system_prompt=system_prompt,
            use_memory=use_memory,
            sounddevice_index_or_alias=sounddevice_index_or_alias,
            wakeword=wakeword,
            tts=tts,
            lidar_usb_port=lidar_usb_port,
        )
        self.sound_receiver = ph.sound_receiver
        self.task_queue = ph.task_queue
        self.lidar = ph.lidar
        self.lidar_bg = ph.lidar_bg
        self.lidar_scale = ph.lidar_scale

        logger.info("Connecting to LLM: %s", model)
        llm = init_chat_model(model)
        self.llm = llm.bind_tools(tools, parallel_tool_calls=False)
        self._model_name = model
        self.tools = tools
        self.tool_name_to_tool = {tool.name: tool for tool in self.tools}
        self.system_message = SystemMessage(content=system_prompt)
        self.message_history = [self.system_message]
        self.history_len = history_len
        # cameras
        self.main_camera = main_camera
        self.camera_fov = camera_fov
        self.servo_controller = servo_controller

        if self.servo_controller and self.servo_controller.left_arm_head_usb:
            self.servo_controller.reset_head_position()

    def invoke_tool(self, tool_call):
        # convert string to real function
        requested_tool = self.tool_name_to_tool[tool_call["name"]]
        args = tool_call["args"]
        tool_output = requested_tool.invoke(args)
        # if additional output is present
        if isinstance(tool_output, tuple) and len(tool_output) == 2:
            additional_output = HumanMessage(content=tool_output[1])
            tool_output = tool_output[0]
        else:
            additional_output = None
        return ToolMessage(tool_output, tool_call_id=tool_call["id"]), additional_output
    
    def cut_off_context(self, nr_of_loops):
        """
        Trims the message history in the state to keep only the most recent context for the agent.
        """        
        human_indices = [i for i, msg in enumerate(self.message_history) if msg.type == "human"]
        if len(human_indices) >= nr_of_loops:
            start_index = human_indices[-nr_of_loops]
            self.message_history = [self.system_message] + self.message_history[start_index:]

    def check_for_new_task(self):
        """Non-blockingly checks the queue for a new task."""
        if self.task_queue is not None and not self.task_queue.empty():
            self.task = self.task_queue.get()
            
    def lidar_content(self, content):
        try:
            from core.lidar import run_scanner
        except ImportError as e:
            raise RuntimeError(
                "LiDAR support requires cv2, numpy, and rplidar "
                "(pip install opencv-python numpy rplidar-roboticia)"
            ) from e
        lidar_buf, lidar_front_dist = run_scanner(self.lidar, self.lidar_bg, self.lidar_scale, flip_x=True)
        lidar_image_base64 = base64.b64encode(lidar_buf.getvalue()).decode('utf-8')
        
        content.extend([{
            "type": "text", 
            "text": f"""\n\nLiDAR Sensor: Distance from your front edge to nearest obstacle in front: {lidar_front_dist:.1f} cm.
            
Remember that lidar scans only in one horizontal plane (0.5m high), so obstacles above or below that plane may not be detected.
            """
        },
        {"type": "text", "text": "\n\nLiDAR Map (Top-down view, obstacles are marked in red):"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{lidar_image_base64}"}
        }])
        return content

    def fetch_camera_images_base64(self):
        image_bytes = capture_main_view(
            self.main_camera,
            camera_fov=self.camera_fov,
            navigation_mode=self.navigation_mode,
        )
        return [base64.b64encode(image_bytes).decode('utf-8')]
    
    def main_loop_content(self):
        camera_images = self.fetch_camera_images_base64()
        
        content=[
                {"type": "text", "text": "Main camera view:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{camera_images[0]}"}
                },
                {"type": "text", "text": f"\n\nYour task is: '{self.task}'"}
        ]
        
        if self.lidar:
            content = self.lidar_content(content)
        message = HumanMessage(content)
        
        self.message_history.append(message)
        logger.info("Invoking LLM with user Task: \'%s\'", self.task)
        response = self.llm.invoke(self.message_history)
        if response.usage_metadata:
            logger.debug("LLM usage: %s", response.usage_metadata)
        logger.info("Planned tools: %s", [tc["name"] for tc in response.tool_calls])
        
        self.message_history.append(response)
        if self.history_len:
            self.cut_off_context(self.history_len)
        # execute tool
        for tool_call in response.tool_calls:
            logger.info("Executing tool: %s args=%s", tool_call["name"], tool_call["args"])
            tool_response, additional_response = self.invoke_tool(tool_call)
            logger.info("Tool result: %s", tool_response.content)
            self.message_history.append(tool_response)
            if additional_response:
                self.message_history.append(additional_response)
            # Special handling for special tools
            if tool_call["name"] == _TOOL_SAVE_CHECKPOINT:
                checkpoint_info = tool_call["args"].get("checkpont_query")
                self.system_message.content += f"\n[CHECKPOINT DONE] {checkpoint_info}"
            if tool_call["name"] == _TOOL_PRECISION_MODE:
                self.navigation_mode = "precision"
            elif tool_call["name"] == _TOOL_NORMAL_MODE:
                self.navigation_mode = "normal"
            if tool_call["name"] == _TOOL_FINISH_TASK:
                self.task = None
                logger.info("Task finished, going idle.")
                return "Task finished, going idle."

    def go(self):
        try:
            while True:
                if self.task:
                    self.main_loop_content()
                else:
                    # idle mode
                    time.sleep(0.5)
                    
                if self.task_queue is not None:
                    self.check_for_new_task()

        except KeyboardInterrupt:
            logger.info("Interrupted by user, shutting down.")

        finally:
            if self.servo_controller:
                logger.info("Disconnecting servo controller...")
                self.servo_controller.disconnect()