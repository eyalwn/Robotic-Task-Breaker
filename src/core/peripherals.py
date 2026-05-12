"""Optional agent peripherals (sound, LiDAR, TTS, memory) — keeps LLMAgent.__init__ thin."""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Any

from core.tools import create_say, remember_thing, recall_thing

logger = logging.getLogger(__name__)


@dataclass
class PeripheralHandles:
    """Runtime handles for optional hardware / IO."""

    sound_receiver: Any = None
    task_queue: queue.Queue | None = None
    lidar: Any = None
    lidar_bg: Any = None
    lidar_scale: Any = None


def apply_memory_tools(
    tools: list,
    system_prompt: str,
    *,
    use_memory: bool,
) -> str:
    if not use_memory:
        return system_prompt
    tools.append(remember_thing)
    tools.append(recall_thing)
    return system_prompt + (
        " You have a memory. When you find important things (like a specific room, object, or person) "
        "or complete a navigation step, use the `remember_thing` tool to save it for later. "
        "Do not wait for the user to tell you to remember. Be proactive."
    )


def apply_tts_tool(
    tools: list,
    system_prompt: str,
    *,
    tts: bool,
    sound_receiver: Any,
) -> str:
    if not tts:
        return system_prompt
    tools.append(create_say(sound_receiver))
    return system_prompt + (
        " You can speak to the user using the `say` tool. "
        "Use it to communicate important updates, greet users, or answer their questions verbally."
    )


def init_sound_receiver(
    sounddevice_index_or_alias: Any,
    wakeword: str,
) -> tuple[Any | None, queue.Queue | None]:
    if sounddevice_index_or_alias is None:
        return None, None
    from core.sound_receiver import SoundReceiver

    q: queue.Queue = queue.Queue()
    receiver = SoundReceiver(sounddevice_index_or_alias, q, wakeword)
    return receiver, q


def init_lidar(lidar_usb_port: str | None) -> tuple[Any, Any, Any]:
    if not lidar_usb_port:
        return None, None, None
    try:
        from core.lidar import init_lidar
    except ImportError as e:
        raise RuntimeError(
            "LiDAR support requires cv2, numpy, and rplidar "
            "(pip install opencv-python numpy rplidar-roboticia)"
        ) from e
    return init_lidar(lidar_usb_port)


def build_peripherals(
    *,
    tools: list,
    system_prompt: str,
    use_memory: bool,
    sounddevice_index_or_alias: Any,
    wakeword: str,
    tts: bool,
    lidar_usb_port: str | None,
) -> tuple[str, PeripheralHandles]:
    """Apply memory + TTS prompt/tool changes; construct sound + lidar handles."""
    sp = apply_memory_tools(tools, system_prompt, use_memory=use_memory)

    sound_receiver, task_queue = init_sound_receiver(
        sounddevice_index_or_alias,
        wakeword,
    )
    sp = apply_tts_tool(tools, sp, tts=tts, sound_receiver=sound_receiver)

    handles = PeripheralHandles(
        sound_receiver=sound_receiver,
        task_queue=task_queue,
    )
    if lidar_usb_port:
        lidar, lidar_bg, lidar_scale = init_lidar(lidar_usb_port)
        handles.lidar = lidar
        handles.lidar_bg = lidar_bg
        handles.lidar_scale = lidar_scale

    return sp, handles
