"""Linux/ALSA-specific audio device resolution for XLeRobot hardware."""

import os
import re


def resolve_alsa_device_index(pyaudio_instance, alias: str) -> int:
    """Resolve a udev symlink alias to a PyAudio input device index via ALSA.

    Resolves /dev/<alias> → real path → ALSA card number → PyAudio index.
    Only meaningful on Linux with udev-managed audio devices.
    """
    symlink_path = f"/dev/{alias}"
    if not os.path.exists(symlink_path):
        raise FileNotFoundError(f"Symlink {symlink_path} not found.")
    real_path = os.path.realpath(symlink_path)
    match = re.search(r"controlC(\d+)", real_path)
    if not match:
        raise ValueError(f"Could not parse ALSA card ID from {real_path}")
    alsa_card_index = int(match.group(1))
    info_count = pyaudio_instance.get_device_count()
    for i in range(info_count):
        info = pyaudio_instance.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            if f"(hw:{alsa_card_index}," in info['name']:
                return i
    raise RuntimeError(
        f"ALSA Card {alsa_card_index} found, but no matching PyAudio device detected."
    )
