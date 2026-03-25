"""Text-to-speech synthesis using the Piper TTS engine, played back via pygame.

Requires pygame + piper-tts (see requirements-optional.txt).
"""
import os
import sys
import subprocess
import urllib.request
import time


_DATA_DIR = os.environ.get(
    "ROBOCREW_TTS_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "Longor"),
)
MODEL_PATH = os.path.join(_DATA_DIR, "en_amy.onnx")
CONFIG_PATH = os.path.join(_DATA_DIR, "en_amy.onnx.json")
OUTPUT_WAV = os.path.join(_DATA_DIR, "speech.wav")


def setup_voice():
    os.makedirs(_DATA_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100:
        print(f"Downloading TTS model to {MODEL_PATH}...")
        model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx"
        urllib.request.urlretrieve(model_url, MODEL_PATH)
    if not os.path.exists(CONFIG_PATH) or os.path.getsize(CONFIG_PATH) < 100:
        print(f"Downloading TTS config to {CONFIG_PATH}...")
        config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx.json"
        urllib.request.urlretrieve(config_url, CONFIG_PATH)


def speak_and_play(text):
    """Synthesize text to speech and play it immediately."""
    try:
        import pygame as _pygame
    except ImportError:
        raise RuntimeError(
            "TTS playback requires pygame. pip install -r requirements-optional.txt (tts lines) or install pygame manually."
        )
    setup_voice()
    subprocess.run(
        [sys.executable, "-m", "piper", "--model", MODEL_PATH, "--config", CONFIG_PATH, "--output_file", OUTPUT_WAV],
        input=text.encode("utf-8"),
        check=True,
        capture_output=True,
    )
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
    _pygame.mixer.init(frequency=22050)
    _pygame.mixer.music.load(OUTPUT_WAV)
    _pygame.mixer.music.play()
    while _pygame.mixer.music.get_busy():
        time.sleep(0.1)
    _pygame.mixer.quit()

