"""anima_voice 内置麦克风 ASR 模块。"""

from .audio_source import MicrophoneAudioSource
from .recognizer import StreamingRecognizer
from .runtime import RedirectTarget, VoiceMicrophoneAdapter
from .text_quality import (
    TextQualityDecision,
    is_likely_normal_utterance,
    is_obviously_invalid_asr_text,
)

__all__ = [
    "MicrophoneAudioSource",
    "RedirectTarget",
    "StreamingRecognizer",
    "TextQualityDecision",
    "VoiceMicrophoneAdapter",
    "is_likely_normal_utterance",
    "is_obviously_invalid_asr_text",
]