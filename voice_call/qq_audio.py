"""QQ 16 kHz PCM 的静音分句和 WAV 封装。"""

from __future__ import annotations

import io
import wave
from collections import deque

import numpy as np


class PCMUtteranceSegmenter:
    """用持续能量和静音端点切分语句，保留前滚并限制单段时长。"""

    def __init__(
        self,
        threshold: float,
        silence_ms: int,
        max_audio_ms: int,
        preroll_ms: int,
    ) -> None:
        """使用现有 ASR 的能量阈值和时长设置初始化。"""
        if threshold <= 0 or silence_ms <= 0 or max_audio_ms <= 0 or preroll_ms < 0:
            raise ValueError("音频分句阈值和时长无效")
        self._threshold = threshold
        self._silence_frames = max(1, (silence_ms + 19) // 20)
        self._max_frames = max(1, max_audio_ms // 20)
        self._preroll: deque[bytes] = deque(maxlen=preroll_ms // 20)
        self._pending = bytearray()
        self._utterance: list[bytes] = []
        self._silence_count = 0
        self._above_threshold_frames = 0
        self._voice_frames = 0
        self._voice_run = 0
        self._confirmed = False
        self._processed_samples = 0
        self.completed_stats: list[dict[str, str | int]] = []
        self.discarded_stats: list[dict[str, str | int]] = []

    def _complete(self, reason: str) -> bytes:
        """闭合候选语句，只提交含至少 120ms 连续活动的音频。"""
        pcm = b"".join(self._utterance)
        stats: dict[str, str | int] = {
            "end_reason": reason,
            "duration_ms": len(pcm) // 32,
            "above_threshold_ms": self._above_threshold_frames * 20,
            "voice_ms": self._voice_frames * 20,
            "trailing_silence_ms": self._silence_count * 20,
            "start_sample": self._processed_samples - len(pcm) // 2,
            "end_sample": self._processed_samples,
        }
        accepted = self._confirmed
        if accepted:
            self.completed_stats.append(stats)
        else:
            stats["boundary_reason"] = reason
            stats["end_reason"] = "short_activity"
            self.discarded_stats.append(stats)
        self._utterance.clear()
        self._silence_count = 0
        self._above_threshold_frames = 0
        self._voice_frames = 0
        self._voice_run = 0
        self._confirmed = accepted and reason == "max_duration"
        return pcm if accepted else b""

    def feed(self, pcm: bytes) -> list[bytes]:
        """消费任意大小的 S16LE 块，返回已闭合语句。"""
        self.completed_stats = []
        self.discarded_stats = []
        self._pending.extend(pcm)
        completed: list[bytes] = []
        while len(self._pending) >= 640:
            frame = bytes(self._pending[:640])
            del self._pending[:640]
            self._processed_samples += 320
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples * samples)))
            voiced = rms >= self._threshold
            active = rms >= self._threshold * 0.5
            if not self._utterance and not voiced and not (self._confirmed and active):
                self._confirmed = False
                self._preroll.append(frame)
                continue
            if not self._utterance:
                self._utterance.extend(self._preroll)
                self._preroll.clear()
            self._utterance.append(frame)
            self._above_threshold_frames += int(voiced)
            self._voice_frames += int(active)
            self._voice_run = self._voice_run + 1 if active else 0
            self._confirmed = self._confirmed or self._voice_run >= 6
            self._silence_count = 0 if active else self._silence_count + 1
            if self._silence_count >= self._silence_frames or len(self._utterance) >= self._max_frames:
                reason = "silence" if self._silence_count >= self._silence_frames else "max_duration"
                utterance = self._complete(reason)
                if utterance:
                    completed.append(utterance)
        return completed

    def finish(self) -> bytes:
        """在正常采音结束时取出未闭合语句；纯静音返回空值。"""
        if len(self._pending) % 2:
            raise ValueError("PCM 末尾不是完整采样")
        self.completed_stats = []
        self.discarded_stats = []
        if self._utterance:
            self._processed_samples += len(self._pending) // 2
            self._utterance.append(bytes(self._pending))
            result = self._complete("capture_eof")
        else:
            result = b""
        self._pending.clear()
        self._utterance.clear()
        self._preroll.clear()
        self._silence_count = 0
        self._above_threshold_frames = 0
        self._voice_frames = 0
        self._voice_run = 0
        self._confirmed = False
        return result


def encode_wav(pcm: bytes) -> bytes:
    """将 16 kHz 单声道 S16LE 封装为完整 WAV。"""
    if not pcm or len(pcm) % 2:
        raise ValueError("PCM 必须包含完整采样")
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(pcm)
    return output.getvalue()