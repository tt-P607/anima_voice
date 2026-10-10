"""从共享 Provider 接收通话 PCM，并通过单个输出流播放停顿分隔的回复。"""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator
from typing import Any

from .._internal_compat import create_task
from ..protocol import AnimaVoicePluginView, get_asr_output
from .backend import TTSRequest, get_tts_service

_BLOCK_BYTES = 9600
_playbacks: dict[str, set[asyncio.Task[Any]]] = {}


async def stop_stream_playback(stream_id: str | None = None) -> None:
    """取消并等待本插件的回复，不关闭共享 Provider 或其他消费方。"""

    groups = (
        [_playbacks.get(stream_id, set())]
        if stream_id is not None else list(_playbacks.values())
    )
    current = asyncio.current_task()
    tasks = {task for group in groups for task in group if task is not current}
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def play_streaming_speech(
    *, view: AnimaVoicePluginView, stream_id: str, platform: str,
    segments: list[tuple[str, float]], tts_params: dict[str, Any],
) -> int:
    """首块即播，EOF 后请求下一段；local_asr 完播后沿消息链记录语音。"""

    if view.config is None:
        raise RuntimeError("插件配置缺失")
    service = get_tts_service()
    asr = get_asr_output()
    output = asr if platform == "local_asr" else view.audio_player
    if output is None:
        raise RuntimeError("本机音频播放器未初始化")
    current = asyncio.current_task()
    if current is None:
        raise RuntimeError("通话播放必须在任务中运行")
    tasks = _playbacks.setdefault(stream_id, set())
    tasks.add(current)
    buffers: list[asyncio.Queue[bytes | Exception | None]] = [
        asyncio.Queue() for _ in segments
    ]
    recorded = bytearray()

    async def receive() -> None:
        """独立接收当前回复，单次请求结束后立即请求下一段。"""

        for (text, _), buffer in zip(segments, buffers, strict=True):
            try:
                async with service.open_pcm_stream(
                    TTSRequest(stream_id=stream_id, text=text, markers=dict(tts_params))
                ) as stream:
                    if (stream.sample_rate, stream.channels, stream.sample_format) != (
                        48000, 1, "s16le",
                    ):
                        raise ValueError("Provider PCM 必须是 48000Hz 单声道 s16le")
                    async for chunk in stream.chunks:
                        if not isinstance(chunk, bytes):
                            raise TypeError("PCM 块必须是 bytes")
                        for offset in range(0, len(chunk), _BLOCK_BYTES):
                            buffer.put_nowait(chunk[offset:offset + _BLOCK_BYTES])
            except Exception as error:
                buffer.put_nowait(error)
                raise
            buffer.put_nowait(None)

    async def continuous_pcm() -> AsyncIterator[bytes]:
        """拼接对齐后的音频和显式静音，拒绝空流或尾部不完整采样。"""

        for (_, wait), buffer in zip(segments, buffers, strict=True):
            frames = round(wait * 48000)
            while frames > 0:
                size = min(frames, _BLOCK_BYTES // 2)
                block = b"\x00\x00" * size
                if platform == "local_asr":
                    recorded.extend(block)
                yield block
                frames -= size
            pending = bytearray()
            received = False
            while True:
                chunk = await buffer.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                pending.extend(chunk)
                usable = len(pending) - len(pending) % 2
                if usable:
                    block = bytes(pending[:usable])
                    del pending[:usable]
                    if platform == "local_asr":
                        recorded.extend(block)
                    yield block
                    received = True
            if pending:
                raise ValueError("PCM 流结束时存在不完整采样帧")
            if not received:
                raise RuntimeError("TTS 未输出可播放的 PCM")

    receiver = create_task(receive(), f"anima_voice.pcm_receive.{id(current)}").task
    try:
        if platform == "local_asr":
            frames_written = await output.play_pcm_stream(continuous_pcm(), sample_rate=48000)
        else:
            async with asr.suppress_microphone_during_playback():
                frames_written = await output.play_pcm_stream(continuous_pcm(), sample_rate=48000)
        if frames_written <= sum(round(wait * 48000) for _, wait in segments):
            raise RuntimeError("TTS 未输出可播放的 PCM")
        await receiver
        if platform == "local_asr":
            audio = io.BytesIO()
            with wave.open(audio, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(recorded)
            text = " ".join(body for body, _ in segments)
            if not await asr.send_played_voice(audio.getvalue(), stream_id=stream_id, text=text):
                raise RuntimeError("已播放语音未成功记录到消息历史")
        return len(segments)
    finally:
        if not receiver.done():
            receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)
        tasks.discard(current)
        if not tasks:
            _playbacks.pop(stream_id, None)