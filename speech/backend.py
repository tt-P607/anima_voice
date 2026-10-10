"""通过公开 TTS Service 使用共享的完整音频与 PCM 合成能力。"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from src.app.plugin_system.api import service_api

from ..constants import TTS_SERVICE


@dataclass(slots=True)
class TTSRequest:
    """传给公开 TTS Service 的一次通话合成请求。"""

    stream_id: str
    text: str
    markers: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


class PCMStream(Protocol):
    """共享 TTS 的 PCM 格式与异步音频块。"""

    sample_rate: int
    channels: int
    sample_format: str
    chunks: AsyncIterator[bytes]


class SpeechService(Protocol):
    """通话依赖的 TTS Service 最小接口。"""

    async def synthesize(self, request: TTSRequest) -> Any:
        """返回带音频数据的完整合成响应。"""
        ...

    def open_pcm_stream(self, request: TTSRequest) -> AbstractAsyncContextManager[PCMStream]:
        """打开一次 PCM 请求，退出上下文释放请求资源。"""
        ...


def get_tts_service() -> SpeechService:
    """获取公开语音服务，不创建或关闭合成 runtime。"""

    service = service_api.get_service(TTS_SERVICE)
    if service is None:
        raise RuntimeError(f"TTS Service 未加载：{TTS_SERVICE}")
    if any(not callable(getattr(service, method, None)) for method in ("synthesize", "open_pcm_stream")):
        raise TypeError("TTS Service 未提供完整音频和 PCM 合成接口")
    return cast(SpeechService, service)


@dataclass(slots=True)
class TTSArtifact:
    """TTS 返回的音频和元数据。"""

    audio: bytes | None
    mime_type: str
    metadata: dict[str, Any]
    error: str | None = None


async def synthesize_audio(stream_id: str, text: str, params: dict[str, Any]) -> TTSArtifact:
    """请求共享 Service 生成完整音频，保留参数并拒绝空响应。"""

    response = await get_tts_service().synthesize(
        TTSRequest(stream_id=stream_id, text=text, markers=dict(params))
    )
    audio = base64.b64decode(response.audio_base64, validate=True)
    if not audio:
        raise RuntimeError("TTS 未返回可播放音频")
    return TTSArtifact(audio, response.mime_type, {})
