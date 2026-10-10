"""跨插件服务的本地协议与插件资源视图。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, cast

from src.app.plugin_system.api import adapter_api

from .config import AnimaVoiceConfig
from .constants import MICROPHONE_ADAPTER


class LocalMicrophone(Protocol):
    """本插件内置麦克风组件的通话和播放能力。"""

    async def start_voice_call_session(
        self,
        platform: str,
        stream_id: str,
        *,
        target_user_id: str,
        target_user_name: str,
        target_group_id: str,
    ) -> bool: ...

    async def end_voice_call_session(self) -> None: ...

    async def play_pcm_stream(
        self, chunks: AsyncIterator[bytes], *, sample_rate: int,
        channels: int = 1, sample_format: str = "s16le",
    ) -> int:
        """使用 ASR 的输出设备播放，并在播放期间抑制回采。"""
        ...

    async def send_played_voice(
        self, audio_data: bytes, *, stream_id: str, text: str,
    ) -> bool:
        """记录已播放语音的媒体、历史和事件，不重复播放。"""
        ...

    async def stop_pcm_playback(self) -> None:
        """停止自有 PCM 播放，不关闭共享 TTS。"""
        ...

    def suppress_microphone_during_playback(self) -> AbstractAsyncContextManager[None]:
        """持续抑制外部播放器的麦克风回采，不修改输出设备。"""
        ...


class ChatCoreService(Protocol):
    """NDFC chat_core 的会话创建能力。"""

    def create_session(self, *, stream_id: str, plugin: Any | None = None) -> Any: ...


class AudioPlayerLike(Protocol):
    """本插件播放器的最小播放接口。"""

    async def play_audio(self, audio_data: bytes) -> None: ...

    async def play_pcm_stream(
        self, chunks: AsyncIterator[bytes], *, sample_rate: int,
        channels: int = 1, sample_format: str = "s16le",
    ) -> int:
        """在本插件自有设备流中增量播放音频。"""
        ...

    async def stop(self) -> None:
        """停止当前播放，允许后续继续使用播放器。"""
        ...


def get_asr_output() -> LocalMicrophone:
    """从框架取得本插件唯一麦克风实例，不创建第二份识别资源。"""

    adapter = adapter_api.get_adapter(MICROPHONE_ADAPTER)
    methods = (
        "play_pcm_stream", "send_played_voice", "stop_pcm_playback",
        "suppress_microphone_during_playback",
    )
    if adapter is None or any(not callable(getattr(adapter, name, None)) for name in methods):
        raise RuntimeError("本插件麦克风组件未初始化")
    return cast(LocalMicrophone, adapter)


class AnimaVoicePluginView:
    """经运行时校验的本插件资源视图。"""

    def __init__(self, plugin: Any) -> None:
        """验证并包装框架注入的插件实例。"""

        if not hasattr(plugin, "audio_player") or not hasattr(
            plugin, "tts_capabilities"
        ):
            raise TypeError("组件绑定的插件缺少 anima_voice 运行时资源")
        self._plugin = plugin

    @property
    def config(self) -> AnimaVoiceConfig | None:
        """返回类型化插件配置。"""

        config = self._plugin.config
        return config if isinstance(config, AnimaVoiceConfig) else None

    @property
    def audio_player(self) -> AudioPlayerLike | None:
        """返回独立本机音频播放器。"""

        return self._plugin.audio_player

    @property
    def tts_capabilities(self) -> Any | None:
        """返回缓存的 TTS 参数描述。"""

        return self._plugin.tts_capabilities


def require_plugin(plugin: Any) -> AnimaVoicePluginView:
    """将组件的插件引用收窄为 anima_voice 资源视图。"""

    return AnimaVoicePluginView(plugin)
