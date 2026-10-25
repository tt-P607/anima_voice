"""anima_voice 组件装配与独立资源生命周期。"""

from __future__ import annotations

from typing import Any, ClassVar

from src.app.plugin_system.api import adapter_api, service_api
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BasePlugin, register_plugin

from .actions import (
    AcceptVoiceCallAction,
    EndVoiceCallAction,
    PassAndWaitAction,
    RejectVoiceCallAction,
    SayAction,
    StartVoiceCallAction,
)
from .asr import VoiceMicrophoneAdapter
from .audio.player import AudioPlayer
from .chatter import AnimaVoiceChatter
from .chatter.ndfc_handlers import (
    AnimaVoiceBuildHistoryTextHandler,
    AnimaVoiceCreateRequestHandler,
    AnimaVoiceFetchUnreadsHandler,
    AnimaVoiceFormatUnreadLineHandler,
    AnimaVoiceInjectUnreadsHandler,
    AnimaVoiceInjectUsablesHandler,
    AnimaVoicePreprocessHandler,
)
from .commands import VoiceCommand
from .config import AnimaVoiceConfig
from .constants import MICROPHONE_ADAPTER, TTS_SERVICE
from .runtime import call_state
from .speech.streaming import stop_stream_playback
from .voice_call.events import QQCallNoticeHandler
from .voice_call.lifecycle import finalize_call
from .voice_call.qq_calls import QQCallRuntime
from .voice_call.service import VoiceCallService

logger = get_logger("anima_voice.plugin")


@register_plugin
class AnimaVoicePlugin(BasePlugin):
    """注册独立语音通话 Chatter 与动作。"""

    plugin_name = "anima_voice"
    configs: ClassVar[list[type]] = [AnimaVoiceConfig]
    audio_player: AudioPlayer | None = None
    tts_capabilities: Any | None = None
    qq_calls: QQCallRuntime | None = None

    async def on_plugin_loaded(self) -> None:
        """初始化本地播放器并缓存公开 TTS service 的能力描述。"""

        config = self.config
        if not isinstance(config, AnimaVoiceConfig):
            return
        self.qq_calls = QQCallRuntime(config)
        target = config.audio.loudness_target_dbfs
        self.audio_player = AudioPlayer(config.audio.output_device, target)
        service = service_api.get_service(TTS_SERVICE)
        if service is None:
            raise RuntimeError(f"TTS service 未加载：{TTS_SERVICE}")
        self.tts_capabilities = service.get_capabilities()

    async def on_plugin_unloaded(self) -> None:
        """结束本插件通话并释放自有播放器资源。"""

        errors: list[BaseException] = []
        if self.qq_calls is not None:
            try:
                await self.qq_calls.close()
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
            finally:
                self.qq_calls = None
        try:
            await stop_stream_playback()
            active = await call_state.get_active_call()
            if active is not None:
                result = await finalize_call(
                    stream_id=active.caller_stream_id,
                    farewell="服务停止，通话中断。",
                    end_reason="plugin_unload",
                    plugin=self,
                )
                if not result[0]:
                    logger.warning(result[1])
        except BaseException as error:  # noqa: BLE001
            errors.append(error)
        finally:
            try:
                if (
                    adapter_api.get_adapter(MICROPHONE_ADAPTER) is not None
                    and not await adapter_api.stop_adapter(MICROPHONE_ADAPTER)
                ):
                    raise RuntimeError("停止本插件麦克风组件失败")
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
            finally:
                try:
                    if self.audio_player is not None:
                        await self.audio_player.close()
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
                finally:
                    self.audio_player = None
                    self.tts_capabilities = None
        if errors:
            raise BaseExceptionGroup("anima_voice 资源清理失败", errors)

    def get_components(self) -> list[type]:
        """返回语音通话插件组件。"""

        config = self.config if isinstance(self.config, AnimaVoiceConfig) else None
        if config is not None and not config.plugin.enabled:
            return []
        return [
            VoiceMicrophoneAdapter,
            AnimaVoiceChatter,
            SayAction,
            StartVoiceCallAction,
            EndVoiceCallAction,
            AcceptVoiceCallAction,
            RejectVoiceCallAction,
            PassAndWaitAction,
            VoiceCommand,
            VoiceCallService,
            QQCallNoticeHandler,
            AnimaVoicePreprocessHandler,
            AnimaVoiceInjectUnreadsHandler,
            AnimaVoiceInjectUsablesHandler,
            AnimaVoiceCreateRequestHandler,
            AnimaVoiceFetchUnreadsHandler,
            AnimaVoiceFormatUnreadLineHandler,
            AnimaVoiceBuildHistoryTextHandler,
        ]
