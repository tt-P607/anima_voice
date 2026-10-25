"""暴露唯一 QQ 通话 runtime 的插件 Service。"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.base import BaseService


class VoiceCallService(BaseService):
    """为插件消费者提供 QQ 通话状态和控制接口。"""

    name = "voice_call"
    description = "Anima 原生 QQ 语音通话控制"

    def _runtime(self) -> Any:
        """获取插件生命周期持有的唯一 runtime。"""
        runtime = getattr(self.plugin, "qq_calls", None)
        if runtime is None:
            raise RuntimeError("QQ 通话 runtime 尚未初始化")
        return runtime

    def get_state(self, stream_id: str) -> dict[str, Any] | None:
        """查询指定流的 QQ 通话状态。"""
        return self._runtime().get_state(stream_id)

    def get_speech_options(self) -> dict[str, Any]:
        """查询共享 TTS 当前可用风格与默认风格。"""
        return self._runtime().get_speech_options()

    async def speak(
        self, stream_id: str, text: str, *, voice_style: str = "", trace_id: str = ""
    ) -> dict[str, Any]:
        """通过 QQ 通话发送语音。"""
        return await self._runtime().speak(stream_id, text, voice_style=voice_style, trace_id=trace_id)

    async def dial(self, stream: Any) -> dict[str, Any]:
        """从现有聊天流发起 QQ 呼叫。"""
        return await self._runtime().dial(stream)

    async def accept(self, stream_id: str) -> dict[str, Any]:
        """接听指定流的 QQ 来电。"""
        return await self._runtime().accept(stream_id)

    async def reject(self, stream_id: str) -> dict[str, Any]:
        """拒绝指定流的 QQ 来电。"""
        return await self._runtime().reject(stream_id)

    async def hangup(self, stream_id: str) -> dict[str, Any]:
        """请求挂断指定流的 QQ 通话。"""
        return await self._runtime().hangup(stream_id)