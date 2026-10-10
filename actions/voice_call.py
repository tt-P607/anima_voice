"""发起和结束本地语音通话的 Action。"""

from __future__ import annotations

from typing import Annotated, ClassVar

from src.app.plugin_system.base import BaseAction
from src.app.plugin_system.types import ChatType

from ..runtime import call_state
from ..voice_call.lifecycle import finalize_call, start_call


class StartVoiceCallAction(BaseAction):
    """在任意允许的私聊 Chatter 中发起通话。"""

    name = "start_voice_call"
    associated_types: ClassVar[list[str]] = ["text"]
    chatter_allow: ClassVar[list[str]] = []
    chat_type = ChatType.PRIVATE
    description = "在当前私聊中接通本地语音通话。"

    async def go_activate(self) -> bool:
        """只在无活动通话时公开。"""

        return await call_state.get_active_call() is None

    async def execute(
        self, reason: Annotated[str, "接通提示语"] = ""
    ) -> tuple[bool, str]:
        """验证私聊并由 anima_voice 接管。"""

        if str(self.chat_stream.chat_type).lower() != "private":
            return False, "语音通话仅支持私聊"
        return await start_call(
            stream=self.chat_stream, plugin=self.plugin, reason=reason
        )


class EndVoiceCallAction(BaseAction):
    """结束本插件拥有的当前通话。"""

    name = "end_voice_call"
    associated_types: ClassVar[list[str]] = ["text", "voice"]
    chatter_allow: ClassVar[list[str]] = ["anima_voice"]
    chat_type = ChatType.PRIVATE
    description = "挂断当前语音通话并恢复此前聊天器。"

    async def go_activate(self) -> bool:
        """仅在当前流由活动通话占用时公开。"""

        return await call_state.is_call_active_for_stream(self.chat_stream.stream_id)

    async def execute(
        self, farewell: Annotated[str, "本机播放的告别语"] = ""
    ) -> tuple[bool, str]:
        """结束当前通话。"""

        return await finalize_call(
            stream_id=self.chat_stream.stream_id,
            farewell=farewell.strip(),
            end_reason="model",
            plugin=self.plugin,
        )
