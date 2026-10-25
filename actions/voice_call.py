"""发起、接听、拒绝和结束原生 QQ 语音通话的 Action。"""

from __future__ import annotations

from typing import Annotated, ClassVar

from src.app.plugin_system.api import chat_api
from src.app.plugin_system.base import BaseAction
from src.app.plugin_system.types import ChatType

from ..voice_call.qq_calls import QQCallRuntime

_KFC_CHATTER = "kokoro_flow_chatter"


def _runtime(plugin: object) -> QQCallRuntime:
    """返回插件生命周期持有的 QQ runtime。"""
    runtime = getattr(plugin, "qq_calls", None)
    if runtime is None:
        raise RuntimeError("QQ 通话 runtime 尚未初始化")
    return runtime


def _is_supported_stream(action: BaseAction) -> bool:
    """确认当前动作位于 KFC 的 QQ 私聊。"""
    chatter = chat_api.get_chatter_by_stream(action.chat_stream.stream_id)
    return (
        action.chat_stream.platform == "qq"
        and str(action.chat_stream.chat_type).lower() == "private"
        and getattr(chatter, "name", "") == _KFC_CHATTER
    )


def _failure(error: Exception) -> tuple[bool, str]:
    """返回可读的控制失败原因。"""
    detail = str(error).strip() or type(error).__name__
    return False, f"QQ 通话操作失败：{detail}"


def _available(action: BaseAction) -> bool:
    """保持 KFC 私聊工具列表稳定，由执行时校验实际状态。"""
    return _is_supported_stream(action) and getattr(action.plugin, "qq_calls", None) is not None


class StartVoiceCallAction(BaseAction):
    """在 KFC QQ 私聊中发起原生 QQ 呼叫。"""

    name = "start_voice_call"
    associated_types: ClassVar[list[str]] = ["text"]
    chatter_allow: ClassVar[list[str]] = [_KFC_CHATTER]
    chat_type = ChatType.PRIVATE
    description = "在当前 KFC QQ 私聊中发起原生语音呼叫。"

    async def go_activate(self) -> bool:
        """只向已有通话 runtime 的 KFC QQ 私聊公开。"""
        return _available(self)

    async def execute(self) -> tuple[bool, str]:
        """显式请求 QQ 呼叫，不把请求回执当作接通确认。"""
        if not _is_supported_stream(self):
            return False, "原生 QQ 通话仅支持 KFC QQ 私聊"
        try:
            state = await _runtime(self.plugin).dial(self.chat_stream)
        except Exception as error:  # noqa: BLE001
            return _failure(error)
        return True, f"已发起 QQ 呼叫请求，当前状态：{state.get('phase', 'unknown')}；尚未确认接通。"


class AcceptVoiceCallAction(BaseAction):
    """接听当前 KFC QQ 私聊的来电。"""

    name = "accept_voice_call"
    associated_types: ClassVar[list[str]] = ["text"]
    chatter_allow: ClassVar[list[str]] = [_KFC_CHATTER]
    chat_type = ChatType.PRIVATE
    description = "接听当前 KFC QQ 私聊中的 QQ 语音来电。"

    async def go_activate(self) -> bool:
        """使后续到来的来电仍可使用已注册的接听动作。"""
        return _available(self)

    async def execute(self) -> tuple[bool, str]:
        """请求接听并等待后续通话状态通知。"""
        if not _is_supported_stream(self):
            return False, "原生 QQ 通话仅支持 KFC QQ 私聊"
        try:
            await _runtime(self.plugin).accept(self.chat_stream.stream_id)
        except Exception as error:  # noqa: BLE001
            return _failure(error)
        return True, "已请求接听 QQ 来电，尚未确认接通。"


class RejectVoiceCallAction(BaseAction):
    """拒绝当前 KFC QQ 私聊的来电。"""

    name = "reject_voice_call"
    associated_types: ClassVar[list[str]] = ["text"]
    chatter_allow: ClassVar[list[str]] = [_KFC_CHATTER]
    chat_type = ChatType.PRIVATE
    description = "拒绝当前 KFC QQ 私聊中的 QQ 语音来电。"

    async def go_activate(self) -> bool:
        """使后续到来的来电仍可使用已注册的拒接动作。"""
        return _available(self)

    async def execute(self) -> tuple[bool, str]:
        """请求拒接并等待后续通话状态通知。"""
        if not _is_supported_stream(self):
            return False, "原生 QQ 通话仅支持 KFC QQ 私聊"
        try:
            await _runtime(self.plugin).reject(self.chat_stream.stream_id)
        except Exception as error:  # noqa: BLE001
            return _failure(error)
        return True, "已请求拒接 QQ 来电，等待状态确认。"


class EndVoiceCallAction(BaseAction):
    """请求结束当前 KFC QQ 私聊通话。"""

    name = "end_voice_call"
    associated_types: ClassVar[list[str]] = ["text", "voice"]
    chatter_allow: ClassVar[list[str]] = [_KFC_CHATTER]
    chat_type = ChatType.PRIVATE
    description = "请求挂断当前 KFC QQ 语音通话。"

    async def go_activate(self) -> bool:
        """保留通话建立后需要的挂断动作。"""
        return _available(self)

    async def execute(
        self, farewell: Annotated[str, "可选的 QQ 通话告别语"] = ""
    ) -> tuple[bool, str]:
        """在可选语音告别后仍请求挂断，并明确报告每项结果。"""
        if not _is_supported_stream(self):
            return False, "原生 QQ 通话仅支持 KFC QQ 私聊"
        stream_id = self.chat_stream.stream_id
        try:
            runtime = _runtime(self.plugin)
            state = runtime.get_state(stream_id)
        except Exception as error:  # noqa: BLE001
            return _failure(error)
        if state is not None and state.get("phase") == "unknown":
            return False, "QQ 通话状态未知；当前只允许查询状态，不执行控制操作。"
        speech_error: Exception | None = None
        if farewell.strip() and state is not None and state.get("phase") == "connected":
            try:
                result = await runtime.speak(stream_id, farewell.strip())
                if result.get("drained") is not True:
                    raise RuntimeError("QQ 语音输出未确认排空")
            except Exception as error:  # noqa: BLE001
                speech_error = error
        try:
            await runtime.hangup(stream_id)
        except Exception as error:  # noqa: BLE001
            hangup_result = _failure(error)[1]
            if speech_error is not None:
                return False, f"告别语播放失败：{speech_error}；{hangup_result}"
            return False, hangup_result
        if speech_error is not None:
            return False, f"告别语播放失败：{speech_error}；已请求挂断，等待 QQ 状态确认。"
        return True, "已请求挂断，等待 QQ 状态确认。"
