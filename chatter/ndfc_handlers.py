"""仅为当前绑定的 AnimaVoiceChatter 接管 NDFC 扩展事件。"""

from __future__ import annotations

from typing import Any, ClassVar

from src.app.plugin_system.api import chat_api
from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.base import BaseEventHandler
from src.app.plugin_system.types import ROLE, LLMPayload, Text

from ..config import AnimaVoiceConfig
from ..prompts import build_call_status, build_system_prompt
from ..runtime import call_state
from .core import AnimaVoiceChatter

_EVENTS = {
    "preprocess": "neo_default_chatter:preprocess",
    "inject_unread_payload": "neo_default_chatter:inject_unread_payload",
    "inject_usables": "neo_default_chatter:inject_usables",
    "create_request": "neo_default_chatter:create_request",
    "fetch_unreads": "neo_default_chatter:fetch_unreads",
    "format_unread_line": "neo_default_chatter:format_unread_line",
    "build_history_text": "neo_default_chatter:build_history_text",
}


def _replace_system_prompt(response: Any, system_text: str) -> None:
    """替换共享 LLM response 中的第一条 SYSTEM payload。"""

    payloads = getattr(response, "payloads", None)
    if payloads:
        for payload in payloads:
            if getattr(payload, "role", None) == ROLE.SYSTEM:
                payload.content = [Text(system_text)]
                return
    response.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_text)))


def _resolve(params: dict[str, Any]) -> AnimaVoiceChatter | None:
    """通过聊天 API 获取绑定实例并严格检查具体 Chatter 类型。"""

    stream_id = str(params.get("stream_id") or "")
    if not stream_id:
        return None
    chatter = chat_api.get_chatter_by_stream(stream_id)
    return chatter if isinstance(chatter, AnimaVoiceChatter) else None


class _VoiceHandler(BaseEventHandler):
    """按事件名将 NDFC 扩展点转交给语音 Chatter。"""

    weight = 100

    async def execute(
        self, event_name: str, params: dict[str, Any]
    ) -> tuple[EventDecision, dict[str, Any]]:
        """只处理本插件 chatter，并沿用 NDFC 已有参数键。"""

        chatter = _resolve(params)
        if chatter is None:
            return EventDecision.PASS, params
        if event_name == _EVENTS["preprocess"]:
            if "proceed" in params:
                params["proceed"] = True
                params["reason"] = "AnimaVoice 语音流直通"
                return EventDecision.STOP, params
            return EventDecision.PASS, params
        if event_name == _EVENTS["inject_usables"]:
            request = params.get("request")
            if request is None:
                return EventDecision.PASS, params
            params["tool_registry"] = await chatter.inject_usables(request)
            return EventDecision.STOP, params
        if event_name == _EVENTS["create_request"]:
            params["request"] = chatter.create_request(
                params.get("task_name") or "actor",
                params.get("request_name") or "",
                params.get("with_reminder"),
            )
            return EventDecision.STOP, params
        if event_name == _EVENTS["inject_unread_payload"]:
            response = params.get("response")
            if response is None:
                return EventDecision.PASS, params
            config = chatter.plugin.config
            custom_prompt = (
                config.plugin.custom_prompt
                if isinstance(config, AnimaVoiceConfig)
                else ""
            )
            active_call = await call_state.get_active_call()
            system_prompt = await build_system_prompt(
                custom_prompt=custom_prompt,
                enable_action_suspend=(
                    config.plugin.enable_action_suspend
                    if isinstance(config, AnimaVoiceConfig)
                    else True
                ),
                call_status=build_call_status(active_call, chatter.stream_id),
            )
            _replace_system_prompt(response, system_prompt)
            response.add_payload(
                LLMPayload(ROLE.USER, Text(str(params.get("formatted_text") or "")))
            )
            params["skip"] = True
            return EventDecision.STOP, params
        if event_name == _EVENTS["fetch_unreads"]:
            _, messages = await chatter.fetch_unreads()
            params["messages"] = messages
            return EventDecision.STOP, params
        if event_name == _EVENTS["format_unread_line"]:
            message = params.get("message")
            if message is None:
                return EventDecision.PASS, params
            params["formatted_line"] = chatter.format_message_line(
                message, params.get("time_format") or "%H:%M"
            )
            return EventDecision.STOP, params
        if event_name == _EVENTS["build_history_text"]:
            chat_stream = params.get("chat_stream")
            if chat_stream is None:
                return EventDecision.PASS, params
            messages = chat_stream.context.history_messages
            lines = [chatter.format_message_line(message) for message in messages]
            params["lines"] = lines
            return EventDecision.STOP, params
        return EventDecision.PASS, params


class AnimaVoicePreprocessHandler(_VoiceHandler):
    """替换本插件流的 NDFC preprocess 阶段。"""

    name = "anima_voice_preprocess"
    description = "转发 NDFC preprocess 到 anima_voice"
    init_subscribe: ClassVar[list[str]] = [_EVENTS["preprocess"]]


class AnimaVoiceInjectUnreadsHandler(_VoiceHandler):
    """注入语音通话未读消息。"""

    name = "anima_voice_inject_unreads"
    description = "转发 NDFC 未读 payload 注入"
    init_subscribe: ClassVar[list[str]] = [_EVENTS["inject_unread_payload"]]


class AnimaVoiceInjectUsablesHandler(_VoiceHandler):
    """过滤并注入语音通话工具。"""

    name = "anima_voice_inject_usables"
    description = "转发 NDFC 工具注入"
    init_subscribe: ClassVar[list[str]] = [_EVENTS["inject_usables"]]


class AnimaVoiceCreateRequestHandler(_VoiceHandler):
    """构造保留模型配置和 reminder 来源的请求。"""

    name = "anima_voice_create_request"
    description = "转发 NDFC 请求构造"
    init_subscribe: ClassVar[list[str]] = [_EVENTS["create_request"]]


class AnimaVoiceFetchUnreadsHandler(_VoiceHandler):
    """拉取当前语音 Chatter 的未读消息。"""

    name = "anima_voice_fetch_unreads"
    description = "转发 NDFC 未读拉取"
    init_subscribe: ClassVar[list[str]] = [_EVENTS["fetch_unreads"]]


class AnimaVoiceFormatUnreadLineHandler(_VoiceHandler):
    """格式化一条语音 Chatter 未读消息。"""

    name = "anima_voice_format_unread_line"
    description = "转发 NDFC 单条未读格式化"
    init_subscribe: ClassVar[list[str]] = [_EVENTS["format_unread_line"]]


class AnimaVoiceBuildHistoryTextHandler(_VoiceHandler):
    """格式化当前语音 Chatter 的历史消息。"""

    name = "anima_voice_build_history_text"
    description = "转发 NDFC 历史文本构造"
    init_subscribe: ClassVar[list[str]] = [_EVENTS["build_history_text"]]
