"""anima_voice 专用 NDFC 会话编排器。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, ClassVar, cast

from src.app.plugin_system.api import stream_api
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.service_api import get_service
from src.app.plugin_system.base import (
    BaseChatter,
    Failure,
    Stop,
    Success,
    Wait,
    WaitResumeEvent,
)
from src.app.plugin_system.types import (
    ROLE,
    ChatStream,
    ChatType,
    LLMPayload,
    LLMRequest,
    LLMUsable,
    Message,
    ToolRegistry,
)

from .._internal_compat import cancel_task, create_task
from ..config import AnimaVoiceConfig
from ..constants import CHAT_CORE_SERVICE
from ..prompts import PLAIN_TEXT_REMINDER, SPEECH_RULES_TEMPLATE_NAME
from ..runtime import call_state
from ..voice_call.lifecycle import finalize_call
from . import request_factory
from .session_bridge import ChatCoreServiceLike

logger = get_logger("anima_voice.chatter")
_BLOCKED_TOOL_NAMES = frozenset(
    {
        "action-send_text",
        "action-stop_conversation",
        "action-sing_a_song",
        "action-say_and_perform",
        "pass_and_wait",
        "action-pass_and_wait",
        "create_agent",
        "get_agent",
        "kill_agent",
    }
)
_OWN_PASS_SIGNATURE = "anima_voice:action:pass_and_wait"
_OWN_ACTION_SIGNATURES = {
    "action-say": "anima_voice:action:say",
    "action-end_voice_call": "anima_voice:action:end_voice_call",
    "action-pass_and_wait": _OWN_PASS_SIGNATURE,
}


class AnimaVoiceChatter(BaseChatter):
    """为 local_asr 及显式接管的私聊执行语音会话。"""

    name = "anima_voice"
    description = "Anima 独立语音通话 Chatter。"
    associated_platforms: ClassVar[list[str]] = ["local_asr"]
    speech_rules_template_name: ClassVar[str] = SPEECH_RULES_TEMPLATE_NAME
    chat_type = ChatType.ALL
    stream_tick_interval = 0.1
    allow_message_buffer = False

    def __init__(self, stream_id: str, plugin: Any) -> None:
        """初始化绑定流和插件引用。"""

        super().__init__(stream_id, plugin)
        self._active_stream: ChatStream | None = None
        self._recorded_unreads: set[tuple[str, float]] = set()

    async def fetch_unreads(
        self, time_format: str = "%H:%M"
    ) -> tuple[str, list[Message]]:
        """保留未读文本，并将本次通话的用户发言入档一次。"""

        formatted_text, messages = await super().fetch_unreads(time_format)
        active = await call_state.get_active_call()
        if active is not None and active.caller_stream_id == self.stream_id:
            for message in messages:
                if message.is_sent or message.time < active.started_at:
                    continue
                identity = (message.message_id, message.time)
                if identity in self._recorded_unreads:
                    continue
                text = message.processed_plain_text or str(message.content)
                await call_state.record_user_message(
                    self.stream_id, text, ts=message.time
                )
                self._recorded_unreads.add(identity)
        return formatted_text, messages

    def create_request(
        self,
        task: str = "actor",
        request_name: str = "",
        with_reminder: str | None = None,
    ) -> LLMRequest:
        """使用配置模型集构造请求。"""

        config = self.plugin.config
        if not isinstance(config, AnimaVoiceConfig):
            return super().create_request(task, request_name, with_reminder)
        return request_factory.create_request(
            section=config.plugin,
            stream_id=self.stream_id,
            task=task,
            request_name=request_name or self.name,
            with_reminder=with_reminder,
        )

    async def inject_usables(self, request: LLMRequest) -> ToolRegistry:
        """只注入适用于语音通话的动作并移除冲突工具。"""

        usables: list[type[LLMUsable]] = await self.get_llm_usables()
        usables = await self.modify_llm_usables(usables)
        registry = ToolRegistry()
        for usable in usables:
            schema = usable.to_schema()
            function = schema.get("function", {})
            tool_name = str(function.get("name", ""))
            signature = str(getattr(usable, "get_signature", lambda: "")())
            if tool_name in _BLOCKED_TOOL_NAMES and signature != _OWN_PASS_SIGNATURE:
                continue
            owned_signature = _OWN_ACTION_SIGNATURES.get(tool_name)
            if owned_signature is not None and signature != owned_signature:
                continue
            registry.register(usable)
        tools = registry.get_all()
        if tools:
            request.add_payload(LLMPayload(ROLE.TOOL, cast(Any, tools)))
        return registry

    async def _timeout_watchdog(self) -> None:
        """在静默达到配置期限后统一结束通话。"""

        while await call_state.is_call_active_for_stream(self.stream_id):
            await asyncio.sleep(1.0)
            remaining = await call_state.get_remaining_seconds()
            if remaining is not None and remaining <= 0:
                await finalize_call(
                    stream_id=self.stream_id,
                    farewell="（通话超时挂断）",
                    end_reason="timeout",
                    plugin=self.plugin,
                )
                return

    def handle_plain_text_response(
        self, *, message: str, retry_count: int, response: Any
    ) -> dict[str, str]:
        """要求模型通过 say 输出语音。"""

        del message, response
        config = self.plugin.config
        limit = (
            config.plugin.plain_text_retry_limit
            if isinstance(config, AnimaVoiceConfig)
            else 1
        )
        if retry_count < limit:
            return {"action": "retry", "reminder_text": PLAIN_TEXT_REMINDER}
        return {"action": "wait", "reminder_text": ""}

    async def execute(
        self,
    ) -> AsyncGenerator[Wait | Success | Failure, WaitResumeEvent | None]:
        """创建并驱动 NDFC chat_core session。"""

        chat_stream = await stream_api.activate_stream(self.stream_id)
        if chat_stream is None:
            yield Failure("无法激活聊天流")
            return
        self._active_stream = chat_stream
        service = get_service(CHAT_CORE_SERVICE)
        if service is None:
            yield Failure("neo_default_chatter chat_core service 不可用")
            return
        session = cast(ChatCoreServiceLike, service).create_session(
            stream_id=self.stream_id, plugin=self.plugin
        )
        watchdog = create_task(
            self._timeout_watchdog(), f"anima_voice.timeout.{self.stream_id[:8]}"
        )
        try:
            runner = session.execute()
            resume_event: WaitResumeEvent | None = None
            while True:
                try:
                    result = await runner.asend(resume_event)
                except StopAsyncIteration:
                    return
                if isinstance(result, Stop):
                    yield Success("会话已结束")
                    return
                resume_event = yield result
        finally:
            self._active_stream = None
            cancel_task(watchdog)
