"""SnowLuma QQ 通话通知适配器。"""

from __future__ import annotations

from typing import Any, ClassVar

from src.app.plugin_system.api.event_api import EventDecision
from src.app.plugin_system.base import BaseEventHandler


class QQCallNoticeHandler(BaseEventHandler):
    """将原生 QQ 通话状态通知交给共享 runtime。"""

    name = "qq_call_notice"
    description = "处理 SnowLuma 原生 QQ 通话状态通知"
    init_subscribe: ClassVar[list[str]] = ["snowluma_adapter:notice_received"]

    async def execute(
        self, event_name: str, params: dict[str, Any]
    ) -> tuple[EventDecision, dict[str, Any]]:
        """仅向 runtime 转发 QQ 通话 notice，不改事件参数。"""
        if event_name not in self.init_subscribe:
            return EventDecision.PASS, params
        notice = params.get("notice")
        if not isinstance(notice, dict) or notice.get("notice_type") != "voice_call":
            return EventDecision.PASS, params
        runtime = getattr(self.plugin, "qq_calls", None)
        if runtime is not None:
            runtime.enqueue_notice(str(params["adapter_signature"]), notice)
        return EventDecision.SUCCESS, params