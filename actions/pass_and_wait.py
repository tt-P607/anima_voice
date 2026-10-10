"""NDFC 会话等待动作。"""

from __future__ import annotations

from typing import Annotated, ClassVar

from src.app.plugin_system.base import BaseAction


class PassAndWaitAction(BaseAction):
    """登记等待新输入或固定时长。"""

    name = "pass_and_wait"
    associated_types: ClassVar[list[str]] = ["text"]
    chatter_allow: ClassVar[list[str]] = ["anima_voice"]
    description = "通话回复后等待用户继续说话；可设置秒数后主动恢复。"

    async def execute(
        self, seconds: Annotated[float | None, "等待秒数"] = None
    ) -> tuple[bool, str]:
        """向 NDFC 会话返回等待控制意图。"""

        if seconds is None:
            return True, "等待新的用户输入"
        return True, f"等待 {seconds} 秒"
