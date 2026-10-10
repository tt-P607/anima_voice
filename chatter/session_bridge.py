"""NDFC chat_core Service 的本地结构协议。"""

from __future__ import annotations

from typing import Any, Protocol


class SessionLike(Protocol):
    """可执行的 NDFC 会话对象。"""

    def execute(self) -> Any: ...


class ChatCoreServiceLike(Protocol):
    """NDFC chat_core Service 的必要公开能力。"""

    def create_session(
        self, *, stream_id: str, plugin: Any | None = None
    ) -> SessionLike: ...
