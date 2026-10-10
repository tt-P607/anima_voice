"""单通话运行状态及通话消息记录。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal, TypedDict

DEFAULT_CALL_TIMEOUT_SECONDS = 300.0
CallEndReason = Literal["model", "user", "timeout", "manual", "plugin_unload"]


class CallMessage(TypedDict):
    """通话历史中的一条消息。"""

    role: str
    text: str
    ts: float


@dataclass(slots=True)
class ActiveCall:
    """当前通话的身份、时间和临时历史。"""

    caller_stream_id: str
    started_at: float
    last_activity_at: float
    timeout_seconds: float
    previous_chatter_signature: str = ""
    messages_in_call: list[CallMessage] = field(default_factory=list)


_active_call: ActiveCall | None = None
_lock = asyncio.Lock()
_DEDUP_WINDOW_SECONDS = 0.5


async def get_active_call() -> ActiveCall | None:
    """返回当前活动通话。"""

    async with _lock:
        return _active_call


async def is_call_active_for_stream(stream_id: str) -> bool:
    """判断给定聊天流是否由当前通话接管。"""

    async with _lock:
        return _active_call is not None and _active_call.caller_stream_id == stream_id


async def set_active_call(
    caller_stream_id: str,
    *,
    previous_chatter_signature: str = "",
    timeout_seconds: float = DEFAULT_CALL_TIMEOUT_SECONDS,
) -> ActiveCall:
    """占用唯一通话槽位并创建临时状态。"""

    global _active_call
    async with _lock:
        if _active_call is not None:
            raise RuntimeError("已有进行中的通话")
        now = time.time()
        _active_call = ActiveCall(
            caller_stream_id=caller_stream_id,
            started_at=now,
            last_activity_at=now,
            timeout_seconds=max(1.0, timeout_seconds),
            previous_chatter_signature=previous_chatter_signature,
        )
        return _active_call


async def end_call(reason: CallEndReason) -> ActiveCall | None:
    """清除通话状态并返回历史快照。"""

    del reason
    global _active_call
    async with _lock:
        snapshot = _active_call
        _active_call = None
        return snapshot


async def clear_active_call() -> None:
    """清除通话状态，用于启动失败回滚。"""

    global _active_call
    async with _lock:
        _active_call = None


async def get_remaining_seconds() -> float | None:
    """按最后一次活动计算静默超时剩余秒数。"""

    async with _lock:
        if _active_call is None:
            return None
        idle = time.time() - _active_call.last_activity_at
        return _active_call.timeout_seconds - idle


async def record_message(
    stream_id: str,
    role: str,
    text: str,
    *,
    ts: float | None = None,
) -> None:
    """为当前流记录通话消息并刷新活动时间。"""

    async with _lock:
        active = _active_call
        if active is None or active.caller_stream_id != stream_id or not text.strip():
            return
        timestamp = time.time() if ts is None else ts
        if role == "user" and active.messages_in_call:
            previous = active.messages_in_call[-1]
            if (
                previous["role"] == role
                and previous["text"] == text
                and abs(timestamp - previous["ts"]) <= _DEDUP_WINDOW_SECONDS
            ):
                return
        active.messages_in_call.append({"role": role, "text": text, "ts": timestamp})
        active.last_activity_at = time.time()


async def record_user_message(
    stream_id: str, text: str, *, ts: float | None = None
) -> None:
    """记录通话用户消息。"""

    await record_message(stream_id, "user", text, ts=ts)


async def record_assistant_message(stream_id: str, text: str) -> None:
    """记录通话助手消息。"""

    await record_message(stream_id, "assistant", text)


async def record_system_note(stream_id: str, text: str) -> None:
    """记录通话边界标注。"""

    await record_message(stream_id, "system", text)
