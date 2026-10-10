"""anima_voice 通话状态测试。"""

from __future__ import annotations

import pytest

from plugins.anima_voice.runtime import call_state


async def test_only_one_call_can_be_active() -> None:
    """唯一通话槽位拒绝第二次占用。"""

    await call_state.set_active_call("stream-a")
    with pytest.raises(RuntimeError, match="已有进行中的通话"):
        await call_state.set_active_call("stream-b")


async def test_timeout_uses_last_activity_and_end_returns_history() -> None:
    """消息刷新静默时间，结束返回消息快照。"""

    active = await call_state.set_active_call("stream-a", timeout_seconds=100)
    active.last_activity_at -= 50
    await call_state.record_user_message("stream-a", "你好", ts=1000)
    remaining = await call_state.get_remaining_seconds()
    ended = await call_state.end_call("user")

    assert remaining is not None and remaining > 99
    assert ended is not None
    assert ended.messages_in_call[0]["text"] == "你好"
    assert await call_state.get_active_call() is None


async def test_wrong_stream_and_duplicate_user_message_are_ignored() -> None:
    """只记录对应通话流，并按短时间窗去重用户消息。"""

    await call_state.set_active_call("stream-a")
    await call_state.record_user_message("stream-b", "错流", ts=1000)
    await call_state.record_user_message("stream-a", "重复", ts=1000)
    await call_state.record_user_message("stream-a", "重复", ts=1000.2)
    active = await call_state.get_active_call()

    assert active is not None
    assert len(active.messages_in_call) == 1


async def test_call_records_previous_signature_and_clamps_timeout() -> None:
    """状态保留原 chatter 签名并将超时下限限制为一秒。"""

    active = await call_state.set_active_call(
        "stream-a", previous_chatter_signature="default:chatter:main", timeout_seconds=0
    )

    assert active.previous_chatter_signature == "default:chatter:main"
    assert active.timeout_seconds == 1.0
    assert active.last_activity_at == active.started_at
    assert active.messages_in_call == []


async def test_call_activity_and_message_order_are_preserved() -> None:
    """消息按 system/user/assistant 顺序保存，助手消息刷新静默计时。"""

    active = await call_state.set_active_call("stream-a", timeout_seconds=100)
    active.last_activity_at -= 50
    await call_state.record_system_note("stream-a", "call start")
    await call_state.record_user_message("stream-a", "hello", ts=1000)
    await call_state.record_assistant_message("stream-a", "hi")

    remaining = await call_state.get_remaining_seconds()
    current = await call_state.get_active_call()

    assert remaining is not None and remaining > 99
    assert current is not None
    assert [message["role"] for message in current.messages_in_call] == [
        "system",
        "user",
        "assistant",
    ]


async def test_wrong_stream_message_is_discarded_and_end_is_idempotent() -> None:
    """错流消息不会进入历史，重复结束时返回空状态。"""

    await call_state.set_active_call("stream-a")
    await call_state.record_user_message("stream-b", "wrong stream")

    active = await call_state.get_active_call()
    first = await call_state.end_call("user")
    second = await call_state.end_call("user")

    assert active is not None and active.messages_in_call == []
    assert first is not None
    assert second is None
    assert await call_state.get_active_call() is None


async def test_same_user_text_after_dedup_window_is_kept() -> None:
    """超过短去重窗口的相同用户文本应保留为两次发言。"""

    await call_state.set_active_call("stream-a")
    await call_state.record_user_message("stream-a", "repeat", ts=1000)
    await call_state.record_user_message("stream-a", "repeat", ts=1005)

    active = await call_state.get_active_call()

    assert active is not None
    assert len(active.messages_in_call) == 2


async def test_missing_timestamp_duplicate_user_message_is_deduplicated() -> None:
    """没有 ASR 时间戳时，相邻重复文本仍按当前时间去重。"""

    await call_state.set_active_call("stream-a")
    await call_state.record_user_message("stream-a", "repeat")
    await call_state.record_user_message("stream-a", "repeat")

    active = await call_state.get_active_call()

    assert active is not None
    assert len(active.messages_in_call) == 1


async def test_active_call_query_is_scoped_to_caller_stream() -> None:
    """公开状态查询只匹配发起通话的流。"""

    assert await call_state.get_active_call() is None
    await call_state.set_active_call("stream-a")

    assert await call_state.is_call_active_for_stream("stream-a") is True
    assert await call_state.is_call_active_for_stream("stream-b") is False
