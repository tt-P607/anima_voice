"""语音通话生命周期与工具注入回归测试。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from plugins.anima_voice import _internal_compat
from plugins.anima_voice.actions import (
    EndVoiceCallAction,
    PassAndWaitAction,
    SayAction,
    _tts_schema,
)
from plugins.anima_voice.chatter.core import AnimaVoiceChatter
from plugins.anima_voice.config import AnimaVoiceConfig
from plugins.anima_voice.runtime import call_state
from plugins.anima_voice.speech import streaming
from plugins.anima_voice.voice_call import lifecycle
from plugins.anima_voice.voice_call.identity import resolve_caller_identity


class _Usable:
    """提供实际工具 schema 形状的测试 usable。"""

    tool_name = ""
    signature = ""

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        """返回工具注册器使用的 OpenAI function schema。"""

        return {"type": "function", "function": {"name": cls.tool_name}}

    @classmethod
    def get_signature(cls) -> str:
        """返回测试组件签名。"""

        return cls.signature


class _Request:
    """记录 chatter 注入的工具 payload。"""

    def __init__(self) -> None:
        self.payloads: list[Any] = []

    def add_payload(self, payload: Any) -> None:
        """保存注入 payload。"""

        self.payloads.append(payload)


class _Chatter:
    """代表本插件 chatter 身份的最小对象。"""

    def get_signature(self) -> str:
        """返回 chatter 签名。"""

        return "anima_voice:chatter:anima_voice"


class _ThirdPartyChatter:
    """代表接管流的第三方 chatter。"""

    def get_signature(self) -> str:
        """返回第三方签名。"""

        return "third_party:chatter:other"


@pytest.mark.asyncio
async def test_inject_usables_keeps_owned_voice_actions_and_ordinary_external_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """按真实 schema 名和组件签名只排除冲突工具。"""

    for action in (SayAction, EndVoiceCallAction, PassAndWaitAction):
        monkeypatch.setattr(action, "_plugin_", "anima_voice")
    monkeypatch.setattr(
        _tts_schema, "get_service",
        lambda _: SimpleNamespace(get_capabilities=lambda: SimpleNamespace()),
    )
    assert [
        action.to_schema()["function"]["name"]
        for action in (SayAction, EndVoiceCallAction, PassAndWaitAction)
    ] == ["action-say", "action-end_voice_call", "action-pass_and_wait"]

    class ForeignWait(_Usable):
        tool_name = "action-pass_and_wait"
        signature = "anima_chatter:action:pass_and_wait"

    class GlobalWait(_Usable):
        tool_name = "pass_and_wait"
        signature = "neo_default_chatter:action:pass_and_wait"

    class ForeignSay(_Usable):
        tool_name = "action-say"
        signature = "another_voice:action:say"

    class LiveAction(_Usable):
        tool_name = "action-sing_a_song"
        signature = "anima_chatter:action:sing_a_song"

    class OrdinaryTool(_Usable):
        tool_name = "lookup_info"
        signature = "utility:tool:lookup_info"

    chatter = AnimaVoiceChatter("stream", object())
    usables = [
        SayAction,
        EndVoiceCallAction,
        PassAndWaitAction,
        ForeignWait,
        GlobalWait,
        ForeignSay,
        LiveAction,
        OrdinaryTool,
    ]

    async def get_usables() -> list[type]:
        return usables

    async def modify_usables(items: list[type]) -> list[type]:
        return items

    monkeypatch.setattr(chatter, "get_llm_usables", get_usables)
    monkeypatch.setattr(chatter, "modify_llm_usables", modify_usables)
    request = _Request()
    registry = await chatter.inject_usables(request)

    assert registry.get_all_names() == [
        "action-end_voice_call",
        "action-pass_and_wait",
        "action-say",
        "lookup_info",
    ]
    assert len(request.payloads) == 1


@pytest.mark.asyncio
async def test_start_asr_failure_releases_call_slot_and_previous_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ASR 启动失败时回滚活动状态且释放缓存的原 chatter。"""

    previous = _ThirdPartyChatter()
    monkeypatch.setattr(lifecycle.chat_api, "get_chatter_by_stream", lambda _: previous)
    monkeypatch.setattr(
        lifecycle, "resolve_caller_identity", lambda _: ("user", "User")
    )
    monkeypatch.setattr(lifecycle, "start_asr_voice_session", _false_async)
    monkeypatch.setattr(lifecycle, "end_asr_voice_session", _noop_async)
    monkeypatch.setattr(lifecycle, "restart_stream_loop", _noop_async)
    stream = type("Stream", (), {"stream_id": "stream", "platform": "qq"})()
    plugin = type(
        "Plugin", (), {"audio_player": None, "tts_capabilities": None, "config": None}
    )()

    result = await lifecycle.start_call(stream=stream, plugin=plugin, reason="hello")

    assert result[0] is False
    assert await call_state.get_active_call() is None
    assert "stream" not in lifecycle._previous_chatters


@pytest.mark.asyncio
async def test_start_slot_conflict_does_not_clear_existing_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """并发竞争失败不能清除先占用通话槽位的调用。"""

    previous = _ThirdPartyChatter()
    monkeypatch.setattr(lifecycle.chat_api, "get_chatter_by_stream", lambda _: previous)
    monkeypatch.setattr(
        lifecycle, "resolve_caller_identity", lambda _: ("user", "User")
    )
    stream = type("Stream", (), {"stream_id": "stream-b", "platform": "qq"})()
    plugin = type(
        "Plugin", (), {"audio_player": None, "tts_capabilities": None, "config": None}
    )()
    await call_state.set_active_call("stream-a")

    result = await lifecycle.start_call(stream=stream, plugin=plugin, reason="hello")

    active = await call_state.get_active_call()
    assert result[0] is False
    assert active is not None and active.caller_stream_id == "stream-a"
    assert "stream-b" not in lifecycle._previous_chatters


@pytest.mark.asyncio
async def test_start_call_adds_notice_boundary_to_stream_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """接通成功时在流通用历史中插入与通话状态一致的 NOTICE。"""

    history: list[Any] = []
    sequence: list[str] = []
    stream = type(
        "Stream",
        (),
        {
            "stream_id": "stream",
            "platform": "qq",
            "context": type("Context", (), {"add_history_message": history.append})(),
        },
    )()
    previous = _ThirdPartyChatter()
    bound: list[Any] = []

    class VoiceChatter:
        def __init__(self, **_: Any) -> None:
            pass

    async def started(*_: Any, **__: Any) -> bool:
        return True

    async def send_text(*_: Any, **__: Any) -> bool:
        return True

    async def no_speak(*_: Any, **__: Any) -> None:
        return None

    async def publish(*_: Any, **__: Any) -> None:
        sequence.append("event")

    async def restart(_: str) -> None:
        sequence.append("restart")

    monkeypatch.setattr(lifecycle.chat_api, "get_chatter_by_stream", lambda _: previous)
    monkeypatch.setattr(lifecycle.chat_api, "get_chatter_class", lambda _: VoiceChatter)
    monkeypatch.setattr(
        lifecycle.chat_api,
        "bind_chatter_for_stream",
        lambda _, item: bound.append(item),
    )
    monkeypatch.setattr(
        lifecycle, "resolve_caller_identity", lambda _: ("user", "User")
    )
    monkeypatch.setattr(lifecycle, "start_asr_voice_session", started)
    monkeypatch.setattr(lifecycle, "restart_stream_loop", restart)
    monkeypatch.setattr(lifecycle, "speak", no_speak)
    monkeypatch.setattr(lifecycle, "_publish_started", publish)
    monkeypatch.setattr("src.app.plugin_system.api.send_api.send_text", send_text)
    plugin = type(
        "Plugin", (), {"audio_player": None, "tts_capabilities": None, "config": None}
    )()

    result = await lifecycle.start_call(stream=stream, plugin=plugin, reason="hello")

    assert result[0] is True
    assert len(history) == 1
    assert "语音通话开始" in history[0].processed_plain_text
    assert "下一条 [语音通话结束]" in history[0].processed_plain_text
    assert len(bound) == 1
    assert sequence == ["event", "restart"]


@pytest.mark.asyncio
async def test_call_boundary_uses_pcm_and_suppresses_owned_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """streaming 通话边界使用共享 PCM，并在本机播放期间抑制自有麦克风。"""

    calls: list[str] = []

    class Service:
        """提供通话边界所需的 PCM 流。"""

        @asynccontextmanager
        async def open_pcm_stream(self, _: Any) -> Any:
            async def chunks() -> Any:
                yield b"\x01\x00\x02\x00"

            try:
                yield SimpleNamespace(
                    sample_rate=48000, channels=1, sample_format="s16le",
                    chunks=chunks(),
                )
            finally:
                calls.append("pcm_closed")

        async def synthesize(self, _: Any) -> Any:
            pytest.fail("streaming call boundary must request PCM")

    class Output:
        """记录通话提示的 PCM 播放。"""

        async def play_pcm_stream(self, chunks: Any, **kwargs: Any) -> int:
            calls.append("play")
            assert kwargs == {"sample_rate": 48000}
            frames = 0
            async for chunk in chunks:
                frames += len(chunk) // 2
            return frames

    class ASROutput:
        """记录自有麦克风播放抑制范围。"""

        @asynccontextmanager
        async def suppress_microphone_during_playback(self) -> Any:
            calls.append("suppressed")
            try:
                yield
            finally:
                calls.append("restored")

    config = AnimaVoiceConfig()
    config.tts.streaming_enabled = True
    output = Output()
    asr_output = ASROutput()
    plugin = SimpleNamespace(
        config=config, audio_player=output, tts_capabilities=None,
    )
    monkeypatch.setattr(streaming, "get_tts_service", lambda: Service())
    monkeypatch.setattr(streaming, "get_asr_output", lambda: asr_output)
    monkeypatch.setattr(
        lifecycle, "get_asr_output",
        lambda: pytest.fail("PCM playback obtains suppression from streaming adapter"),
    )

    await lifecycle.speak(plugin, "stream", "hello")

    assert calls == ["suppressed", "play", "pcm_closed", "restored"]


@pytest.mark.asyncio
async def test_cancelled_start_rolls_back_asr_and_owned_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """接通途中取消仍关闭已尝试的 ASR 会话，不遗留通话占位。"""

    cleanup: list[str] = []

    async def start(*_: Any, **__: Any) -> bool:
        raise asyncio.CancelledError

    async def end() -> None:
        cleanup.append("asr")

    monkeypatch.setattr(lifecycle, "resolve_caller_identity", lambda _: ("user", "User"))
    monkeypatch.setattr(lifecycle.chat_api, "get_chatter_by_stream", lambda _: _ThirdPartyChatter())
    monkeypatch.setattr(lifecycle, "start_asr_voice_session", start)
    monkeypatch.setattr(lifecycle, "end_asr_voice_session", end)
    stream = type("Stream", (), {"stream_id": "stream", "platform": "qq"})()
    plugin = type("Plugin", (), {"config": None, "audio_player": None, "tts_capabilities": None})()
    with pytest.raises(asyncio.CancelledError):
        await lifecycle.start_call(stream=stream, plugin=plugin, reason="hello")
    assert cleanup == ["asr"]
    assert await call_state.get_active_call() is None
    assert "stream" not in lifecycle._previous_chatters


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_with_third_party", [False, True])
async def test_finalize_call_cleans_up_after_failures_without_overriding_takeover(
    monkeypatch: pytest.MonkeyPatch,
    replace_with_third_party: bool,
) -> None:
    """挂断异常仍释放状态并发事件，第三方 chatter 不被重绑或重启。"""

    previous = _ThirdPartyChatter()
    own = _Chatter()
    current: list[Any] = [_ThirdPartyChatter() if replace_with_third_party else own]
    binds: list[Any] = []
    restarts: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []
    sequence: list[str] = []

    def bind(_: str, chatter: Any) -> None:
        binds.append(chatter)
        current[0] = chatter

    async def restart(stream_id: str) -> None:
        restarts.append(stream_id)
        sequence.append("restart")

    async def fail_asr() -> None:
        raise RuntimeError("asr down")

    async def fail_speak(*_: Any) -> None:
        raise RuntimeError("audio down")

    async def publish(name: str, params: dict[str, Any]) -> None:
        events.append((name, params))
        sequence.append("event")

    monkeypatch.setattr(
        lifecycle.chat_api, "get_chatter_by_stream", lambda _: current[0]
    )
    monkeypatch.setattr(lifecycle.chat_api, "bind_chatter_for_stream", bind)
    monkeypatch.setattr(
        lifecycle.chat_api,
        "restore_stream_to_default",
        lambda _: pytest.fail("unexpected restore"),
    )
    monkeypatch.setattr(lifecycle, "restart_stream_loop", restart)
    monkeypatch.setattr(lifecycle, "end_asr_voice_session", fail_asr)
    monkeypatch.setattr(lifecycle, "speak", fail_speak)
    monkeypatch.setattr(lifecycle.stream_api, "get_stream", _none_async)
    monkeypatch.setattr(lifecycle.event_api, "publish_event", publish)
    await call_state.set_active_call(
        "stream", previous_chatter_signature=previous.get_signature()
    )
    lifecycle._previous_chatters["stream"] = previous
    plugin = type(
        "Plugin", (), {"audio_player": None, "tts_capabilities": None, "config": None}
    )()

    result = await lifecycle.finalize_call(
        stream_id="stream", farewell="bye", end_reason="manual", plugin=plugin
    )

    assert result[0] is False
    assert "ASR 关闭失败" in result[1]
    assert "告别语播放失败" in result[1]
    assert await call_state.get_active_call() is None
    assert events[0][0] == lifecycle.EVENT_VOICE_CALL_ENDED
    assert "stream" not in lifecycle._previous_chatters
    if replace_with_third_party:
        assert binds == []
        assert restarts == []
    else:
        assert binds == [previous]
        assert restarts == ["stream"]
        assert sequence == ["event", "restart"]


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_succeeds", [True, False])
async def test_restart_stream_loop_rejects_framework_failure(
    monkeypatch: pytest.MonkeyPatch,
    restart_succeeds: bool,
) -> None:
    """框架重启成功正常返回，失败时产生明确异常。"""

    restart = AsyncMock(return_value=restart_succeeds)
    manager = SimpleNamespace(restart_stream_loop=restart)
    monkeypatch.setattr(
        "src.core.transport.distribution.stream_loop_manager.get_stream_loop_manager",
        lambda: manager,
    )

    if restart_succeeds:
        await _internal_compat.restart_stream_loop("stream")
    else:
        with pytest.raises(RuntimeError, match="流循环重启失败"):
            await _internal_compat.restart_stream_loop("stream")
    restart.assert_awaited_once_with("stream")


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback_restart_succeeds", [True, False])
async def test_start_restart_failure_rolls_back_asr_binding_and_call_slot(
    monkeypatch: pytest.MonkeyPatch,
    rollback_restart_succeeds: bool,
) -> None:
    """接通重启失败时关闭 ASR 并恢复绑定，回滚重启失败也释放状态。"""

    previous = _ThirdPartyChatter()
    current: list[Any] = [previous]
    history: list[Any] = []

    class VoiceChatter(_Chatter):
        """接收运行时装配参数的本插件 chatter 替身。"""

        def __init__(self, **_: Any) -> None:
            """接收 chatter 装配参数。"""

    def bind(_: str, chatter: Any) -> None:
        """记录当前绑定的 chatter。"""

        current[0] = chatter

    restart = AsyncMock(side_effect=[False, rollback_restart_succeeds])
    manager = SimpleNamespace(restart_stream_loop=restart)
    end_asr = AsyncMock()
    monkeypatch.setattr(
        "src.core.transport.distribution.stream_loop_manager.get_stream_loop_manager",
        lambda: manager,
    )
    monkeypatch.setattr(lifecycle.chat_api, "get_chatter_by_stream", lambda _: current[0])
    monkeypatch.setattr(lifecycle.chat_api, "get_chatter_class", lambda _: VoiceChatter)
    monkeypatch.setattr(lifecycle.chat_api, "bind_chatter_for_stream", bind)
    monkeypatch.setattr(lifecycle, "resolve_caller_identity", lambda _: ("user", "User"))
    monkeypatch.setattr(lifecycle, "start_asr_voice_session", AsyncMock(return_value=True))
    monkeypatch.setattr(lifecycle, "end_asr_voice_session", end_asr)
    monkeypatch.setattr(lifecycle, "speak", _noop_async)
    monkeypatch.setattr(lifecycle, "_publish_started", _noop_async)
    monkeypatch.setattr(
        "src.app.plugin_system.api.send_api.send_text", AsyncMock(return_value=True)
    )
    stream = SimpleNamespace(
        stream_id="stream",
        platform="qq",
        context=SimpleNamespace(add_history_message=history.append),
    )
    plugin = SimpleNamespace(config=None, audio_player=None, tts_capabilities=None)

    result = await lifecycle.start_call(stream=stream, plugin=plugin, reason="hello")

    assert result[0] is False
    assert "流循环重启失败" in result[1]
    assert ("Chatter 回滚失败" in result[1]) is (not rollback_restart_succeeds)
    end_asr.assert_awaited_once_with()
    assert restart.await_count == 2
    assert all(arguments.args == ("stream",) for arguments in restart.await_args_list)
    assert current[0] is previous
    assert await call_state.get_active_call() is None
    assert "stream" not in lifecycle._previous_chatters


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_succeeds", [True, False])
async def test_finalize_reports_restart_result_after_releasing_call(
    monkeypatch: pytest.MonkeyPatch,
    restart_succeeds: bool,
) -> None:
    """挂断始终释放状态和恢复绑定，重启失败时返回部分失败。"""

    previous = _ThirdPartyChatter()
    current: list[Any] = [_Chatter()]

    def bind(_: str, chatter: Any) -> None:
        """记录恢复的 chatter 绑定。"""

        current[0] = chatter

    restart = AsyncMock(return_value=restart_succeeds)
    manager = SimpleNamespace(restart_stream_loop=restart)
    end_asr = AsyncMock()
    publish = AsyncMock()
    monkeypatch.setattr(
        "src.core.transport.distribution.stream_loop_manager.get_stream_loop_manager",
        lambda: manager,
    )
    monkeypatch.setattr(lifecycle.chat_api, "get_chatter_by_stream", lambda _: current[0])
    monkeypatch.setattr(lifecycle.chat_api, "bind_chatter_for_stream", bind)
    monkeypatch.setattr(lifecycle, "end_asr_voice_session", end_asr)
    monkeypatch.setattr(lifecycle, "speak", _noop_async)
    monkeypatch.setattr(lifecycle.stream_api, "get_stream", _none_async)
    monkeypatch.setattr(lifecycle.event_api, "publish_event", publish)
    await call_state.set_active_call(
        "stream", previous_chatter_signature=previous.get_signature()
    )
    lifecycle._previous_chatters["stream"] = previous
    plugin = SimpleNamespace(config=None, audio_player=None, tts_capabilities=None)

    result = await lifecycle.finalize_call(
        stream_id="stream", farewell="bye", end_reason="manual", plugin=plugin
    )

    assert result[0] is restart_succeeds
    if not restart_succeeds:
        assert "通话已清理，但部分步骤失败" in result[1]
        assert "流循环重启失败" in result[1]
    end_asr.assert_awaited_once_with()
    restart.assert_awaited_once_with("stream")
    publish.assert_awaited_once()
    assert publish.await_args is not None
    assert publish.await_args.args[0] == lifecycle.EVENT_VOICE_CALL_ENDED
    assert current[0] is previous
    assert await call_state.get_active_call() is None
    assert "stream" not in lifecycle._previous_chatters


async def _false_async(*_: Any, **__: Any) -> bool:
    """返回 ASR 启动失败。"""

    return False


async def _noop_async(*_: Any, **__: Any) -> None:
    """执行无副作用的异步替身。"""


async def _none_async(*_: Any, **__: Any) -> None:
    """返回无流对象的异步替身。"""


def test_resolve_caller_identity_prefers_latest_unread_message() -> None:
    """最新未读对方消息优先于聊天历史。"""

    def message(sender_id: str, sender_name: str = "") -> Any:
        return type(
            "Message",
            (),
            {"sender_id": sender_id, "sender_name": sender_name, "message_type": None},
        )()

    stream = type(
        "Stream",
        (),
        {
            "bot_id": "bot",
            "context": type(
                "Context",
                (),
                {
                    "unread_messages": [message("new-user", "未读用户")],
                    "history_messages": [message("old-user", "历史用户")],
                },
            )(),
        },
    )()

    assert resolve_caller_identity(stream) == ("new-user", "未读用户")


def test_resolve_caller_identity_uses_latest_counterpart_in_history() -> None:
    """没有未读消息时选择历史中最近的对方消息。"""

    history = [
        type(
            "Message",
            (),
            {"sender_id": "old-user", "sender_name": "旧用户", "message_type": None},
        )(),
        type(
            "Message",
            (),
            {"sender_id": "new-user", "sender_name": "新用户", "message_type": None},
        )(),
    ]
    stream = type(
        "Stream",
        (),
        {
            "bot_id": "bot",
            "context": type(
                "Context", (), {"unread_messages": [], "history_messages": history}
            )(),
        },
    )()

    assert resolve_caller_identity(stream) == ("new-user", "新用户")


def test_resolve_caller_identity_skips_bot_system_and_notice_messages() -> None:
    """身份解析跳过 bot、自身 system sender 和 NOTICE 消息。"""

    from src.app.plugin_system.types import MessageType

    def message(sender_id: str, message_type: Any = None) -> Any:
        return type(
            "Message",
            (),
            {"sender_id": sender_id, "sender_name": "", "message_type": message_type},
        )()

    stream = type(
        "Stream",
        (),
        {
            "bot_id": "bot",
            "context": type(
                "Context",
                (),
                {
                    "unread_messages": [],
                    "history_messages": [
                        message("user-id"),
                        message("bot"),
                        message("system", MessageType.NOTICE),
                        message("SYSTEM"),
                    ],
                },
            )(),
        },
    )()

    assert resolve_caller_identity(stream) == ("user-id", "user-id")


def test_resolve_caller_identity_returns_empty_without_counterpart() -> None:
    """找不到有效对方消息时返回空身份。"""

    stream = type(
        "Stream",
        (),
        {
            "bot_id": "bot",
            "context": type(
                "Context",
                (),
                {
                    "unread_messages": [],
                    "history_messages": [
                        type(
                            "Message",
                            (),
                            {
                                "sender_id": "bot",
                                "sender_name": "",
                                "message_type": None,
                            },
                        )()
                    ],
                },
            )(),
        },
    )()

    assert resolve_caller_identity(stream) == ("", "")
