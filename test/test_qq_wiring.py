"""原生 QQ 通话组件装配与生命周期测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from plugins.anima_voice import plugin as plugin_module
from plugins.anima_voice.actions import (
    AcceptVoiceCallAction,
    EndVoiceCallAction,
    RejectVoiceCallAction,
    StartVoiceCallAction,
)
from plugins.anima_voice.config import AnimaVoiceConfig
from plugins.anima_voice.plugin import AnimaVoicePlugin
from plugins.anima_voice.voice_call.events import QQCallNoticeHandler
from plugins.anima_voice.voice_call.service import VoiceCallService
from src.app.plugin_system.api.event_api import EventDecision


def _action(action_type: type, plugin: Any, stream: Any) -> Any:
    """构建绑定测试流的动作实例。"""
    action = object.__new__(action_type)
    action.plugin = plugin
    action.chat_stream = stream
    return action


@pytest.mark.asyncio
async def test_plugin_registers_native_qq_components_and_initializes_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """插件装配 Service/handler 与控制动作，加载不发起通话。"""
    calls: list[str] = []

    class Runtime:
        def __init__(self, config: AnimaVoiceConfig) -> None:
            calls.append("runtime")
            self.config = config

    class Player:
        def __init__(self, *_: Any) -> None:
            calls.append("player")

    service = SimpleNamespace(get_capabilities=lambda: "caps")
    plugin = object.__new__(AnimaVoicePlugin)
    plugin.config = AnimaVoiceConfig()
    plugin.audio_player = None
    plugin.tts_capabilities = None
    plugin.qq_calls = None
    monkeypatch.setattr(plugin_module, "QQCallRuntime", Runtime)
    monkeypatch.setattr(plugin_module, "AudioPlayer", Player)
    monkeypatch.setattr(plugin_module.service_api, "get_service", lambda _: service)

    await plugin.on_plugin_loaded()

    components = plugin.get_components()
    assert VoiceCallService in components
    assert QQCallNoticeHandler in components
    assert {StartVoiceCallAction, AcceptVoiceCallAction, RejectVoiceCallAction, EndVoiceCallAction} <= set(components)
    assert calls == ["runtime", "player"]
    assert plugin.qq_calls.config is plugin.config


def test_service_instances_share_runtime_owned_by_plugin() -> None:
    """新建 Service facade 仍读取同一 plugin.qq_calls 状态。"""
    runtime = SimpleNamespace(get_state=lambda stream_id: {"phase": stream_id})
    plugin = SimpleNamespace(qq_calls=runtime)
    first = VoiceCallService(plugin)
    second = VoiceCallService(plugin)

    assert first.get_state("connected") == second.get_state("connected")


@pytest.mark.asyncio
async def test_notice_handler_forwards_only_voice_call_without_mutating_params() -> None:
    """仅转发 voice_call notice，并原样返回事件参数键和值。"""
    runtime = SimpleNamespace(enqueue_notice=Mock())
    handler = QQCallNoticeHandler(SimpleNamespace(qq_calls=runtime))
    params = {
        "adapter_signature": "snowluma_adapter:adapter:snowluma_adapter",
        "notice": {"notice_type": "voice_call", "state": {"call": {"phase": "connected"}}},
    }
    original_keys = set(params)
    original_notice = params["notice"]

    decision, returned = await handler.execute("snowluma_adapter:notice_received", params)

    assert decision == EventDecision.SUCCESS
    assert returned is params
    assert set(params) == original_keys
    assert params["notice"] is original_notice
    runtime.enqueue_notice.assert_called_once_with(params["adapter_signature"], original_notice)

    ignored = {"adapter_signature": "adapter", "notice": {"notice_type": "group_increase"}}
    decision, returned = await handler.execute("snowluma_adapter:notice_received", ignored)
    assert decision == EventDecision.PASS
    assert returned is ignored
    runtime.enqueue_notice.assert_called_once()


@pytest.mark.asyncio
async def test_voice_actions_are_kfc_only_and_unknown_call_does_not_enable_redial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通话动作只面向 KFC 私聊，既有未知状态阻止重拨。"""
    stream = SimpleNamespace(stream_id="stream", platform="qq", chat_type="private")
    plugin = SimpleNamespace(qq_calls=SimpleNamespace(get_state=lambda _: {"phase": "unknown"}))
    monkeypatch.setattr(
        "plugins.anima_voice.actions.voice_call.chat_api.get_chatter_by_stream",
        lambda _: SimpleNamespace(name="kokoro_flow_chatter"),
    )

    start = _action(StartVoiceCallAction, plugin, stream)
    assert await start.go_activate() is True
    assert StartVoiceCallAction.chatter_allow == ["kokoro_flow_chatter"]
    assert AcceptVoiceCallAction.chatter_allow == ["kokoro_flow_chatter"]
    assert RejectVoiceCallAction.chatter_allow == ["kokoro_flow_chatter"]
    assert EndVoiceCallAction.chatter_allow == ["kokoro_flow_chatter"]

    monkeypatch.setattr(
        "plugins.anima_voice.actions.voice_call.chat_api.get_chatter_by_stream",
        lambda _: SimpleNamespace(name="anima_voice"),
    )
    assert await _action(StartVoiceCallAction, plugin, stream).go_activate() is False


@pytest.mark.asyncio
async def test_end_action_hangs_up_after_speech_failure_but_not_unknown_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """告别语失败仍请求挂断，unknown 状态不发控制请求。"""
    stream = SimpleNamespace(stream_id="stream", platform="qq", chat_type="private")
    monkeypatch.setattr(
        "plugins.anima_voice.actions.voice_call.chat_api.get_chatter_by_stream",
        lambda _: SimpleNamespace(name="kokoro_flow_chatter"),
    )
    runtime = SimpleNamespace(
        get_state=lambda _: {"phase": "connected", "call_id": "call"},
        speak=AsyncMock(side_effect=RuntimeError("TTS failed")),
        hangup=AsyncMock(return_value={"requested": True}),
    )
    action = _action(EndVoiceCallAction, SimpleNamespace(qq_calls=runtime), stream)

    ok, result = await action.execute(farewell="goodbye")

    assert ok is False
    assert "告别语播放失败" in result and "已请求挂断" in result
    runtime.speak.assert_awaited_once_with("stream", "goodbye")
    runtime.hangup.assert_awaited_once_with("stream")

    runtime.get_state = lambda _: {"phase": "unknown", "call_id": "call"}
    runtime.hangup.reset_mock()
    assert await action.go_activate() is True
    ok, result = await action.execute(farewell="goodbye")
    assert ok is False and "只允许查询状态" in result
    runtime.hangup.assert_not_awaited()


@pytest.mark.asyncio
async def test_unload_attempts_legacy_cleanup_when_qq_runtime_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QQ runtime 关闭失败也不跳过旧本机播放、麦克风和播放器清理。"""
    calls: list[str] = []

    class Runtime:
        async def close(self) -> None:
            calls.append("qq")
            raise RuntimeError("qq close failed")

    class Player:
        async def close(self) -> None:
            calls.append("player")

    async def stop_playback() -> None:
        calls.append("legacy_playback")

    async def stop_adapter(_: str) -> bool:
        calls.append("microphone")
        return True

    plugin = object.__new__(AnimaVoicePlugin)
    plugin.qq_calls = Runtime()
    plugin.audio_player = Player()
    plugin.tts_capabilities = object()
    monkeypatch.setattr(plugin_module, "stop_stream_playback", stop_playback)
    monkeypatch.setattr(plugin_module.call_state, "get_active_call", AsyncMock(return_value=None))
    monkeypatch.setattr(plugin_module.adapter_api, "get_adapter", lambda _: object())
    monkeypatch.setattr(plugin_module.adapter_api, "stop_adapter", stop_adapter)

    with pytest.raises(BaseExceptionGroup, match="资源清理失败"):
        await plugin.on_plugin_unloaded()

    assert calls == ["qq", "legacy_playback", "microphone", "player"]
    assert plugin.qq_calls is None
    assert plugin.audio_player is None
    assert plugin.tts_capabilities is None