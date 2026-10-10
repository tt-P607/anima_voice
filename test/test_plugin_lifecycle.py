"""anima_voice 插件 Provider 装配及卸载资源生命周期测试。"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from plugins.anima_voice import plugin as plugin_module
from plugins.anima_voice.asr import VoiceMicrophoneAdapter
from plugins.anima_voice.audio import player as player_module
from plugins.anima_voice.audio.player import AudioPlayer
from plugins.anima_voice.config import AnimaVoiceConfig
from plugins.anima_voice.plugin import AnimaVoicePlugin


@pytest.mark.asyncio
async def test_plugin_load_reads_tts_service_capabilities_without_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """插件装配只读取公开 TTS service 的参数描述。"""

    calls: list[str] = []

    class Service:
        def get_capabilities(self) -> str:
            calls.append("capabilities")
            return "caps"

        async def synthesize(self, *_: Any) -> None:
            pytest.fail("plugin load must not synthesize")

    class Player:
        def __init__(self, *_: Any) -> None:
            calls.append("player")

    plugin = object.__new__(AnimaVoicePlugin)
    plugin.config = AnimaVoiceConfig()
    plugin.audio_player = None
    plugin.tts_capabilities = None
    monkeypatch.setattr(plugin_module, "AudioPlayer", Player)
    monkeypatch.setattr(plugin_module.service_api, "get_service", lambda _: Service())

    await plugin.on_plugin_loaded()

    assert calls == ["player", "capabilities"]
    assert plugin.tts_capabilities == "caps"
    assert plugin.get_components()[0] is VoiceMicrophoneAdapter


@pytest.mark.asyncio
async def test_unload_finalizes_owned_call_and_releases_plugin_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """卸载时统一终结活动通话，并释放播放器与能力引用。"""

    calls: list[tuple[str, str]] = []

    class Player:
        """记录独立播放器关闭。"""

        async def close(self) -> None:
            """记录资源关闭。"""

            calls.append(("player", "close"))

    async def finalize(**kwargs: Any) -> tuple[bool, str]:
        calls.append((kwargs["stream_id"], kwargs["end_reason"]))
        return True, "ended"

    plugin = object.__new__(AnimaVoicePlugin)
    plugin.audio_player = Player()
    plugin.tts_capabilities = object()
    monkeypatch.setattr(plugin_module.call_state, "get_active_call", _active_call)
    monkeypatch.setattr(plugin_module, "finalize_call", finalize)
    monkeypatch.setattr(
        plugin_module.service_api, "get_service",
        lambda _: pytest.fail("unload must not access shared TTS service"),
    )

    await plugin.on_plugin_unloaded()

    assert calls == [("stream", "plugin_unload"), ("player", "close")]
    assert plugin.audio_player is None
    assert plugin.tts_capabilities is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_playback", [False, True])
async def test_player_close_and_cancel_join_owned_worker(
    monkeypatch: pytest.MonkeyPatch, cancel_playback: bool,
) -> None:
    """关闭或取消播放须等待自有线程退出，不调用全局音频停止。"""

    player = AudioPlayer()
    started = threading.Event()
    stopped = threading.Event()
    handles: list[Any] = []
    loop = asyncio.get_running_loop()

    def create(coro: Any, _: str) -> Any:
        handle = SimpleNamespace(task=loop.create_task(coro))
        handles.append(handle)
        return handle

    def play(_: bytes) -> None:
        started.set()
        player._stop_event.wait(timeout=2)
        stopped.set()

    monkeypatch.setattr(player_module, "create_task", create)
    monkeypatch.setattr(player, "_play_sync", play)
    monkeypatch.setattr(player_module.sd, "stop", lambda: pytest.fail("global stop"))
    playback = loop.create_task(player.play_audio(b"audio"))
    assert await asyncio.to_thread(started.wait, 2)
    if cancel_playback:
        playback.cancel()
        with pytest.raises(asyncio.CancelledError):
            await playback
    await player.close()
    if not cancel_playback:
        await playback
    assert stopped.is_set()
    assert all(handle.task.done() for handle in handles)
    with pytest.raises(RuntimeError, match="已关闭"):
        await player.play_audio(b"audio")


async def _active_call() -> Any:
    """返回卸载路径使用的最小通话记录。"""

    return type("ActiveCall", (), {"caller_stream_id": "stream"})()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_playback", [False, True, "close"])
async def test_pcm_first_block_precedes_eof_and_releases_owned_stream(
    monkeypatch: pytest.MonkeyPatch, cancel_playback: bool | str,
) -> None:
    """首块立即写入独占流，取消和完成均释放线程与 PCM 迭代器。"""

    loop = asyncio.get_running_loop()
    written = asyncio.Event()
    finish = asyncio.Event()
    closed: list[str] = []
    blocks: list[bytes] = []
    handles: list[Any] = []

    class Output:
        """记录写入而不访问声卡的输出流。"""

        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["samplerate"] == 48000
            assert kwargs["channels"] == 1
            assert kwargs["dtype"] == "int16"

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_: object) -> None:
            closed.append("output")

        def write(self, samples: Any) -> None:
            blocks.append(samples.tobytes())
            loop.call_soon_threadsafe(written.set)

        def abort(self) -> None:
            closed.append("abort")

    def create(coro: Any, _: str) -> Any:
        handle = SimpleNamespace(task=loop.create_task(coro))
        handles.append(handle)
        return handle

    async def chunks() -> Any:
        try:
            yield b"\x01"
            yield b"\x00\x02\x00"
            await finish.wait()
            yield b"\x03\x00"
        finally:
            closed.append("chunks")

    monkeypatch.setattr(player_module.sd, "OutputStream", Output)
    monkeypatch.setattr(player_module.sd, "stop", lambda: pytest.fail("global stop"))
    monkeypatch.setattr(player_module, "create_task", create)
    player = AudioPlayer()
    execution = loop.create_task(player.play_pcm_stream(chunks(), sample_rate=48000))
    await asyncio.wait_for(written.wait(), timeout=2)
    assert not execution.done()
    assert blocks == [b"\x01\x00\x02\x00"]
    if cancel_playback:
        if cancel_playback == "close":
            await player.close()
        else:
            execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
    else:
        finish.set()
        assert await execution == 3
        assert b"".join(blocks) == b"\x01\x00\x02\x00\x03\x00"
    assert "chunks" in closed and "output" in closed
    assert all(handle.task.done() for handle in handles)
    await player.close()


@pytest.mark.asyncio
async def test_pcm_device_failure_cancels_blocked_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设备无法打开时终结等待下一块的输入，不等待网络 EOF。"""

    closed: list[bool] = []
    loop = asyncio.get_running_loop()

    def fail_output(**_: Any) -> None:
        raise OSError("device unavailable")

    async def chunks() -> Any:
        try:
            yield b"\x01\x00"
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    monkeypatch.setattr(player_module.sd, "OutputStream", fail_output)
    monkeypatch.setattr(
        player_module, "create_task",
        lambda coro, _: SimpleNamespace(task=loop.create_task(coro)),
    )
    player = AudioPlayer()
    with pytest.raises(RuntimeError, match="PCM 输出失败"):
        await asyncio.wait_for(player.play_pcm_stream(chunks(), sample_rate=48000), 2)
    assert closed == [True]
    await player.close()
