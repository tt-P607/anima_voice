"""say Action 的 local_asr 与平台私聊输出路由测试。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from plugins.anima_voice.actions import _tts_schema, say
from plugins.anima_voice.actions.say import SayAction
from plugins.anima_voice.config import AnimaVoiceConfig
from plugins.anima_voice.runtime import call_state
from plugins.anima_voice.speech import backend as backend_module
from plugins.anima_voice.speech import streaming


class _Player:
    """记录平台私聊的本机播放。"""

    def __init__(self) -> None:
        self.audio: list[bytes] = []

    async def play_audio(self, audio: bytes) -> None:
        """保存播放音频。"""

        self.audio.append(audio)


def _complete_audio_config() -> AnimaVoiceConfig:
    """选择完整音频合成路径。"""

    config = AnimaVoiceConfig()
    config.tts.streaming_enabled = False
    return config


async def _audio_artifact(*_: Any) -> Any:
    """返回直接合成服务的完整音频结果。"""

    return SimpleNamespace(audio=b"audio", mime_type="audio/wav", metadata={})


def _capabilities() -> Any:
    """返回含 speed 参数的 TTS 能力描述。"""

    return SimpleNamespace(
        speed_guide=SimpleNamespace(
            param_type="number", description="语速", default=1.0,
            valid_values=None, min_value=0.5, max_value=2.0, required=False,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["local_asr", "qq"])
async def test_say_routes_local_asr_to_voice_and_platform_to_player(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    """local_asr 不要求 active call 并发语音消息，平台流必须本机播放。"""

    async def send_voice(*args: Any, **kwargs: Any) -> bool:
        sent.append((args, kwargs))
        return True

    @asynccontextmanager
    async def no_heartbeat(_: str) -> Any:
        yield

    sent: list[Any] = []
    suppression: list[str] = []
    player = _Player()
    plugin = SimpleNamespace(
        audio_player=player,
        tts_capabilities=None,
        config=_complete_audio_config(),
    )
    stream = SimpleNamespace(stream_id="stream", platform=platform)
    action = SayAction(stream, plugin)
    monkeypatch.setattr(
        say, "synthesize_audio", _audio_artifact
    )
    monkeypatch.setattr(say.send_api, "send_voice", send_voice)
    monkeypatch.setattr(say, "keep_watchdog_alive", no_heartbeat)
    if platform == "qq":
        @asynccontextmanager
        async def suppress_microphone() -> Any:
            suppression.append("start")
            try:
                yield
            finally:
                suppression.append("stop")

        monkeypatch.setattr(
            say, "get_asr_output",
            lambda: SimpleNamespace(suppress_microphone_during_playback=suppress_microphone),
        )
        await call_state.set_active_call("stream")

    result = await action.execute("hello")

    assert result[0] is True
    if platform == "local_asr":
        assert len(sent) == 1
        assert sent[0][1]["platform"] == "local_asr"
        assert player.audio == []
        assert await call_state.get_active_call() is None
    else:
        assert sent == []
        assert player.audio == [b"audio"]
        assert suppression == ["start", "stop"]


@pytest.mark.asyncio
async def test_say_platform_route_fails_without_active_call_or_player(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """平台 chatter 不可用播放器或通话时不得报告成功。"""

    plugin = SimpleNamespace(
        audio_player=None, tts_capabilities=None, config=_complete_audio_config()
    )
    action = SayAction(SimpleNamespace(stream_id="stream", platform="qq"), plugin)
    monkeypatch.setattr(say, "synthesize_audio", _audio_artifact)

    result = await action.execute("hello")

    assert result == (False, "当前聊天流已不处于通话中")


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_audio", [False, True])
async def test_say_does_not_report_success_without_output(
    monkeypatch: pytest.MonkeyPatch, empty_audio: bool,
) -> None:
    """空合成和发送失败均不可报告语音已成功输出。"""

    async def synthesize_audio(*_: Any) -> Any:
        return SimpleNamespace(audio=None if empty_audio else b"audio")

    async def send_voice(*_: Any, **__: Any) -> bool:
        return False

    plugin = SimpleNamespace(
        audio_player=_Player(), tts_capabilities=None, config=_complete_audio_config()
    )
    action = SayAction(SimpleNamespace(stream_id="stream", platform="local_asr"), plugin)
    monkeypatch.setattr(say, "synthesize_audio", synthesize_audio)
    monkeypatch.setattr(say.send_api, "send_voice", send_voice)

    assert await action.execute("hello") == (False, "已输出 0/1 段语音")


@pytest.mark.asyncio
async def test_say_rejects_empty_content() -> None:
    """没有合成片段时返回明确失败结果。"""

    plugin = SimpleNamespace(audio_player=_Player(), tts_capabilities=None, config=AnimaVoiceConfig())
    action = SayAction(SimpleNamespace(stream_id="stream", platform="local_asr"), plugin)
    assert await action.execute("") == (False, "没有可播放的语音内容")


@pytest.mark.asyncio
async def test_call_boundary_speech_rejects_empty_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """接通和挂断的语音为空时必须反馈调用失败且不播放。"""

    from plugins.anima_voice.speech import backend as backend_module
    from plugins.anima_voice.voice_call.lifecycle import speak

    player = _Player()
    plugin = SimpleNamespace(
        audio_player=player, tts_capabilities=None, config=_complete_audio_config()
    )
    async def synthesize_audio(*_: Any) -> Any:
        return SimpleNamespace(audio=None)

    monkeypatch.setattr(backend_module, "synthesize_audio", synthesize_audio)
    with pytest.raises(RuntimeError, match="可播放音频"):
        await speak(plugin, "stream", "hello")
    assert player.audio == []
@pytest.mark.asyncio
async def test_parallel_synthesis_is_bounded_and_output_remains_ordered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """分段合成遵循并发上限，先完成的后段不会越过前段播放。"""

    active = 0
    peak = 0
    second_finished = asyncio.Event()
    finished: list[str] = []

    async def synthesize_audio(_: str, text: str, __: Any) -> Any:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if text == "first":
            await second_finished.wait()
        if text == "second":
            second_finished.set()
        finished.append(text)
        active -= 1
        return SimpleNamespace(audio=text.encode("ascii"))

    player = _Player()
    config = _complete_audio_config()
    config.tts.max_parallel_segments = 2
    plugin = SimpleNamespace(audio_player=player, tts_capabilities=None, config=config)
    action = SayAction(SimpleNamespace(stream_id="stream", platform="qq"), plugin)
    monkeypatch.setattr(say, "synthesize_audio", synthesize_audio)
    @asynccontextmanager
    async def suppress_microphone() -> Any:
        yield

    monkeypatch.setattr(
        say, "get_asr_output",
        lambda: SimpleNamespace(suppress_microphone_during_playback=suppress_microphone),
    )
    monkeypatch.setattr(say, "parse_speech_segments", lambda *_: [(text, 0) for text in ("first", "second", "third")])
    await call_state.set_active_call("stream")

    assert (await action.execute("ignored"))[0] is True
    assert peak == 2
    assert finished.index("second") < finished.index("first")
    assert player.audio == [b"first", b"second", b"third"]


@pytest.mark.asyncio
async def test_cancelled_say_joins_all_synthesis_tasks_before_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取消发言须取消并 join 全部预合成任务。"""

    started = asyncio.Event()
    blocked = asyncio.Event()
    active = 0
    async def synthesize_audio(*_: Any) -> Any:
        nonlocal active
        active += 1
        started.set()
        try:
            await blocked.wait()
        finally:
            active -= 1

    plugin = SimpleNamespace(audio_player=_Player(), tts_capabilities=None, config=_complete_audio_config())
    action = SayAction(SimpleNamespace(stream_id="stream", platform="local_asr"), plugin)
    monkeypatch.setattr(say, "synthesize_audio", synthesize_audio)
    monkeypatch.setattr(say, "parse_speech_segments", lambda *_: [("first", 0), ("second", 0)])
    execution = asyncio.get_running_loop().create_task(action.execute("ignored"))
    await asyncio.wait_for(started.wait(), timeout=2)
    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution
    assert active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("emit_text", [False, True])
@pytest.mark.parametrize("raises", [False, True])
async def test_text_on_failure_requires_explicit_config_and_preserves_failure(
    monkeypatch: pytest.MonkeyPatch, emit_text: bool, raises: bool,
) -> None:
    """文本失败行为只在明确开启时生效，不虚报语音播放成功。"""

    sent: list[str] = []

    async def synthesize_audio(*_: Any) -> Any:
        if raises:
            raise RuntimeError("TTS unavailable")
        return SimpleNamespace(audio=None)

    async def send_text(text: str, **_: Any) -> bool:
        sent.append(text)
        return True

    config = _complete_audio_config()
    config.tts.emit_text_on_tts_failure = emit_text
    plugin = SimpleNamespace(audio_player=_Player(), tts_capabilities=None, config=config)
    action = SayAction(SimpleNamespace(stream_id="stream", platform="local_asr"), plugin)
    monkeypatch.setattr(say, "synthesize_audio", synthesize_audio)
    monkeypatch.setattr(say.send_api, "send_text", send_text)
    assert (await action.execute("hello"))[0] is False
    assert sent == (["hello"] if emit_text else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["local_asr", "qq"])
async def test_shared_pcm_provider_starts_before_eof_and_records_once(
    monkeypatch: pytest.MonkeyPatch, platform: str,
) -> None:
    """共享 Provider 首块即播，只有本地 ASR 完播后记录一次 WAV。"""

    first_write = asyncio.Event()
    release_eof = asyncio.Event()
    calls: list[Any] = []
    writes: list[bytes] = []
    handles: list[Any] = []
    loop = asyncio.get_running_loop()

    class Provider:
        """产生首块后等待独立释放的共享提供方。"""

        @asynccontextmanager
        async def open_pcm_stream(self, request: Any) -> Any:
            calls.append((request.text, request.markers))

            async def chunks() -> Any:
                yield b"\x01\x00"
                await release_eof.wait()
                yield b"\x02\x00"

            try:
                yield SimpleNamespace(
                    sample_rate=48000, channels=1, sample_format="s16le", chunks=chunks(),
                )
            finally:
                calls.append("provider_closed")

    class Output(_Player):
        """消费 PCM，同时提供 ASR 的消息记录能力。"""

        @asynccontextmanager
        async def suppress_microphone_during_playback(self) -> Any:
            calls.append("microphone_muted")
            try:
                yield
            finally:
                calls.append("microphone_restored")

        async def play_pcm_stream(self, chunks: Any, **kwargs: Any) -> int:
            assert kwargs == {"sample_rate": 48000}
            async for chunk in chunks:
                writes.append(chunk)
                first_write.set()
            return sum(len(block) // 2 for block in writes)

        async def send_played_voice(self, audio: bytes, **kwargs: Any) -> bool:
            assert audio.startswith(b"RIFF")
            calls.append(("record", kwargs))
            return True

    def create(coro: Any, _: str) -> Any:
        handle = SimpleNamespace(task=loop.create_task(coro))
        handles.append(handle)
        return handle

    output = Output()
    config = AnimaVoiceConfig()
    view = say.require_plugin(SimpleNamespace(
        config=config, audio_player=output, tts_capabilities=None,
    ))
    monkeypatch.setattr(streaming, "get_tts_service", lambda: Provider())
    monkeypatch.setattr(streaming, "get_asr_output", lambda: output)
    monkeypatch.setattr(streaming, "create_task", create)
    execution = loop.create_task(streaming.play_streaming_speech(
        view=view, stream_id="stream", platform=platform,
        segments=[("hello", 0.0)], tts_params={"speed": 0.8},
    ))
    await asyncio.wait_for(first_write.wait(), 2)
    assert not execution.done()
    assert writes == [b"\x01\x00"]
    release_eof.set()
    assert await execution == 1
    assert writes == [b"\x01\x00", b"\x02\x00"]
    assert ("hello", {"speed": 0.8}) in calls
    if platform == "qq":
        assert calls.index("microphone_muted") < calls.index(("hello", {"speed": 0.8}))
    assert calls.count("provider_closed") == 1
    records = [item for item in calls if isinstance(item, tuple) and item[0] == "record"]
    assert len(records) == (1 if platform == "local_asr" else 0)
    assert calls.count("microphone_muted") == (0 if platform == "local_asr" else 1)
    assert calls.count("microphone_restored") == (0 if platform == "local_asr" else 1)
    assert all(handle.task.done() for handle in handles)
    assert not streaming._playbacks


@pytest.mark.asyncio
async def test_direct_service_preserves_request_without_registry_or_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """直接 Service 接收结构化参数，不查询 Registry 或创建 HTTP 连接。"""

    calls: list[Any] = []

    class Service:
        """只公开直接合成契约的语音服务。"""

        async def synthesize(self, request: Any) -> Any:
            calls.append(request)
            return SimpleNamespace(audio_base64="YXVkaW8=", mime_type="audio/wav")

        def open_pcm_stream(self, request: Any) -> Any:
            pytest.fail("完整音频不会打开 PCM")

    monkeypatch.setattr(backend_module.service_api, "get_service", lambda _: Service())
    artifact = await backend_module.synthesize_audio("stream", "hello", {"speed": 0.8})
    assert artifact.audio == b"audio"
    assert len(calls) == 1
    assert (calls[0].text, calls[0].markers, calls[0].options) == (
        "hello", {"speed": 0.8}, {},
    )


@pytest.mark.asyncio
async def test_say_execute_uses_shared_pcm_streaming_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """say action 的 streaming 配置通过共享 Service 播放并记录 WAV。"""

    calls: list[Any] = []

    class Service:
        """提供单段 PCM 的共享 TTS Service。"""

        @asynccontextmanager
        async def open_pcm_stream(self, request: Any) -> Any:
            calls.append(("request", request.text, request.markers))

            async def chunks() -> Any:
                yield b"\x01\x00\x02\x00"

            try:
                yield SimpleNamespace(
                    sample_rate=48000, channels=1, sample_format="s16le",
                    chunks=chunks(),
                )
            finally:
                calls.append("closed")

        async def synthesize(self, _: Any) -> Any:
            pytest.fail("streaming action must request PCM")

    class Output:
        """记录 local_asr PCM 播放与历史消息入库。"""

        async def play_pcm_stream(self, chunks: Any, **kwargs: Any) -> int:
            calls.append(("play", kwargs))
            frames = 0
            async for chunk in chunks:
                frames += len(chunk) // 2
            return frames

        async def send_played_voice(self, audio: bytes, **kwargs: Any) -> bool:
            assert audio.startswith(b"RIFF")
            calls.append(("record", kwargs))
            return True

    config = AnimaVoiceConfig()
    config.tts.streaming_enabled = True
    output = Output()
    plugin = SimpleNamespace(config=config, audio_player=output, tts_capabilities=None)
    action = SayAction(SimpleNamespace(stream_id="stream", platform="local_asr"), plugin)
    monkeypatch.setattr(streaming, "get_tts_service", lambda: Service())
    monkeypatch.setattr(streaming, "get_asr_output", lambda: output)
    monkeypatch.setattr(
        backend_module.service_api, "get_service",
        lambda _: pytest.fail("PCM branch must not perform another service lookup"),
    )

    result = await action.execute("hello", speed=0.8)

    assert result == (True, "已输出 1/1 段语音")
    assert ("request", "hello", {"speed": 0.8}) in calls
    assert ("play", {"sample_rate": 48000}) in calls
    recorded = [item for item in calls if isinstance(item, tuple) and item[0] == "record"]
    assert len(recorded) == 1
    assert recorded[0][1] == {"stream_id": "stream", "text": "hello"}
    assert calls.count("closed") == 1


def test_say_default_schema_reads_public_service_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认 say schema 从公开 TTS service 获取能力，不访问 Registry。"""

    calls: list[str] = []

    class Service:
        def get_capabilities(self) -> Any:
            calls.append("capabilities")
            return _capabilities()

    monkeypatch.setattr(_tts_schema, "get_service", lambda _: Service())
    monkeypatch.setattr(
        say, "registry",
        lambda: pytest.fail("schema must not query a component Registry"),
        raising=False,
    )

    schema = SayAction.to_schema()

    assert calls == ["capabilities"]
    assert schema["function"]["parameters"]["properties"]["speed"] == {
        "type": "number", "description": "语速", "default": 1.0,
        "minimum": 0.5, "maximum": 2.0,
    }
