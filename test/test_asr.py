"""anima_voice 内置 ASR 配置与运行时测试。"""

from __future__ import annotations

import asyncio
import base64
import io
import threading
import wave
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from plugins.anima_voice.asr import (
    StreamingRecognizer,
    VoiceMicrophoneAdapter,
    is_likely_normal_utterance,
    is_obviously_invalid_asr_text,
)
from plugins.anima_voice.asr import runtime as asr_runtime
from plugins.anima_voice.config import AnimaVoiceConfig
from plugins.anima_voice.protocol import require_plugin
from plugins.anima_voice.speech import streaming
from src.app.plugin_system.base import PlatformSendResult, PlatformSendStatus


def test_config_sections_are_single_typed_source_for_asr_and_audio() -> None:
    """ASR、采音、身份和激活选项位于统一配置模型。"""

    config = AnimaVoiceConfig()

    assert config.asr.enabled is False
    assert config.audio.sample_rate == 16000
    assert config.audio.block_size == 8000
    assert config.bot.speaker_id == "local_microphone"
    assert config.activation.mode == "vad"
    assert not hasattr(config.asr, "provider")
    assert not hasattr(config.tts, "endpoint")
    assert not hasattr(config.tts, "circuit_failure_threshold")


def test_recognizer_does_not_load_model_when_created_or_buffering() -> None:
    """识别器构造和缓存音频不导入或初始化 FunASR。"""

    recognizer = StreamingRecognizer(AnimaVoiceConfig())

    assert recognizer.model_loaded is False
    recognizer.accept_waveform(np.zeros(160, dtype=np.float32))
    assert recognizer.has_audio()
    assert recognizer.model_loaded is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [("今天天气很好", True), ("!!!!!", False), ("啊", False)],
)
def test_text_quality_decisions(text: str, expected: bool) -> None:
    """识别文本过滤器接受常规中文并拦截明显无效文本。"""

    assert is_likely_normal_utterance(text).accepted is expected
    assert is_obviously_invalid_asr_text(text).accepted is (not expected)


class _FakeAudioSource:
    """不访问实际音频设备的测试音频源。"""

    def __init__(self, **_: Any) -> None:
        self.running = False

    async def start(self) -> None:
        self.running = True

    async def stop(self) -> None:
        self.running = False

    def is_running(self) -> bool:
        return self.running


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_pcm_pipeline_uses_real_owned_adapter_and_closes_only_its_request(
    monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    """真实内置 adapter 保护回采、预取下一段，并在取消时退出当前请求。"""

    first_write = asyncio.Event()
    next_request = asyncio.Event()
    release_output = asyncio.Event()
    release_provider = asyncio.Event()
    writes: list[bytes] = []
    closed: list[str] = []
    messages: list[Any] = []
    handles: list[asyncio.Task[Any]] = []

    class Service:
        """不持有关闭接口的共享语音服务。"""

        @asynccontextmanager
        async def open_pcm_stream(self, request: Any) -> Any:
            """第二段在首段设备播放结束之前开始。"""

            if request.text == "second":
                next_request.set()

            async def chunks() -> Any:
                yield b"\x01"
                yield b"\x00"
                if request.text == "second":
                    await release_provider.wait()

            try:
                yield SimpleNamespace(
                    sample_rate=48000, channels=1, sample_format="s16le", chunks=chunks(),
                )
            finally:
                closed.append(request.text)

    class Player:
        """以受控输出等待模拟本插件唯一设备流。"""

        async def play_pcm_stream(self, chunks: Any, **_: Any) -> int:
            async for chunk in chunks:
                assert adapter._external_playback_count == 1
                writes.append(chunk)
                if any(chunk) and not first_write.is_set():
                    first_write.set()
                    await release_output.wait()
            return sum(len(block) // 2 for block in writes)

    async def send_message(message: Any) -> PlatformSendResult:
        assert adapter._external_playback_count == 0
        assert message.message_id in adapter._played_voice_message_ids
        messages.append(message)
        await adapter._send_platform_message({
            "message_info": {"message_id": message.message_id}, "message_segment": [],
        })
        return PlatformSendResult(success=True)

    def create(coro: Any, _: str) -> Any:
        task = asyncio.get_running_loop().create_task(coro)
        handles.append(task)
        return SimpleNamespace(task=task)

    plugin = SimpleNamespace(config=AnimaVoiceConfig(), audio_player=Player(), tts_capabilities=None)
    adapter = VoiceMicrophoneAdapter(AsyncMock(), plugin)
    monkeypatch.setattr(streaming, "get_tts_service", lambda: Service())
    monkeypatch.setattr(streaming, "get_asr_output", lambda: adapter)
    monkeypatch.setattr(streaming, "create_task", create)
    monkeypatch.setattr("src.app.plugin_system.api.send_api.send_message_result", send_message)
    execution = asyncio.get_running_loop().create_task(streaming.play_streaming_speech(
        view=require_plugin(plugin), stream_id="owned-stream", platform="local_asr",
        segments=[("first", 0.01), ("second", 0.02)], tts_params={},
    ))
    try:
        await asyncio.wait_for(first_write.wait(), 2)
        await asyncio.wait_for(next_request.wait(), 2)
        assert not execution.done()
        assert adapter._microphone_suppressed()
        if cancel:
            await streaming.stop_stream_playback("owned-stream")
            with pytest.raises(asyncio.CancelledError):
                await execution
            assert not messages
        else:
            release_output.set()
            release_provider.set()
            assert await asyncio.wait_for(execution, 2) == 2
            assert len(messages) == 1
            encoded = messages[0].content["media"][0]["data"].removeprefix("base64|")
            with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as wav_file:
                assert wav_file.getframerate() == 48000
                assert wav_file.getnframes() == 480 + 960 + 2
            assert messages[0].processed_plain_text == "[语音:first second]"
        assert closed == ["first", "second"]
        assert adapter._external_playback_count == 0
        assert not adapter._played_voice_message_ids
        assert all(task.done() for task in handles)
        assert not streaming._playbacks
    finally:
        if not execution.done():
            execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)


class _FakeRecognizer:
    """不加载模型的测试识别器。"""

    def new_stream(self) -> None:
        return None


@pytest.mark.asyncio
async def test_disabled_adapter_stays_idle_until_voice_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """禁用常驻时不创建采集器，通话启动才创建本地运行时。"""

    source_created = False

    def make_source(**kwargs: Any) -> _FakeAudioSource:
        nonlocal source_created
        source_created = True
        return _FakeAudioSource(**kwargs)

    monkeypatch.setattr(asr_runtime, "MicrophoneAudioSource", make_source)
    monkeypatch.setattr(asr_runtime, "StreamingRecognizer", lambda _config: _FakeRecognizer())
    task = asyncio.create_task
    monkeypatch.setattr(
        asr_runtime._internal_compat,
        "create_task",
        lambda coro, name: SimpleNamespace(task=task(coro), task_id=name),
    )
    monkeypatch.setattr(
        asr_runtime._internal_compat,
        "cancel_task",
        lambda task_info: task_info.task.cancel(),
    )
    config = AnimaVoiceConfig()
    plugin = SimpleNamespace(config=config)
    adapter = VoiceMicrophoneAdapter(AsyncMock(), plugin)

    await adapter.on_adapter_loaded()
    assert source_created is False
    assert adapter.is_connected() is False

    started = await adapter.start_voice_call_session(
        "qq", "existing-stream", "real-user", "User", force_always_on=True
    )
    assert started is True
    assert adapter._activation_mode(config) == "always_on"
    assert adapter._redirect_target is not None
    assert adapter._redirect_target.user_id == "real-user"
    assert adapter._redirect_target.stream_id == "existing-stream"

    await adapter.end_voice_call_session()
    assert adapter._redirect_target is None
    assert adapter._activation_override is None
    assert adapter.is_connected() is False


@pytest.mark.asyncio
async def test_recognition_uses_redirect_identity_and_existing_stream() -> None:
    """通话识别提交真实用户身份并显式路由既有 stream。"""

    sink = SimpleNamespace(send=AsyncMock())
    adapter = VoiceMicrophoneAdapter(sink, SimpleNamespace(config=AnimaVoiceConfig()))
    adapter._config = adapter.plugin.config
    adapter._redirect_target = asr_runtime.RedirectTarget(
        "qq", "known-stream", "qq-user", "Caller", "group-1"
    )

    await adapter._send_text_to_core("测试语音", is_final=True)

    envelope = sink.send.await_args.args[0]
    info = envelope["message_info"]
    assert info["platform"] == "qq"
    assert info["stream_id"] == "known-stream"
    assert info["user_info"] == {
        "platform": "qq",
        "user_id": "qq-user",
        "user_nickname": "Caller",
        "group_id": "group-1",
    }
    assert envelope["message_segment"] == [{"type": "text", "data": "测试语音"}]


@pytest.mark.asyncio
async def test_suppression_tracks_nested_playback_and_tail_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """播放保护计数支持嵌套，退出后保留 300ms 尾窗。"""

    now = 100.0
    monkeypatch.setattr(asr_runtime.time, "monotonic", lambda: now)
    adapter = VoiceMicrophoneAdapter(AsyncMock(), SimpleNamespace(config=AnimaVoiceConfig()))

    async with adapter.suppress_microphone_during_playback():
        assert adapter._external_playback_count == 1
        assert adapter._microphone_suppressed()
        async with adapter.suppress_microphone_during_playback():
            assert adapter._external_playback_count == 2
        assert adapter._external_playback_count == 1

    assert adapter._external_playback_count == 0
    assert adapter._playback_until == pytest.approx(100.3)


@pytest.mark.asyncio
async def test_send_played_voice_uses_public_message_api_and_is_not_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主动播放后通过公开 Message API 入库，并由 adapter message ID 去重。"""

    sent_messages: list[Any] = []

    async def fake_send_message(message: Any) -> PlatformSendResult:
        sent_messages.append(message)
        assert message.message_id in adapter._played_voice_message_ids
        await adapter._send_platform_message(
            {"message_info": {"message_id": message.message_id}, "message_segment": []}
        )
        return PlatformSendResult(success=True)

    monkeypatch.setattr(
        "src.app.plugin_system.api.send_api.send_message_result", fake_send_message
    )
    adapter = VoiceMicrophoneAdapter(
        AsyncMock(), SimpleNamespace(config=AnimaVoiceConfig())
    )

    result = await adapter.send_played_voice(
        b"audio", stream_id="stream", text="你好"
    )
    assert result.status is PlatformSendStatus.SENT
    assert result.success is True
    message = sent_messages[0]
    assert message.message_type.value == "voice"
    assert message.content["media"][0]["data"] == "base64|YXVkaW8="
    assert "voice_id" not in message.content["media"][0]
    assert message.platform == "local_asr"
    assert message.message_id not in adapter._played_voice_message_ids


@pytest.mark.asyncio
async def test_local_output_uses_plugin_owned_player_and_normalized_voice_data() -> None:
    """适配器使用同一播放器并识别框架带前缀的语音媒体。"""

    player = SimpleNamespace(play_audio=AsyncMock())
    plugin = SimpleNamespace(config=AnimaVoiceConfig(), audio_player=player)
    adapter = VoiceMicrophoneAdapter(AsyncMock(), plugin)
    assert adapter._get_audio_player() is player
    await adapter._send_platform_message({
        "message_info": {"message_id": "ordinary"},
        "message_segment": [{"type": "voice", "data": "base64|YXVkaW8="}],
    })
    player.play_audio.assert_awaited_once_with(b"audio")
    assert adapter._external_playback_count == 0


@pytest.mark.asyncio
async def test_record_failure_releases_played_message_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """发送被拦截或失败时不遗留已播放消息标记。"""

    monkeypatch.setattr(
        "src.app.plugin_system.api.send_api.send_message_result",
        AsyncMock(return_value=PlatformSendResult(success=False, error="example refusal")),
    )
    adapter = VoiceMicrophoneAdapter(AsyncMock(), SimpleNamespace(config=AnimaVoiceConfig()))
    result = await adapter.send_played_voice(b"audio", stream_id="stream", text="hello")
    assert result.success is False
    assert result.status is PlatformSendStatus.FAILED
    assert not adapter._played_voice_message_ids


@pytest.mark.asyncio
async def test_cancelled_device_call_waits_until_blocking_thread_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """设备操作取消时等待模拟阻塞线程退出后再传播取消。"""

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_device_call() -> None:
        started.set()
        release.wait()
        finished.set()

    task_factory = asyncio.create_task
    monkeypatch.setattr(
        asr_runtime._internal_compat,
        "create_task",
        lambda coro, name: SimpleNamespace(task=task_factory(coro), task_id=name),
    )
    operation = task_factory(
        asr_runtime.MicrophoneAudioSource._run_device_call(blocking_device_call)
    )
    assert await asyncio.to_thread(started.wait, 2)
    operation.cancel()
    await asyncio.sleep(0)
    assert operation.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert finished.is_set()