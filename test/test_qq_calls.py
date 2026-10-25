"""原 QQ 私聊中的通话会话与双输入模式测试。"""

from __future__ import annotations

import asyncio
import base64
import io
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from ..config import AnimaVoiceConfig
from ..voice_call import qq_calls
from ..voice_call.qq_calls import QQCallRuntime
from .test_qq_audio import make_pcm


@pytest.fixture
def stream() -> Any:
    """提供已有 QQ 私聊及其用户消息。"""
    message = SimpleNamespace(sender_id="10001", sender_name="user", message_type="text")
    return SimpleNamespace(
        stream_id="example-stream", platform="qq", chat_type="private", bot_id="10000",
        context=SimpleNamespace(history_messages=[message], unread_messages=[], current_message=message),
    )


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch, stream: Any) -> Any:
    """隔离外部传输，同时保留实际的通话状态管理。"""
    service = SimpleNamespace(
        get_peer=AsyncMock(return_value={"user_id": 10001, "uid": "peer-uid", "nickname": "user"}),
        dial_call=AsyncMock(return_value={"requestId": "request-a", "targetUid": "peer-uid", "call": {"phase": "ended", "callId": "old", "generation": 1}}),
        get_call=AsyncMock(return_value={"phase": "idle"}),
        accept_call=AsyncMock(return_value={"requested": True}),
        reject_call=AsyncMock(return_value={"requested": True}),
        hangup_call=AsyncMock(return_value={"requested": True}),
    )
    monkeypatch.setattr(qq_calls, "get_service", lambda signature: service)
    monkeypatch.setattr(qq_calls.chat_api, "get_chatter_by_stream", lambda stream_id: SimpleNamespace(name="kokoro_flow_chatter"))
    monkeypatch.setattr(qq_calls.chat_api, "get_active_chatters", lambda: {stream.stream_id: SimpleNamespace(name="kokoro_flow_chatter")})
    monkeypatch.setattr(qq_calls.stream_api, "get_stream", AsyncMock(return_value=stream))
    monkeypatch.setattr(qq_calls.stream_api, "add_message_to_stream", AsyncMock())
    monkeypatch.setattr(qq_calls.event_api, "publish_event", AsyncMock())
    return service


def notice(phase: str, *, request_id: str = "request-a") -> dict[str, Any]:
    """构造同一通话的官方状态通知。"""
    return {"self_id": 10000, "notice_type": "voice_call", "state": {"call": {
        "callId": "call-a", "generation": 2, "phase": phase, "kind": "c2c", "media": "audio",
        "requestId": request_id, "peerUid": "peer-uid", "callerUid": "peer-uid", "direction": "outgoing",
    }}}


@pytest.mark.asyncio
async def test_dial_is_pending_not_connected(transport: Any, stream: Any) -> None:
    """拨号回执不得沿用上一通电话，也不得触发音频采集。"""
    runtime = QQCallRuntime(AnimaVoiceConfig())
    try:
        await runtime.dial(stream)
        state = runtime.get_state(stream.stream_id)
        assert state is not None
        assert state["phase"] == "dialing"
        assert state["call_id"] == ""
        assert state["input_mode"] == "asr"
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["asr", "multimodal"])
async def test_complete_utterance_uses_selected_input_mode(
    transport: Any, stream: Any, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """ASR 默认提交文字，显式多模态提交保真 WAV 且不调用 ASR。"""
    config = AnimaVoiceConfig()
    config.voice_call.input_mode = mode
    runtime = QQCallRuntime(config)
    monkeypatch.setattr(runtime, "_start_capture", lambda call: None)
    recognize = AsyncMock(return_value="example speech")
    monkeypatch.setattr(runtime, "_recognize", recognize)
    try:
        await runtime.dial(stream)
        await runtime.process_notice("snowluma_adapter:adapter:snowluma_adapter", notice("connected"))
        call = runtime.calls[stream.stream_id]
        pcm = make_pcm(200, 4096)
        await runtime.submit_pcm(call, pcm)
        event_name, payload = qq_calls.event_api.publish_event.call_args.args
        assert event_name == "anima_voice:call_input"
        assert payload["stream_id"] == stream.stream_id
        assert payload["mode"] == mode
        if mode == "asr":
            recognize.assert_awaited_once_with(pcm)
            assert payload["text"] == "example speech"
            assert payload["audio_base64"] == ""
        else:
            recognize.assert_not_awaited()
            assert payload["text"] == ""
            with wave.open(io.BytesIO(base64.b64decode(payload["audio_base64"])), "rb") as audio:
                assert audio.getframerate() == 16000
                assert audio.readframes(audio.getnframes()) == pcm
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_hangup_waits_for_observed_end(
    transport: Any, stream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """请求挂断成功仍保留 ending 状态，直到实际结束通知。"""
    runtime = QQCallRuntime(AnimaVoiceConfig())
    monkeypatch.setattr(runtime, "_start_capture", lambda call: None)
    try:
        await runtime.dial(stream)
        await runtime.process_notice("snowluma_adapter:adapter:snowluma_adapter", notice("connected"))
        await runtime.hangup(stream.stream_id)
        assert runtime.get_state(stream.stream_id)["phase"] == "ending"
        transport.hangup_call.assert_awaited_once()
        await runtime.process_notice("snowluma_adapter:adapter:snowluma_adapter", notice("ended"))
        assert runtime.get_state(stream.stream_id) is None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_unknown_incoming_identity_is_not_answered(transport: Any) -> None:
    """无法验证 UID 的来电不能接入其他用户会话或自动接听。"""
    transport.get_peer.return_value = {"user_id": 10001, "uid": "different-uid"}
    runtime = QQCallRuntime(AnimaVoiceConfig())
    incoming = notice("ringing", request_id="")
    incoming["state"]["call"]["direction"] = "incoming"
    try:
        await runtime.process_notice("snowluma_adapter:adapter:snowluma_adapter", incoming)
        assert runtime.calls == {}
        transport.accept_call.assert_not_awaited()
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_capture_failure_preserves_exception_and_received_audio_counts(
    transport: Any, stream: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """采音失败保留原始原因、堆栈和已消费计数，不记录音频正文。"""
    pcm = make_pcm(20, 0)
    failure = "SnowLuma stream response queue overflow"

    async def failed_capture(*args: Any) -> AsyncIterator[bytes]:
        """交付一块 PCM 后模拟接收层故障。"""
        yield pcm
        raise RuntimeError(failure)

    transport.capture_audio = failed_capture
    log_error = Mock()
    monkeypatch.setattr(qq_calls, "logger", SimpleNamespace(error=log_error, info=Mock()))
    runtime = QQCallRuntime(AnimaVoiceConfig())
    monkeypatch.setattr(runtime, "_start_capture", lambda call: None)
    try:
        await runtime.dial(stream)
        await runtime.process_notice("snowluma_adapter:adapter:snowluma_adapter", notice("connected"))
        call = runtime.calls[stream.stream_id]
        await runtime._capture(call)
        assert failure in call.audio_error
        log_error.assert_called_once()
        assert isinstance(log_error.call_args.kwargs["exc_info"], RuntimeError)
        assert str(log_error.call_args.kwargs["exc_info"]) == failure
        logged = log_error.call_args.args[0]
        assert failure in logged
        assert "call_id=call-a, generation=2" in logged
        assert f"chunks=1, bytes={len(pcm)}" in logged
        assert failure in qq_calls.stream_api.add_message_to_stream.call_args.args[0].content
        qq_calls.event_api.publish_event.assert_not_awaited()
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("amplitude", [0, 4096])
async def test_capture_state_distinguishes_silence_from_received_speech(
    transport: Any, stream: Any, monkeypatch: pytest.MonkeyPatch, amplitude: int
) -> None:
    """收包统计和电平能区分静音输入与经端点检测提交的语句。"""
    config = AnimaVoiceConfig()
    config.voice_call.input_mode = "multimodal"
    pcm = make_pcm(200, amplitude) + make_pcm(config.asr.endpoint_silence_ms + 40, 0)

    async def capture(*args: Any) -> AsyncIterator[bytes]:
        """提供固定电平的语句和结束静音。"""
        yield pcm

    transport.capture_audio = capture
    runtime = QQCallRuntime(config)
    monkeypatch.setattr(runtime, "_start_capture", lambda call: None)
    try:
        await runtime.dial(stream)
        await runtime.process_notice("snowluma_adapter:adapter:snowluma_adapter", notice("connected"))
        call = runtime.calls[stream.stream_id]
        await runtime._capture(call)
        state = runtime.get_state(stream.stream_id)
        assert state is not None
        assert state["capture_chunks"] == 1
        assert state["capture_bytes"] == len(pcm)
        assert state["capture_peak"] == amplitude / 32768.0
        if amplitude:
            assert state["capture_rms"] > 0
            utterance, stats = call.queue.get_nowait()
            await runtime.submit_pcm(call, utterance, stats=stats)
            assert runtime.get_state(stream.stream_id)["submitted_utterances"] == 1
        else:
            assert state["capture_rms"] == 0
            assert call.queue.empty()
            qq_calls.event_api.publish_event.assert_not_awaited()
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_style", ["", "test-style"])
async def test_speech_stream_goes_to_sl(
    transport: Any, stream: Any, monkeypatch: pytest.MonkeyPatch, voice_style: str
) -> None:
    """共享 TTS PCM 直接送到 SL，不经本机播放器。"""
    chunks = [b"\x01\x00" * 120, b"\x02\x00" * 120]

    async def audio_chunks() -> Any:
        """提供受控的完整 PCM。"""
        for chunk in chunks:
            yield chunk

    @asynccontextmanager
    async def open_pcm_stream(request: Any) -> Any:
        """保持共享 TTS 的上下文管理协议。"""
        assert request.text == "example reply"
        assert request.markers == {"style": voice_style or "test-style"}
        yield SimpleNamespace(sample_rate=48000, channels=1, sample_format="s16le", chunks=audio_chunks())

    received: list[bytes] = []

    async def play_audio(
        signature: str, call_id: str, generation: int, source: Any, *, trace_id: str = ""
    ) -> dict[str, Any]:
        """消费真实异步音频源并确认排空。"""
        assert trace_id
        async for chunk in source:
            received.append(chunk)
        return {"drained": True, "bytes": sum(map(len, received))}

    transport.play_audio = play_audio
    monkeypatch.setattr(qq_calls, "get_tts_service", lambda: SimpleNamespace(
        open_pcm_stream=open_pcm_stream,
        get_capabilities=lambda: SimpleNamespace(style_guide=SimpleNamespace(
            valid_values=["test-style"], default="test-style",
        )),
    ))
    runtime = QQCallRuntime(AnimaVoiceConfig())
    monkeypatch.setattr(runtime, "_start_capture", lambda call: None)
    try:
        await runtime.dial(stream)
        await runtime.process_notice("snowluma_adapter:adapter:snowluma_adapter", notice("connected"))
        result = await runtime.speak(
            stream.stream_id, "example reply", voice_style=voice_style
        )
        assert result["drained"] is True
        assert b"".join(received) == b"".join(chunks)
        with pytest.raises(ValueError, match="语音风格不可用"):
            await runtime.speak(stream.stream_id, "example reply", voice_style="missing")
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_playback", [False, True])
async def test_complete_speech_is_one_traced_request_and_streams_as_playback_consumes(
    transport: Any, stream: Any, monkeypatch: pytest.MonkeyPatch, cancel_playback: bool
) -> None:
    """完整多句正文一次请求，首块即交付，取消时关闭源且不伪报 EOF。"""
    requested: list[Any] = []
    closed: list[str] = []
    received: list[bytes] = []
    log = Mock()
    monkeypatch.setattr(qq_calls, "logger", SimpleNamespace(info=log))

    async def pcm_chunks() -> AsyncIterator[bytes]:
        """只在播放消费时交付 PCM。"""
        yield b"\x01\x00" * 160
        yield b"\x02\x00" * 160

    @asynccontextmanager
    async def open_pcm(request: Any) -> AsyncIterator[Any]:
        """记录每段正文长度和请求关联，不启动 GSV。"""
        requested.append(request)
        try:
            yield SimpleNamespace(sample_rate=48000, channels=1, sample_format="s16le", chunks=pcm_chunks())
        finally:
            closed.append(request.options["trace_id"])

    async def play(
        signature: str, call_id: str, generation: int, source: Any, *, trace_id: str = ""
    ) -> dict[str, Any]:
        """从实际 Anima 生成器消费，首块不能等待所有片段完成。"""
        assert trace_id == "reply-test"
        async for chunk in source:
            if not received:
                assert len(requested) == 1
            received.append(chunk)
            if cancel_playback:
                raise asyncio.CancelledError
        return {"drained": True, "bytes": sum(map(len, received))}

    transport.play_audio = play
    monkeypatch.setattr(qq_calls, "get_tts_service", lambda: SimpleNamespace(
        open_pcm_stream=open_pcm,
        get_capabilities=lambda: SimpleNamespace(style_guide=SimpleNamespace(
            valid_values=["test-style"], default="test-style",
        )),
    ))
    runtime = QQCallRuntime(AnimaVoiceConfig())
    monkeypatch.setattr(runtime, "_start_capture", lambda call: None)
    text = "第一句。第二句！\n第三句，第四句？"
    try:
        await runtime.dial(stream)
        await runtime.process_notice("snowluma_adapter:adapter:snowluma_adapter", notice("connected"))
        if cancel_playback:
            with pytest.raises(asyncio.CancelledError):
                await runtime.speak(stream.stream_id, text, trace_id="reply-test")
        else:
            await runtime.speak(stream.stream_id, text, trace_id="reply-test")
            assert b"".join(received) == b"\x01\x00" * 160 + b"\x02\x00" * 160
        assert len(requested) == 1
        assert requested[0].text == text
        assert requested[0].options["trace_id"] == "reply-test"
        assert len(closed) == len(requested)
        logged = "\n".join(call.args[0] for call in log.call_args_list)
        assert "trace_id=reply-test" in logged
        assert f"end_reason={'cancelled' if cancel_playback else 'drained'}" in logged
        assert text not in logged
        assert runtime.calls[stream.stream_id].playback is None
    finally:
        await runtime.close()