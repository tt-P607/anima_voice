"""保持 KFC 原会话的 QQ 私聊通话运行时。"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.app.plugin_system.api import chat_api, event_api, stream_api
from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.api.service_api import get_service
from src.app.plugin_system.types import Message, MessageType

from .._internal_compat import create_task
from ..asr.recognizer import StreamingRecognizer
from ..config import AnimaVoiceConfig
from ..speech.backend import TTSRequest, get_tts_service
from .identity import resolve_caller_identity
from .qq_audio import PCMUtteranceSegmenter, encode_wav

SL_SERVICE = "snowluma_adapter:service:voice_call"
SL_ADAPTER = "snowluma_adapter:adapter:snowluma_adapter"
CALL_INPUT_EVENT = "anima_voice:call_input"
KFC_NAME = "kokoro_flow_chatter"
logger = get_logger("anima_voice.qq_call")


@dataclass(slots=True)
class QQCall:
    """一个原私聊流拥有的通话及受管资源。"""

    stream_id: str
    adapter_signature: str
    user_id: str
    user_name: str
    bot_id: str
    peer_uid: str
    input_mode: str
    call_id: str = ""
    generation: int = 0
    request_id: str = ""
    phase: str = "dialing"
    last_activity: float = field(default_factory=time.monotonic)
    capture: Any = None
    consumer: Any = None
    monitor: Any = None
    playback: Any = None
    audio_started: bool = False
    audio_error: str = ""
    capture_chunks: int = 0
    capture_bytes: int = 0
    capture_peak: float = 0.0
    capture_square_sum: float = 0.0
    submitted_utterances: int = 0
    playback_bytes: int = 0
    queue: asyncio.Queue[tuple[bytes, dict[str, str | int]]] = field(default_factory=lambda: asyncio.Queue(maxsize=4))
    speech_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class QQCallRuntime:
    """经公开 Service 编排通话，不绑定或替换 Chatter。"""

    def __init__(self, config: AnimaVoiceConfig) -> None:
        """保存配置，不打开设备、连接或模型。"""
        self.config = config
        self.calls: dict[str, QQCall] = {}
        self._recognizer: StreamingRecognizer | None = None
        self._asr_lock = asyncio.Lock()
        self._dial_lock = asyncio.Lock()
        self._notices: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=32)
        self._notice_task: Any = None

    def enqueue_notice(self, signature: str, notice: dict[str, Any]) -> None:
        """排队处理状态，避免在 OneBot 收包回调中等待同连接回执。"""
        self._notices.put_nowait((signature, notice))
        if self._notice_task is None:
            self._notice_task = create_task(self._process_notices(), "anima_voice.qq.notices")

    async def _process_notices(self) -> None:
        """串行关联官方状态，资料查询失败明确记录。"""
        while True:
            signature, notice = await self._notices.get()
            try:
                await self.process_notice(signature, notice)
            except (RuntimeError, OSError, ValueError, TypeError, LookupError) as error:
                logger.error(f"QQ 通话状态处理失败：{type(error).__name__}")
            finally:
                self._notices.task_done()

    @staticmethod
    def _transport() -> Any:
        """获取已有 SL 实例拥有的公开语音服务。"""
        service = get_service(SL_SERVICE)
        if service is None:
            raise RuntimeError("SnowLuma 语音 Service 未加载")
        return service

    @staticmethod
    def _require_kfc(stream: Any) -> tuple[str, str]:
        """确认是已有 KFC QQ 私聊并解析实际聊天对象。"""
        chatter = chat_api.get_chatter_by_stream(stream.stream_id)
        if stream.platform != "qq" or str(stream.chat_type) != "private" or getattr(chatter, "name", "") != KFC_NAME:
            raise ValueError("QQ 通话只支持当前 KFC 私聊")
        user_id, user_name = resolve_caller_identity(stream)
        if not user_id.isdigit():
            raise ValueError("无法从原私聊确定对方 QQ 号")
        return user_id, user_name

    def get_state(self, stream_id: str) -> dict[str, Any] | None:
        """返回该流当前通话的只读快照。"""
        call = self.calls.get(stream_id)
        if call is None:
            return None
        return {
            "stream_id": call.stream_id, "phase": call.phase,
            "call_id": call.call_id, "generation": call.generation,
            "request_id": call.request_id, "input_mode": call.input_mode,
            "user_id": call.user_id, "user_name": call.user_name, "bot_id": call.bot_id,
            "audio_error": call.audio_error,
            "capture_chunks": call.capture_chunks, "capture_bytes": call.capture_bytes,
            "capture_peak": call.capture_peak,
            "capture_rms": (call.capture_square_sum / (call.capture_bytes / 2)) ** 0.5
            if call.capture_bytes else 0.0,
            "submitted_utterances": call.submitted_utterances,
            "playback_bytes": call.playback_bytes,
        }

    async def dial(self, stream: Any) -> dict[str, Any]:
        """串行申请账号通话，防止并发创建两次拨号。"""
        async with self._dial_lock:
            return await self._dial(stream)

    async def _dial(self, stream: Any) -> dict[str, Any]:
        """显式发起呼叫；回执仅表示请求受理。"""
        user_id, user_name = self._require_kfc(stream)
        if stream.stream_id in self.calls:
            raise RuntimeError("当前流已有通话或结果未知的拨号请求")
        signature = SL_ADAPTER
        for message in reversed(list(stream.context.history_messages)):
            extra = getattr(message, "extra", {})
            if isinstance(extra, dict) and extra.get("adapter_signature"):
                signature = str(extra["adapter_signature"])
                break
        if any(call.adapter_signature == signature for call in self.calls.values()):
            raise RuntimeError("该 QQ 账号已有通话请求")
        peer = await self._transport().get_peer(signature, user_id)
        call = QQCall(
            stream.stream_id, signature, user_id, user_name, str(stream.bot_id or ""),
            str(peer.get("uid") or ""), self.config.voice_call.input_mode,
        )
        self.calls[stream.stream_id] = call
        try:
            result = await self._transport().dial_call(signature, user_id)
        except asyncio.CancelledError:
            call.phase = "unknown"
            self._start_monitor(call)
            raise
        except Exception as error:
            if getattr(error, "outcomeUnknown", False):
                call.phase = "unknown"
                call.request_id = str(getattr(error, "requestId", "") or "")
                self._start_monitor(call)
            else:
                self.calls.pop(stream.stream_id, None)
            raise
        call.request_id = str(result.get("requestId") or "")
        call.peer_uid = str(result.get("targetUid") or call.peer_uid)
        self._start_monitor(call)
        snapshot = result.get("call")
        if isinstance(snapshot, dict):
            await self.process_notice(signature, {"self_id": call.bot_id, "state": {"call": snapshot}})
        return self.get_state(stream.stream_id) or {"phase": "ended"}

    def _start_monitor(self, call: QQCall) -> None:
        """为当前请求建立状态恢复和静默超时任务。"""
        if call.monitor is None:
            call.monitor = create_task(self._monitor(call), f"anima_voice.qq.monitor.{call.stream_id}")

    async def _monitor(self, call: QQCall) -> None:
        """重连后查询官方状态，超时只请求一次挂断。"""
        while self.calls.get(call.stream_id) is call:
            await asyncio.sleep(2.0)
            try:
                snapshot = await self._transport().get_call(call.adapter_signature)
                await self.process_notice(call.adapter_signature, {"self_id": call.bot_id, "state": {"call": snapshot}})
                timeout = self.config.voice_call.timeout_seconds
                if timeout > 0 and time.monotonic() - call.last_activity >= timeout and call.call_id and call.phase not in {"ending", "ended", "unknown"}:
                    await self.hangup(call.stream_id)
            except asyncio.CancelledError:
                raise
            except (RuntimeError, OSError, ValueError, TypeError, LookupError) as error:
                call.audio_error = f"通话状态查询失败：{type(error).__name__}"
                call.phase = "unknown"
                await self._stop_audio(call)

    async def process_notice(self, signature: str, notice: dict[str, Any]) -> None:
        """按账号、请求和通话代次关联官方快照。"""
        snapshot = notice.get("state", {}).get("call")
        if not isinstance(snapshot, dict) or not snapshot.get("callId"):
            return
        call_id = str(snapshot["callId"])
        generation = snapshot.get("generation")
        if not isinstance(generation, int):
            return
        selected: QQCall | None = None
        for call in list(self.calls.values()):
            if call.adapter_signature != signature or call.bot_id != str(notice.get("self_id") or ""):
                continue
            if (
                call.call_id and (call.call_id, call.generation) == (call_id, generation)
                or not call.call_id and call.request_id and call.request_id == snapshot.get("requestId")
            ):
                selected = call
                break
            if not call.call_id and not call.request_id and call.peer_uid and snapshot.get("direction") == "outgoing" and snapshot.get("peerUid") == call.peer_uid and snapshot.get("phase") != "ended":
                selected = call
                break
        if selected is None and snapshot.get("direction") != "outgoing" and snapshot.get("phase") in {"ringing", "connecting", "connected"}:
            selected = await self._match_incoming(signature, notice, snapshot)
        if selected is None:
            return
        previous = selected.phase
        phase = str(snapshot.get("phase") or "unknown")
        ranks = {"ringing": 1, "connecting": 2, "connected": 3, "ending": 4, "ended": 5}
        if previous in ranks and phase in ranks and ranks[phase] < ranks[previous]:
            return
        selected.call_id, selected.generation = call_id, generation
        selected.phase = phase
        if selected.phase == "connected":
            if not selected.audio_started:
                selected.last_activity = time.monotonic()
                self._start_capture(selected)
            if previous != "connected":
                await self._notify(selected, "QQ 语音通话已接通；继续当前会话，回复将通过电话播放。")
        elif selected.phase == "ended":
            await self._stop_call(selected)
            self.calls.pop(selected.stream_id, None)
            await self._notify(selected, "QQ 语音通话已结束；继续当前文字会话。")
        elif selected.phase == "ringing" and previous != "ringing":
            await self._notify(selected, "收到 QQ 语音来电，尚未接听；可以选择接听或拒接。")
        elif selected.phase != "connected":
            await self._stop_audio(selected)

    async def _match_incoming(self, signature: str, notice: dict[str, Any], snapshot: dict[str, Any]) -> QQCall | None:
        """只把经用户资料验证 UID 的来电对应到已有 KFC 私聊。"""
        uid = str(snapshot.get("callerUid") or "")
        if not uid:
            return None
        for stream_id, chatter in chat_api.get_active_chatters().items():
            if getattr(chatter, "name", "") != KFC_NAME or stream_id in self.calls:
                continue
            stream = await stream_api.get_stream(stream_id)
            if stream is None or str(stream.bot_id or "") != str(notice.get("self_id") or ""):
                continue
            try:
                user_id, user_name = self._require_kfc(stream)
                peer = await self._transport().get_peer(signature, user_id)
            except (ValueError, RuntimeError, LookupError):
                continue
            if peer.get("uid") != uid:
                continue
            call = QQCall(stream_id, signature, user_id, user_name, str(stream.bot_id), uid, self.config.voice_call.input_mode)
            self.calls[stream_id] = call
            self._start_monitor(call)
            return call
        return None

    async def _notify(self, call: QQCall, text: str) -> None:
        """把通话边界作为原流通知，不复制或迁移对话历史。"""
        if await stream_api.get_stream(call.stream_id) is None:
            return
        message = Message(
            message_id=f"qq-call-{call.call_id}-{time.time_ns()}",
            content=f"[通话状态] {text}", processed_plain_text=f"[通话状态] {text}",
            message_type=MessageType.NOTICE, sender_id=call.user_id, sender_name=call.user_name,
            platform="qq", chat_type="private", stream_id=call.stream_id, time=time.time(),
        )
        await stream_api.add_message_to_stream(message)

    def _start_capture(self, call: QQCall) -> None:
        """开始独立收音和有界语句消费，不阻塞 OneBot 事件处理。"""
        call.audio_started = True
        call.capture = create_task(self._capture(call), f"anima_voice.qq.capture.{call.stream_id}")
        call.consumer = create_task(self._consume(call), f"anima_voice.qq.consume.{call.stream_id}")

    async def _capture(self, call: QQCall) -> None:
        """收音、分句；处理跟不上时明确停止，不丢帧后声称成功。"""
        config = self.config
        segmenter = PCMUtteranceSegmenter(
            config.asr.endpoint_rms_threshold, config.asr.endpoint_silence_ms,
            config.asr.max_audio_window_ms, config.activation.preroll_ms,
        )
        try:
            async for pcm in self._transport().capture_audio(call.adapter_signature, call.call_id, call.generation):
                if self.calls.get(call.stream_id) is not call or call.phase != "connected":
                    return
                call.capture_chunks += 1
                call.capture_bytes += len(pcm)
                if pcm:
                    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
                    call.capture_peak = max(call.capture_peak, float(np.max(np.abs(samples))))
                    call.capture_square_sum += float(np.dot(samples, samples))
                if call.capture_chunks == 1:
                    logger.info(
                        f"QQ 采音首块：bytes={len(pcm)}, peak={call.capture_peak:.6f}"
                    )
                utterances = segmenter.feed(pcm)
                for stats in segmenter.discarded_stats:
                    logger.info(
                        f"QQ 音频已丢弃：call_id={call.call_id}, generation={call.generation}, {stats}"
                    )
                for utterance, stats in zip(utterances, segmenter.completed_stats, strict=True):
                    logger.info(
                        f"QQ 音频分句：call_id={call.call_id}, generation={call.generation}, "
                        f"pcm_bytes={len(utterance)}, {stats}"
                    )
                    call.last_activity = time.monotonic()
                    call.queue.put_nowait((utterance, stats))
            tail = segmenter.finish()
            for stats in segmenter.discarded_stats:
                logger.info(f"QQ 音频已丢弃：call_id={call.call_id}, generation={call.generation}, {stats}")
            if tail:
                logger.info(
                    f"QQ 音频分句：call_id={call.call_id}, generation={call.generation}, "
                    f"pcm_bytes={len(tail)}, {segmenter.completed_stats[0]}"
                )
                call.queue.put_nowait((tail, segmenter.completed_stats[0]))
        except asyncio.CancelledError:
            raise
        except (RuntimeError, OSError, ValueError, TypeError, asyncio.QueueFull) as error:
            call.audio_error = f"采音失败：{type(error).__name__}: {error}"
            logger.error(
                f"{call.audio_error} (call_id={call.call_id}, generation={call.generation}, "
                f"chunks={call.capture_chunks}, bytes={call.capture_bytes})",
                exc_info=error,
            )
            await self._notify(call, f"{call.audio_error}，本通话不再提交音频；可挂断后重新发起。")

    async def _consume(self, call: QQCall) -> None:
        """按原始顺序处理有界语句队列。"""
        try:
            while self.calls.get(call.stream_id) is call:
                pcm, stats = await call.queue.get()
                await self.submit_pcm(call, pcm, stats=stats)
        except asyncio.CancelledError:
            raise
        except (RuntimeError, OSError, ValueError, TypeError) as error:
            call.audio_error = f"语音输入失败：{type(error).__name__}"
            logger.error(call.audio_error)
            if call.capture is not None:
                call.capture.task.cancel()
            await self._notify(call, call.audio_error)

    async def submit_pcm(
        self, call: QQCall, pcm: bytes, *, stats: dict[str, str | int] | None = None
    ) -> None:
        """向 KFC 提交 ASR 文字或完整 WAV，禁止隐式切换模式。"""
        if self.calls.get(call.stream_id) is not call or call.phase != "connected":
            return
        text = ""
        audio = ""
        if call.input_mode == "asr":
            text = (await self._recognize(pcm)).strip()
            if not text:
                return
        else:
            audio = base64.b64encode(encode_wav(pcm)).decode("ascii")
        if self.calls.get(call.stream_id) is not call or call.phase != "connected":
            return
        utterance_id = uuid.uuid4().hex
        sequence = call.submitted_utterances + 1
        await event_api.publish_event(CALL_INPUT_EVENT, {
            "stream_id": call.stream_id, "user_id": call.user_id, "user_name": call.user_name,
            "call_id": call.call_id, "generation": call.generation, "mode": call.input_mode,
            "text": text, "audio_base64": audio, "format": "wav",
            "utterance_id": utterance_id, "sequence": sequence,
            "audio_stats": dict(stats or {}),
        })
        call.submitted_utterances += 1
        logger.info(
            f"QQ 通话输入已提交：call_id={call.call_id}, generation={call.generation}, "
            f"utterance_id={utterance_id}, sequence={sequence}, mode={call.input_mode}, "
            f"utterances={call.submitted_utterances}, pcm_bytes={len(pcm)}, {stats or {}}"
        )

    async def _recognize(self, pcm: bytes) -> str:
        """在受管线程任务中复用本地 ASR，不打开麦克风。"""
        async with self._asr_lock:
            if self._recognizer is None:
                config = self.config.model_copy(deep=True)
                config.audio.sample_rate = 16000
                config.audio.channels = 1
                self._recognizer = StreamingRecognizer(config)
            recognizer = self._recognizer

            def recognize() -> str:
                """识别一段已通过静音分句的 PCM。"""
                recognizer.new_stream()
                recognizer.accept_waveform(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
                recognizer.mark_endpoint()
                return recognizer.get_text()

            worker = create_task(asyncio.to_thread(recognize), "anima_voice.qq.asr")
            try:
                return await asyncio.shield(worker.task)
            except asyncio.CancelledError:
                await worker.task
                raise

    async def accept(self, stream_id: str) -> dict[str, Any]:
        """显式接听已归属当前流的来电，等待后续接通状态。"""
        call = self.calls.get(stream_id)
        if call is None or call.phase != "ringing":
            raise ValueError("当前流没有可接听的来电")
        call.phase = "connecting"
        try:
            return await self._transport().accept_call(call.adapter_signature, call.call_id)
        except (RuntimeError, OSError, TimeoutError):
            call.phase = "unknown"
            raise

    async def reject(self, stream_id: str) -> dict[str, Any]:
        """拒接当前流尚未接听的来电。"""
        call = self.calls.get(stream_id)
        if call is None or call.phase != "ringing":
            raise ValueError("当前流没有可拒接的来电")
        call.phase = "ending"
        try:
            return await self._transport().reject_call(call.adapter_signature, call.call_id)
        except (RuntimeError, OSError, TimeoutError):
            call.phase = "unknown"
            raise

    async def hangup(self, stream_id: str) -> dict[str, Any]:
        """请求挂断，不把控制回执当成官方终态。"""
        call = self.calls.get(stream_id)
        if call is None or not call.call_id:
            raise ValueError("通话尚未绑定官方 callId，请先查询状态")
        call.phase = "ending"
        await self._stop_audio(call)
        try:
            return await self._transport().hangup_call(call.adapter_signature, call.call_id)
        except (RuntimeError, OSError, TimeoutError):
            call.phase = "unknown"
            raise

    def get_speech_options(self) -> dict[str, Any]:
        """直接使用共享 TTS 的动态风格能力。"""
        guide = get_tts_service().get_capabilities().style_guide
        return {"styles": list(guide.valid_values or []), "default_style": guide.default}

    async def speak(
        self, stream_id: str, text: str, *, voice_style: str = "", trace_id: str = ""
    ) -> dict[str, Any]:
        """通过共享 TTS 和 SL 发送回复，不使用本机扬声器。"""
        call = self.calls.get(stream_id)
        if call is None or call.phase != "connected":
            raise RuntimeError("QQ 通话未接通")
        options = self.get_speech_options()
        style = voice_style.strip() or str(options["default_style"] or "")
        if not style or style not in options["styles"]:
            raise ValueError("所选通话语音风格不可用")
        async with call.speech_lock:
            if self.calls.get(stream_id) is not call or call.phase != "connected":
                raise RuntimeError("QQ 通话已变化")
            playback_id = trace_id or uuid.uuid4().hex
            logger.info(
                f"QQ 播音开始：trace_id={playback_id}, call_id={call.call_id}, generation={call.generation}, "
                f"text_chars={len(text)}"
            )
            generated_bytes = 0

            async def chunks() -> AsyncGenerator[bytes, None]:
                """一次请求完整正文并立即交付收到的 PCM，不等待整段生成。"""
                nonlocal generated_bytes
                request = TTSRequest(
                    stream_id, text, markers={"style": style}, options={"trace_id": playback_id}
                )
                logger.info(
                    f"QQ TTS 请求：trace_id={playback_id}, call_id={call.call_id}, generation={call.generation}, "
                    f"text_chars={len(text)}"
                )
                end_reason = "consumer_closed"
                try:
                    async with get_tts_service().open_pcm_stream(request) as stream:
                        if (stream.sample_rate, stream.channels, stream.sample_format) != (48000, 1, "s16le"):
                            raise ValueError("TTS 必须输出 48 kHz 单声道 S16LE")
                        async for chunk in stream.chunks:
                            if call.phase != "connected":
                                raise RuntimeError("QQ 通话已离开接通状态")
                            generated_bytes += len(chunk)
                            yield chunk
                    end_reason = "eof"
                except asyncio.CancelledError:
                    end_reason = "cancelled"
                    raise
                except Exception:
                    end_reason = "error"
                    raise
                finally:
                    logger.info(
                        f"QQ TTS 结束：trace_id={playback_id}, text_chars={len(text)}, pcm_bytes={generated_bytes}, "
                        f"audio_seconds={generated_bytes / 96000:.3f}, end_reason={end_reason}"
                    )

            source = chunks()
            call.playback = create_task(
                self._transport().play_audio(
                    call.adapter_signature, call.call_id, call.generation, source, trace_id=playback_id
                ),
                f"anima_voice.qq.speak.{stream_id}",
            )
            end_reason = "error"
            try:
                result = await call.playback.task
                if result.get("drained") is not True:
                    raise RuntimeError("QQ 语音输出未确认排空")
                call.playback_bytes += int(result.get("bytes", 0))
                end_reason = "drained"
                call.last_activity = time.monotonic()
                return result
            except asyncio.CancelledError:
                end_reason = "cancelled"
                raise
            finally:
                try:
                    await source.aclose()
                finally:
                    logger.info(
                        f"QQ 通话播音结束：trace_id={playback_id}, call_id={call.call_id}, generation={call.generation}, "
                        f"pcm_bytes={generated_bytes}, end_reason={end_reason}"
                    )
                    call.playback = None

    async def _stop_audio(self, call: QQCall) -> None:
        """只取消本插件拥有的采音、识别和播放任务。"""
        tasks = [handle.task for handle in (call.capture, call.consumer, call.playback) if handle is not None and handle.task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _stop_call(self, call: QQCall) -> None:
        """等待当前会话的任务退出。"""
        await self._stop_audio(call)
        if call.monitor is not None and call.monitor.task is not asyncio.current_task():
            call.monitor.task.cancel()
            await asyncio.gather(call.monitor.task, return_exceptions=True)

    async def close(self) -> None:
        """卸载时请求结束自有通话并释放任务，不关闭共享服务。"""
        if self._notice_task is not None:
            self._notice_task.task.cancel()
            await asyncio.gather(self._notice_task.task, return_exceptions=True)
            self._notice_task = None
        for call in list(self.calls.values()):
            try:
                if call.call_id and call.phase != "ended":
                    await self._transport().hangup_call(call.adapter_signature, call.call_id)
            except (RuntimeError, OSError, ValueError, TypeError) as error:
                logger.error(f"卸载挂断未确认：{type(error).__name__}")
            finally:
                await self._stop_call(call)
                self.calls.pop(call.stream_id, None)