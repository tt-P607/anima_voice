"""anima_voice 本地麦克风适配器运行时。"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mofox_wire import MessageEnvelope

from src.app.plugin_system.api.log_api import get_logger
from src.app.plugin_system.base import BaseAdapter
from src.app.plugin_system.types import Message, MessageType
from src.core.components.types import PlatformSendResult, PlatformSendStatus

from .. import _internal_compat
from ..config import AnimaVoiceConfig
from .audio_source import MicrophoneAudioSource
from .recognizer import StreamingRecognizer
from .text_quality import is_likely_normal_utterance

logger = get_logger("anima_voice.asr")


@dataclass(frozen=True)
class RedirectTarget:
    """识别文本临时转发目标。"""

    platform: str
    stream_id: str
    user_id: str
    user_name: str
    group_id: str = ""


class VoiceMicrophoneAdapter(BaseAdapter):
    """接入 local_asr 流并按需采集本机麦克风。"""

    name = "voice_microphone"
    adapter_version = "1.0.0"
    description = "anima_voice 内置本机麦克风 ASR 适配器"
    platform = "local_asr"
    run_in_subprocess = False

    def __init__(self, core_sink: Any, plugin: Any = None, **kwargs: Any) -> None:
        """初始化适配器状态，不加载模型或打开设备。"""

        super().__init__(core_sink, plugin, **kwargs)
        self._config: AnimaVoiceConfig | None = None
        self._audio_source: MicrophoneAudioSource | None = None
        self._recognizer: StreamingRecognizer | None = None
        self._recognition_task_info: Any | None = None
        self._recognition_running = False
        self._redirect_target: RedirectTarget | None = None
        self._activation_override: str | None = None
        self._last_partial_text = ""
        self._last_partial_emit_at = 0.0
        self._in_speech = False
        self._keyboard: Any | None = None
        self._toggle_active = False
        self._toggle_key_was_pressed = False
        self._playback_until = 0.0
        self._external_playback_count = 0
        self._played_voice_message_ids: set[str] = set()
        self._preroll_chunks: deque[Any] = deque()
        self._runtime_lock = asyncio.Lock()

    async def on_adapter_loaded(self) -> None:
        """仅在 ASR 常驻开关开启时启动麦克风。"""

        self._config = self._get_config()
        if self._config.asr.enabled:
            await self._start_runtime()

    async def on_adapter_unloaded(self) -> None:
        """停止识别并释放本地音频资源。"""

        await self._stop_runtime()
        if self.plugin is not None and self.plugin.audio_player is not None:
            await self.plugin.audio_player.stop()

    async def from_platform_message(self, raw: Any) -> MessageEnvelope | None:
        """本地麦克风适配器不解析外部平台消息。"""

        _ = raw
        return None

    async def _send_platform_message(self, envelope: MessageEnvelope) -> PlatformSendResult:
        """消费本地平台外发语音，避免重复播放已主动输出的语音。"""

        message_info = envelope.get("message_info") or {}
        message_id = str(message_info.get("message_id") or "")
        if message_id and message_id in self._played_voice_message_ids:
            self._played_voice_message_ids.discard(message_id)
            return PlatformSendResult(
                success=True, message_id=message_id, status=PlatformSendStatus.SENT
            )
        return await self._play_voice_envelope(envelope)

    async def get_bot_info(self) -> dict[str, Any]:
        """返回本地 ASR 的 Bot 身份。"""

        config = self._config or self._get_config()
        return {
            "bot_id": config.bot.bot_id,
            "bot_name": config.bot.bot_name,
            "platform": self.platform,
        }

    def is_connected(self) -> bool:
        """返回本地麦克风和识别循环是否运行。"""

        return bool(
            self._recognition_running
            and self._audio_source is not None
            and self._audio_source.is_running()
            and self._recognizer is not None
        )

    async def health_check(self) -> bool:
        """常驻关闭时报告健康；开启时检查本地识别运行状态。"""

        config = self._config or self._get_config()
        return not config.asr.enabled or self.is_connected()

    async def start_voice_call_session(
        self,
        target_platform: str,
        target_stream_id: str,
        target_user_id: str,
        target_user_name: str,
        target_group_id: str = "",
        force_always_on: bool = True,
    ) -> bool:
        """按需启动识别并将文本定向到通话发起者所在流。"""

        if not target_platform or not target_stream_id or not target_user_id:
            raise ValueError("通话 redirect 必须包含平台、流 ID 和真实用户 ID")
        async with self._runtime_lock:
            self._redirect_target = RedirectTarget(
                target_platform,
                target_stream_id,
                target_user_id,
                target_user_name or "语音通话",
                target_group_id,
            )
            self._activation_override = "always_on" if force_always_on else None
            try:
                await self._start_runtime()
            except Exception:
                self._redirect_target = None
                self._activation_override = None
                logger.error("启动语音通话 ASR 失败", exc_info=True)  # noqa: G201
                return False
            return self._recognition_running

    async def end_voice_call_session(self) -> None:
        """清除通话目标并按 ASR 常驻设置停止或保留识别。"""

        async with self._runtime_lock:
            self._redirect_target = None
            self._activation_override = None
            config = self._config or self._get_config()
            if not config.asr.enabled:
                await self._stop_runtime()

    async def _start_runtime(self) -> None:
        """创建本地识别器和音频源，并启动受管识别循环。"""

        if self._recognition_running:
            return
        config = self._config or self._get_config()
        mode = self._activation_mode(config)
        self._toggle_active = config.activation.toggle_initially_active
        self._toggle_key_was_pressed = False
        self._keyboard = self._load_keyboard_backend() if mode in {
            "push_to_talk", "toggle_key"
        } else None
        source = MicrophoneAudioSource(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels,
            device=config.audio.device,
            block_size=config.audio.block_size,
            queue_max_chunks=config.audio.queue_max_chunks,
        )
        recognizer = StreamingRecognizer(config)
        try:
            await source.start()
        except Exception:
            await source.stop()
            self._keyboard = None
            raise
        self._config = config
        self._audio_source = source
        self._recognizer = recognizer
        self._recognition_running = True
        try:
            self._recognition_task_info = _internal_compat.create_task(
                self._recognition_loop(), "anima_voice.asr.recognition_loop"
            )
        except Exception:
            self._recognition_running = False
            self._audio_source = None
            self._recognizer = None
            await source.stop()
            raise

    async def _stop_runtime(self) -> None:
        """停止识别循环、等待线程任务并释放麦克风。"""

        self._recognition_running = False
        task_info = self._recognition_task_info
        self._recognition_task_info = None
        if task_info is not None and task_info.task is not asyncio.current_task():
            _internal_compat.cancel_task(task_info)
            await asyncio.gather(task_info.task, return_exceptions=True)
        source = self._audio_source
        self._audio_source = None
        self._recognizer = None
        if source is not None:
            await source.stop()
        self._last_partial_text = ""
        self._last_partial_emit_at = 0.0
        self._in_speech = False
        self._keyboard = None
        self._toggle_active = False
        self._toggle_key_was_pressed = False
        self._preroll_chunks.clear()

    async def _recognition_loop(self) -> None:
        """消费麦克风音频并向核心提交 partial/final 文本。"""

        while self._recognition_running:
            try:
                source = self._audio_source
                recognizer = self._recognizer
                config = self._config
                if source is None or recognizer is None or config is None:
                    return
                samples_batch = [await source.read()]
                samples_batch.extend(source.drain_pending())
                if self._microphone_suppressed():
                    recognizer.new_stream()
                    self._in_speech = False
                    self._preroll_chunks.clear()
                    continue
                for samples in samples_batch:
                    accepted_chunks = self._select_audio_chunks(samples, config)
                    if not accepted_chunks:
                        if recognizer.has_audio() and self._is_explicit_activation_mode(config):
                            recognizer.mark_endpoint()
                            await self._emit_final_text()
                            recognizer.new_stream()
                        continue
                    for chunk in accepted_chunks:
                        await self._run_blocking(recognizer.accept_waveform, chunk)
                if config.message.commit_partial_results and self._can_run_partial_decode():
                    await self._run_blocking(recognizer.decode)
                    await self._maybe_emit_partial()
                if recognizer.is_endpoint():
                    await self._run_blocking(recognizer.decode)
                    await self._emit_final_text()
                    recognizer.new_stream()
                    self._in_speech = False
                    self._preroll_chunks.clear()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("ASR 识别循环发生错误", exc_info=True)  # noqa: G201
                await asyncio.sleep(1.0)

    async def _run_blocking(self, function: Any, *args: Any) -> Any:
        """在线程中运行阻塞调用，取消时等待线程真正完成。"""

        task_info = _internal_compat.create_task(
            asyncio.to_thread(function, *args), "anima_voice.asr.blocking_call"
        )
        try:
            return await asyncio.shield(task_info.task)
        except asyncio.CancelledError:
            await asyncio.gather(task_info.task, return_exceptions=True)
            raise

    async def _maybe_emit_partial(self) -> None:
        """按配置间隔提交变化的中间文本。"""

        recognizer = self._recognizer
        config = self._config
        if recognizer is None or config is None or not config.message.commit_partial_results:
            return
        text = await self._run_blocking(recognizer.get_text)
        if len(text) < config.message.min_text_length or text == self._last_partial_text:
            return
        if not self._should_accept_text(text, config):
            return
        now = time.monotonic()
        if now - self._last_partial_emit_at < config.message.partial_emit_interval:
            return
        self._last_partial_text = text
        self._last_partial_emit_at = now
        await self._send_text_to_core(text, is_final=False)

    def _can_run_partial_decode(self) -> bool:
        """积压超过队列四分之一时跳过耗时的中间推理。"""

        source = self._audio_source
        if source is None:
            return False
        capacity = source.queue_capacity()
        return capacity <= 0 or source.queue_size() <= max(1, capacity // 4)

    async def _emit_final_text(self) -> None:
        """校验并提交完整识别文本。"""

        recognizer = self._recognizer
        config = self._config
        if recognizer is None or config is None:
            return
        text = await self._run_blocking(recognizer.get_text)
        if len(text) < config.message.min_text_length:
            return
        if config.message.enable_confidence_filter:
            average = recognizer.get_avg_confidence()
            tokens = recognizer.get_token_confidences()
            if average is not None and (
                average < config.message.min_avg_confidence
                or tokens and min(tokens) < config.message.min_token_confidence
            ):
                return
        if self._should_accept_text(text, config):
            await self._send_text_to_core(text, is_final=True)
        self._last_partial_text = ""

    @staticmethod
    def _should_accept_text(text: str, config: AnimaVoiceConfig) -> bool:
        """过滤空文本、命令、明显噪声和可配置低质量文本。"""

        normalized = text.strip()
        if not normalized or normalized.startswith("/"):
            return False
        if VoiceMicrophoneAdapter._is_obvious_noise_text(normalized):
            return False
        if config.message.enable_quality_filter:
            return is_likely_normal_utterance(
                normalized,
                min_chars=config.message.min_text_length,
                max_ascii_ratio=config.message.max_ascii_ratio,
                min_cjk_ratio=config.message.min_cjk_ratio,
                min_common_cjk_ratio=config.message.min_common_cjk_ratio,
            ).accepted
        return True

    @staticmethod
    def _is_obvious_noise_text(text: str) -> bool:
        """过滤单字语气词和极端重复幻听。"""

        content = "".join(
            char
            for char in text.strip()
            if not char.isspace() and char not in "，。！？、；：,.!?;:()（）[]【】'\"“”‘’《》<>-—_"
        )
        if content in {"啊", "嗯", "呃", "额", "唔"}:
            return True
        if len(content) < 4:
            return False
        if len(set(content)) == 1:
            return True
        longest = 1
        run = 0
        previous = ""
        for char in content:
            run = run + 1 if char == previous else 1
            longest = max(longest, run)
            previous = char
        return longest >= 6 and longest * 2 >= len(content) and len(set(content)) <= 3

    async def _send_text_to_core(self, text: str, *, is_final: bool) -> None:
        """将本地识别结果转换为标准 incoming envelope。"""

        if not self.core_sink or not self.plugin:
            return
        config = self._config or self._get_config()
        target = self._redirect_target
        target_platform = target.platform if target else self.platform
        user_id = target.user_id if target else config.bot.speaker_id
        user_name = target.user_name if target else config.bot.speaker_name
        stream_id = target.stream_id if target else None
        user_info: dict[str, str] = {
            "platform": target_platform,
            "user_id": user_id,
            "user_nickname": user_name,
        }
        if target is not None and target.group_id:
            user_info["group_id"] = target.group_id
        envelope: dict[str, Any] = {
            "direction": "incoming",
            "message_info": {
                "platform": target_platform,
                "message_id": f"asr-{uuid.uuid4()}",
                "message_type": "private",
                "time": time.time(),
                "user_info": user_info,
                "extra": {
                    "asr": {
                        "engine": "funasr",
                        "is_final": is_final,
                        "redirected": stream_id is not None,
                    }
                },
            },
            "message_segment": [{"type": "text", "data": text.strip()}],
            "raw_message": {
                "text": text.strip(),
                "source": "microphone",
                "engine": "funasr",
                "is_final": is_final,
            },
        }
        if stream_id is not None:
            envelope["message_info"]["stream_id"] = stream_id
        await self.core_sink.send(cast(MessageEnvelope, envelope))

    def _activation_mode(self, config: AnimaVoiceConfig) -> str:
        """返回实例 override 优先的规范化激活模式。"""

        raw = self._activation_override or config.activation.mode or "vad"
        mode = raw.strip().lower()
        aliases = {
            "ptt": "push_to_talk", "push-to-talk": "push_to_talk",
            "push to talk": "push_to_talk", "key_hold": "push_to_talk",
            "hold_key": "push_to_talk", "toggle": "toggle_key",
            "key_toggle": "toggle_key", "always": "always_on",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"vad", "push_to_talk", "toggle_key", "always_on"}:
            logger.warning(f"未知 ASR 激活方式 {mode!r}，关闭音频识别")
            return "disabled"
        return mode

    def _is_explicit_activation_mode(self, config: AnimaVoiceConfig) -> bool:
        """判断是否由按键定义语音段边界。"""

        return self._activation_mode(config) in {"push_to_talk", "toggle_key"}

    @staticmethod
    def _load_keyboard_backend() -> Any:
        """加载按键激活所需的 keyboard 后端。"""

        try:
            import keyboard
        except ImportError as exc:
            raise RuntimeError("按键激活需要安装 keyboard 依赖") from exc
        return keyboard

    def _should_accept_audio_chunk(
        self, samples: Any, config: AnimaVoiceConfig
    ) -> bool:
        """依据 VAD、热键或持续模式判定音频是否送入识别器。"""

        mode = self._activation_mode(config)
        if mode == "always_on":
            self._in_speech = True
            return True
        if mode == "push_to_talk":
            self._in_speech = self._is_hotkey_pressed(config)
            return self._in_speech
        if mode == "toggle_key":
            self._update_toggle_activation(config)
            self._in_speech = self._toggle_active
            return self._in_speech
        if mode == "disabled":
            self._in_speech = False
            return False
        loud = self._is_loud_enough(samples, config.activation.vad_threshold)
        if not self._in_speech and not loud:
            return False
        if loud:
            self._in_speech = True
        return True

    def _select_audio_chunks(
        self, samples: Any, config: AnimaVoiceConfig
    ) -> list[Any]:
        """筛选识别音频并在 VAD 首次触发时补入前置块。"""

        was_in_speech = self._in_speech
        if not self._should_accept_audio_chunk(samples, config):
            self._push_preroll_chunk(samples, config)
            return []
        if was_in_speech:
            return [samples]
        chunks = list(self._preroll_chunks)
        chunks.append(samples)
        self._preroll_chunks.clear()
        return chunks

    def _push_preroll_chunk(self, samples: Any, config: AnimaVoiceConfig) -> None:
        """保存配置时长内的前置音频。"""

        limit = self._preroll_chunk_limit(config)
        if limit <= 0:
            self._preroll_chunks.clear()
            return
        self._preroll_chunks.append(samples)
        while len(self._preroll_chunks) > limit:
            self._preroll_chunks.popleft()

    @staticmethod
    def _preroll_chunk_limit(config: AnimaVoiceConfig) -> int:
        """按采样率、块大小换算前置音频块数。"""

        samples = int(
            config.audio.sample_rate * max(0, config.activation.preroll_ms) / 1000
        )
        if samples <= 0:
            return 0
        block_size = max(1, config.audio.block_size)
        return max(1, (samples + block_size - 1) // block_size)

    @staticmethod
    def _is_loud_enough(samples: Any, threshold: float) -> bool:
        """计算 RMS 音量是否达到激活阈值。"""

        if threshold <= 0:
            return True
        import numpy as np

        array = np.asarray(samples, dtype=np.float32)
        return bool(array.size and float(np.sqrt(np.mean(np.square(array)))) >= threshold)

    def _is_hotkey_pressed(self, config: AnimaVoiceConfig) -> bool:
        """查询当前激活热键状态。"""

        if self._keyboard is None:
            self._keyboard = self._load_keyboard_backend()
        hotkey = config.activation.hotkey.strip() or "space"
        return bool(self._keyboard.is_pressed(hotkey))

    def _update_toggle_activation(self, config: AnimaVoiceConfig) -> None:
        """在热键按下边沿切换监听状态。"""

        pressed = self._is_hotkey_pressed(config)
        if pressed and not self._toggle_key_was_pressed:
            self._toggle_active = not self._toggle_active
        self._toggle_key_was_pressed = pressed

    @asynccontextmanager
    async def suppress_microphone_during_playback(self) -> AsyncIterator[None]:
        """在外部播放器播放期间抑制麦克风，并保留 300ms 尾窗。"""

        self._external_playback_count += 1
        try:
            yield
        finally:
            self._external_playback_count -= 1
            self._playback_until = max(self._playback_until, time.monotonic() + 0.3)

    async def play_pcm_stream(
        self,
        chunks: AsyncIterator[bytes],
        *,
        sample_rate: int,
        channels: int = 1,
        sample_format: str = "s16le",
    ) -> int:
        """经插件自有播放器输出 PCM，并屏蔽网络间隙中的麦克风回采。"""

        async with self.suppress_microphone_during_playback():
            return await self._get_audio_player().play_pcm_stream(
                chunks,
                sample_rate=sample_rate,
                channels=channels,
                sample_format=sample_format,
            )

    async def stop_pcm_playback(self) -> None:
        """停止 anima_voice 自有播放器，不影响共享 TTS 服务。"""

        await self._get_audio_player().stop()

    async def send_played_voice(
        self,
        audio_data: bytes,
        *,
        stream_id: str,
        text: str,
    ) -> PlatformSendResult:
        """发送并入库已播放的语音，避免 adapter 再次播放。"""

        from src.app.plugin_system.api.send_api import send_message_result

        if not audio_data or not stream_id:
            return PlatformSendResult(
                success=False, error="已播放语音缺少音频或目标流",
                status=PlatformSendStatus.FAILED,
            )
        message_id = f"anima-voice-{uuid.uuid4()}"
        self._played_voice_message_ids.add(message_id)
        label = f"[语音:{text.strip()}]" if text.strip() else "[语音]"
        content = {
            "text": label,
            "media": [
                {
                    "type": "voice",
                    "data": f"base64|{base64.b64encode(audio_data).decode('ascii')}",
                    "context_mode": "placeholder",
                }
            ],
        }
        message = Message(
            message_id=message_id,
            content=content,
            processed_plain_text=label,
            message_type=MessageType.VOICE,
            platform="local_asr",
            chat_type="private",
            stream_id=stream_id,
        )
        try:
            result = await send_message_result(message)
        except BaseException:
            self._played_voice_message_ids.discard(message_id)
            raise
        if result.status in {
            PlatformSendStatus.FAILED, PlatformSendStatus.INTERCEPTED,
        }:
            self._played_voice_message_ids.discard(message_id)
        return result

    async def _play_voice_envelope(self, envelope: MessageEnvelope) -> PlatformSendResult:
        """提取普通外发 voice 音频并通过自有 AudioPlayer 播放。"""

        for segment in envelope.get("message_segment") or []:
            if not isinstance(segment, dict) or segment.get("type") != "voice":
                continue
            data = segment.get("data")
            audio: bytes | None = None
            if isinstance(data, bytes):
                audio = data
            elif isinstance(data, str):
                audio = self._decode_audio_data(data)
                if audio is None:
                    audio = self._read_audio_file(data)
            elif isinstance(data, dict):
                encoded = data.get("audio_base64") or data.get("data")
                if isinstance(encoded, str):
                    audio = self._decode_audio_data(encoded)
                path = data.get("path") or data.get("artifact_path")
                if audio is None and isinstance(path, str):
                    audio = self._read_audio_file(path)
            if audio is not None:
                try:
                    async with self.suppress_microphone_during_playback():
                        await self._get_audio_player().play_audio(audio)
                except (RuntimeError, ValueError, OSError) as error:
                    return PlatformSendResult(
                        success=False, error=f"本机语音播放失败：{error}",
                        status=PlatformSendStatus.FAILED,
                    )
                return PlatformSendResult(success=True, status=PlatformSendStatus.SENT)
            return PlatformSendResult(
                success=False, error="外发语音没有可播放的音频数据",
                status=PlatformSendStatus.FAILED,
            )
        return PlatformSendResult(
            success=False, error="本地 ASR 适配器不输出文本消息",
            status=PlatformSendStatus.FAILED,
        )

    def _get_audio_player(self) -> Any:
        """使用插件唯一播放器，避免本地监听和平台通话重复占用设备。"""

        player = self.plugin.audio_player if self.plugin is not None else None
        if player is None:
            raise RuntimeError("通话插件播放器尚未初始化")
        return player

    @staticmethod
    def _decode_audio_data(value: str) -> bytes | None:
        """严格解码 base64 音频。"""

        try:
            encoded = value.removeprefix("base64|").removeprefix("base64://")
            return base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _read_audio_file(value: str) -> bytes | None:
        """读取存在的本地音频文件。"""

        path = Path(value)
        if not path.is_file():
            return None
        return path.read_bytes()

    def _microphone_suppressed(self) -> bool:
        """检查 PCM 播放计数及其结束尾窗。"""

        return self._external_playback_count > 0 or time.monotonic() < self._playback_until

    def _get_config(self) -> AnimaVoiceConfig:
        """取得插件已注入的 anima_voice 配置。"""

        if self.plugin is None or self.plugin.config is None:
            raise RuntimeError("VoiceMicrophoneAdapter 缺少 anima_voice 配置")
        return cast(AnimaVoiceConfig, self.plugin.config)


__all__ = ["RedirectTarget", "VoiceMicrophoneAdapter"]