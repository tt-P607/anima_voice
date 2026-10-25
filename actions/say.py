"""将通话文本经共享 TTS Service 播放或发送到 local_asr。"""

from __future__ import annotations

import asyncio
import base64
from typing import Annotated, Any, ClassVar

from src.app.plugin_system.api import send_api
from src.app.plugin_system.base import BaseAction

from .._internal_compat import create_task
from ..protocol import get_asr_output, require_plugin
from ..runtime import call_state
from ..runtime.heartbeat import keep_watchdog_alive
from ..speech.backend import synthesize_audio
from ..speech.markers import parse_speech_segments
from ..speech.streaming import play_streaming_speech
from ._tts_schema import inject_tts_params


class SayAction(BaseAction):
    """在语音通话中合成并输出语音。"""

    name = "say"
    associated_types: ClassVar[list[str]] = ["voice", "text"]
    chatter_allow: ClassVar[list[str]] = ["anima_voice"]
    description = "将内容合成为语音；等待用户时请另调 pass_and_wait。"

    @classmethod
    def to_schema(cls) -> dict[str, Any]:
        """返回共享 TTS Service 的语音参数 schema。"""

        return inject_tts_params(super().to_schema())

    async def execute(
        self, content: Annotated[str, "要说出的内容"], **tts_params: Any
    ) -> tuple[bool, str]:
        """合成文本片段，并按平台选择 ASR voice 消息或本机播放。"""

        view = require_plugin(self.plugin)
        if view.config is None:
            return False, "插件配置缺失"
        output_receipts: list[str] = []
        segments = parse_speech_segments(
            content,
            False if view.config.tts.streaming_enabled else view.config.tts.sentence_split_enabled,
        )
        if not segments:
            return False, "没有可播放的语音内容"
        if view.config.tts.streaming_enabled:
            if self.chat_stream.platform != "local_asr":
                if not await call_state.is_call_active_for_stream(self.chat_stream.stream_id):
                    return False, "当前聊天流已不处于通话中"
                if view.audio_player is None:
                    return False, "本机音频播放器未初始化"
            try:
                async with keep_watchdog_alive(self.chat_stream.stream_id):
                    played = await play_streaming_speech(
                        view=view, stream_id=self.chat_stream.stream_id,
                        platform=self.chat_stream.platform,
                        segments=segments, tts_params=tts_params,
                    )
            except (RuntimeError, ValueError, OSError) as error:
                fallback = await self._send_text_fallback(
                    " ".join(text for text, _ in segments)
                )
                return False, f"语音输出失败：{error}{fallback}"
            for text, _ in segments:
                await call_state.record_assistant_message(self.chat_stream.stream_id, text)
            return True, f"已输出 {played}/{len(segments)} 段语音"
        played = 0
        semaphore = asyncio.Semaphore(max(1, view.config.tts.max_parallel_segments))

        async def synthesize(text: str) -> Any:
            """在配置并发上限内合成片段。"""

            async with semaphore:
                return await synthesize_audio(self.chat_stream.stream_id, text, tts_params)

        handles = []
        try:
            async with keep_watchdog_alive(self.chat_stream.stream_id):
                handles = [
                    create_task(synthesize(text), f"anima_voice.tts.{id(self)}.{index}")
                    for index, (text, _) in enumerate(segments)
                ]
                for (text, wait), handle in zip(segments, handles, strict=True):
                    if wait:
                        await asyncio.sleep(wait)
                    try:
                        artifact = await handle.task
                    except (RuntimeError, ValueError, OSError) as error:
                        fallback = await self._send_text_fallback(text)
                        return False, f"语音输出失败：{error}{fallback}"
                    if not artifact.audio:
                        fallback = await self._send_text_fallback(text)
                        return False, f"TTS 未生成可播放音频{fallback}"
                    if self.chat_stream.platform == "local_asr":
                        send_result = await send_api.send_voice_result(
                            base64.b64encode(artifact.audio).decode("ascii"),
                            stream_id=self.chat_stream.stream_id,
                            platform=self.chat_stream.platform,
                            processed_plain_text=text,
                        )
                        sent, receipt = send_api.format_send_result(
                            send_result, "语音已发送"
                        )
                        if not sent:
                            return False, f"{played}/{len(segments)} 段已确认发送；{receipt}"
                        if "本地后处理失败" in receipt:
                            output_receipts.append(receipt)
                    else:
                        if not await call_state.is_call_active_for_stream(
                            self.chat_stream.stream_id
                        ):
                            return False, "当前聊天流已不处于通话中"
                        if view.audio_player is None:
                            return False, "本机音频播放器未初始化"
                        async with get_asr_output().suppress_microphone_during_playback():
                            await view.audio_player.play_audio(artifact.audio)
                    await call_state.record_assistant_message(
                        self.chat_stream.stream_id, text
                    )
                    played += 1
        finally:
            for handle in handles:
                if not handle.task.done():
                    handle.task.cancel()
            if handles:
                await asyncio.gather(*(handle.task for handle in handles), return_exceptions=True)
        details = "；".join(output_receipts)
        if self.chat_stream.platform == "local_asr":
            message = f"已确认发送 {played}/{len(segments)} 段语音"
        else:
            message = f"已在本机播放 {played}/{len(segments)} 段语音"
        return played == len(segments), f"{message}；{details}" if details else message

    async def _send_text_fallback(self, text: str) -> str:
        """按配置发送语音失败后的文字，并返回独立的平台回执。"""

        view = require_plugin(self.plugin)
        if view.config is None or not view.config.tts.emit_text_on_tts_failure:
            return ""
        result = await send_api.send_text_result(
            text,
            stream_id=self.chat_stream.stream_id,
            platform=self.chat_stream.platform,
        )
        sent, receipt = send_api.format_send_result(result, "降级文字已发送")
        return f"；{receipt}" if sent else f"；降级文字发送未确认：{receipt}"
