"""ASR 接管、Chatter 绑定和通话事件生命周期。"""

from __future__ import annotations

import asyncio
import datetime
import time
from typing import Any

from src.app.plugin_system.api import chat_api, event_api, stream_api
from src.app.plugin_system.api.log_api import get_logger

from .._internal_compat import build_notice_message, restart_stream_loop
from ..constants import (
    CHATTER_SIGNATURE,
    EVENT_VOICE_CALL_ENDED,
    EVENT_VOICE_CALL_STARTED,
)
from ..protocol import get_asr_output, require_plugin
from ..runtime import call_state
from ..speech.markers import parse_speech_segments
from ..speech.streaming import play_streaming_speech, stop_stream_playback
from .identity import resolve_caller_identity

logger = get_logger("anima_voice.voice_call")
_previous_chatters: dict[str, Any] = {}
_call_lifecycle_lock = asyncio.Lock()


async def start_asr_voice_session(
    platform: str, stream_id: str, *, user_id: str, user_name: str
) -> bool:
    """启动 ASR runtime 并建立转发目标。"""

    return await get_asr_output().start_voice_call_session(
        platform,
        stream_id,
        target_user_id=user_id,
        target_user_name=user_name,
        target_group_id="",
    )


async def end_asr_voice_session() -> None:
    """结束 ASR 转发会话。"""

    await get_asr_output().end_voice_call_session()


async def _restore_previous_chatter(stream_id: str, previous_signature: str) -> bool:
    """只在当前仍由 anima_voice 接管时恢复此前绑定。"""

    previous = _previous_chatters.pop(stream_id, None)
    current = chat_api.get_chatter_by_stream(stream_id)
    if current is None or current.get_signature() != CHATTER_SIGNATURE:
        return False
    if previous_signature and previous is not None and previous.get_signature() == previous_signature:
        chat_api.bind_chatter_for_stream(stream_id, previous)
        return True
    chat_api.restore_stream_to_default(stream_id)
    return True


async def _publish_started(active: call_state.ActiveCall) -> None:
    """发布通话开始事件。"""

    await event_api.publish_event(
        EVENT_VOICE_CALL_STARTED,
        {
            "caller_stream_id": active.caller_stream_id,
            "started_at": active.started_at,
            "previous_chatter_signature": active.previous_chatter_signature,
        },
    )


async def start_call(*, stream: Any, plugin: Any, reason: str) -> tuple[bool, str]:
    """通过 ASR 会话接管普通平台私聊。"""

    stream_id = stream.stream_id
    platform = (stream.platform or "").strip()
    if not platform or platform == "local_asr":
        return False, "当前流不支持发起平台私聊通话"
    async with _call_lifecycle_lock:
        if await call_state.get_active_call() is not None:
            return False, "已有进行中的通话"
        user_id, user_name = resolve_caller_identity(stream)
        if not user_id:
            return False, "无法从最近的对方消息确定平台用户 ID"
        current = chat_api.get_chatter_by_stream(stream_id)
        previous_signature = (
            (current.get_signature() or "") if current is not None else ""
        )
        if current is not None:
            _previous_chatters[stream_id] = current
        active = None
        slot_acquired = False
        redirect_attempted = False
        bound = False
        try:
            config = require_plugin(plugin).config
            timeout = config.voice_call.timeout_seconds if config is not None else 300.0
            active = await call_state.set_active_call(
                stream_id,
                previous_chatter_signature=previous_signature,
                timeout_seconds=timeout,
            )
            slot_acquired = True
            redirect_attempted = True
            if not await start_asr_voice_session(
                platform, stream_id, user_id=user_id, user_name=user_name
            ):
                raise RuntimeError("ASR 通话会话启动失败")
            chatter_class = chat_api.get_chatter_class(CHATTER_SIGNATURE)
            if chatter_class is None:
                raise RuntimeError("anima_voice Chatter 未注册")
            chat_api.bind_chatter_for_stream(
                stream_id, chatter_class(stream_id=stream_id, plugin=plugin)
            )
            bound = True
            start_note = (
                f"[语音通话开始 @{active.started_at:.0f}] 用户已接通本地语音通话。"
                "从此处到下一条 [语音通话结束] 之间的对话发生在电话里；"
                "回复通过本机扬声器播放，用户的话来自麦克风 ASR 识别。"
            )
            await call_state.record_system_note(stream_id, start_note)
            stream.context.add_history_message(
                build_notice_message(
                    message_id=f"call_start_{active.started_at}",
                    content=start_note,
                    platform=platform,
                    stream_id=stream_id,
                    time=active.started_at,
                )
            )
            prompt = reason.strip() or "我打给你吧，咱们语音聊？"
            await call_state.record_assistant_message(stream_id, prompt)
            from src.app.plugin_system.api.send_api import send_text

            await send_text(prompt, stream_id=stream_id, platform=platform)
            await speak(plugin, stream_id, prompt)
            await _publish_started(active)
            await restart_stream_loop(stream_id)
        except (Exception, asyncio.CancelledError) as exc:
            failures = [str(exc)]
            if redirect_attempted:
                try:
                    await end_asr_voice_session()
                except Exception as cleanup_exc:  # noqa: BLE001 - 继续释放本地资源
                    failures.append(f"ASR 回滚失败：{cleanup_exc}")
            if bound:
                try:
                    restored = await _restore_previous_chatter(
                        stream_id, previous_signature
                    )
                    if restored:
                        await restart_stream_loop(stream_id)
                except Exception as cleanup_exc:  # noqa: BLE001 - 继续清除通话状态
                    failures.append(f"Chatter 回滚失败：{cleanup_exc}")
            if slot_acquired:
                await call_state.clear_active_call()
            _previous_chatters.pop(stream_id, None)
            if isinstance(exc, asyncio.CancelledError):
                raise
            return False, f"启动通话失败：{'；'.join(failures)}"
        return True, "通话已开启，对方可以开始说话了"


async def speak(plugin: Any, stream_id: str, text: str) -> None:
    """按通话配置合成并在本机播放接通或挂断提示。"""

    from ..speech.backend import synthesize_audio

    view = require_plugin(plugin)
    if view.config is None:
        raise RuntimeError("插件配置缺失，无法播放通话语音")
    if view.audio_player is None:
        raise RuntimeError("本机音频播放器未初始化，无法播放通话语音")
    if view.config.tts.streaming_enabled:
        segments = parse_speech_segments(text, split_sentences=False)
        if not segments:
            raise RuntimeError("TTS 未返回可播放音频")
        await play_streaming_speech(
            view=view, stream_id=stream_id, platform="voice_call",
            segments=segments, tts_params={},
        )
        return
    artifact = await synthesize_audio(stream_id, text, {})
    if not artifact.audio:
        raise RuntimeError("TTS 未返回可播放音频")
    async with get_asr_output().suppress_microphone_during_playback():
        await view.audio_player.play_audio(artifact.audio)


def _end_note(started_at: float, ended_at: float, reason: str) -> str:
    """生成通话结束边界标注。"""

    start = datetime.datetime.fromtimestamp(started_at, datetime.UTC).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    end = datetime.datetime.fromtimestamp(ended_at, datetime.UTC).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    labels = {
        "model": "由你主动挂断",
        "user": "由用户挂断",
        "timeout": "超时自动挂断",
        "manual": "管理员手动挂断",
        "plugin_unload": "服务停止，通话中断",
    }
    return f"[语音通话结束 @ {end}] 本通话起于 {start}，持续 {max(0, int(ended_at - started_at))} 秒，{labels[reason]}。接下来的对话回到普通文字聊天。"


async def finalize_call(
    *, stream_id: str, farewell: str, end_reason: call_state.CallEndReason, plugin: Any
) -> tuple[bool, str]:
    """停止 ASR、恢复原绑定、播放本机告别语并发布结束事件。"""

    async with _call_lifecycle_lock:
        active = await call_state.get_active_call()
        if active is None or active.caller_stream_id != stream_id:
            _previous_chatters.pop(stream_id, None)
            return False, "当前 stream 没有进行中的通话"
        failures: list[str] = []
        ended_at = time.time()
        snapshot = active
        farewell_text = farewell or "嗯，那先这样吧。"
        note = _end_note(active.started_at, ended_at, end_reason)
        await stop_stream_playback(stream_id)
        try:
            try:
                await end_asr_voice_session()
            except Exception as exc:  # noqa: BLE001 - 继续执行本地挂断清理
                failures.append(f"ASR 关闭失败：{exc}")
            try:
                await call_state.record_assistant_message(stream_id, farewell_text)
                await speak(plugin, stream_id, farewell_text)
            except Exception as exc:  # noqa: BLE001 - TTS 失败不得阻止释放通话
                failures.append(f"告别语播放失败：{exc}")
            try:
                await call_state.record_system_note(stream_id, note)
                chat_stream = await stream_api.get_stream(stream_id)
                if chat_stream is not None:
                    chat_stream.context.add_history_message(
                        build_notice_message(
                            message_id=f"call_end_{ended_at}",
                            content=note,
                            platform=chat_stream.platform,
                            stream_id=stream_id,
                            time=ended_at,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - 边界标注失败不得阻止释放通话
                failures.append(f"结束边界记录失败：{exc}")
        finally:
            snapshot = await call_state.end_call(end_reason) or active
            restored = False
            try:
                restored = await _restore_previous_chatter(
                    stream_id, snapshot.previous_chatter_signature
                )
            except Exception as exc:  # noqa: BLE001 - 状态与事件仍需完成
                failures.append(f"Chatter 恢复失败：{exc}")
            finally:
                _previous_chatters.pop(stream_id, None)
            try:
                await event_api.publish_event(
                    EVENT_VOICE_CALL_ENDED,
                    {
                        "caller_stream_id": snapshot.caller_stream_id,
                        "started_at": snapshot.started_at,
                        "ended_at": ended_at,
                        "previous_chatter_signature": snapshot.previous_chatter_signature,
                        "duration_seconds": max(0.0, ended_at - snapshot.started_at),
                        "messages_in_call": list(snapshot.messages_in_call),
                        "end_reason": end_reason,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - 将事件失败纳入调用结果
                failures.append(f"结束事件发布失败：{exc}")
            if restored:
                try:
                    await restart_stream_loop(stream_id)
                except Exception as exc:  # noqa: BLE001 - 结束事务先于 loop 重启完成
                    failures.append(f"流循环重启失败：{exc}")
        if failures:
            return False, f"通话已清理，但部分步骤失败：{'；'.join(failures)}"
        return True, f"通话已结束（原因：{end_reason}）"
