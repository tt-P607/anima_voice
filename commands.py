"""语音通话兜底控制命令。"""

from __future__ import annotations

import time

from src.app.plugin_system.api import send_api
from src.app.plugin_system.base import BaseCommand, cmd_route
from src.app.plugin_system.types import PermissionLevel

from .runtime import call_state
from .voice_call.lifecycle import finalize_call


class VoiceCommand(BaseCommand):
    """强制结束通话或查看当前通话状态。"""

    name: str = "voice"
    description: str = "语音通话控制：off=挂断，status=查看状态。"
    permission_level: PermissionLevel = PermissionLevel.OWNER

    async def _reply(self, text: str) -> None:
        """向当前命令流发送回执。"""

        await send_api.send_text(text, stream_id=self.stream_id)

    @cmd_route("off")
    async def handle_off(self) -> tuple[bool, str]:
        """手动挂断当前流的通话。"""

        runtime = getattr(self.plugin, "qq_calls", None)
        qq_state = runtime.get_state(self.stream_id) if runtime is not None else None
        if runtime is not None and qq_state is not None:
            if qq_state.get("phase") == "unknown":
                text = "QQ 通话状态未知；当前只允许查询状态，不执行控制操作。"
                await self._reply(text)
                return False, text
            if not qq_state.get("call_id"):
                text = "QQ 呼叫状态尚未绑定通话编号；请先查询状态，不能重拨或挂断。"
                await self._reply(text)
                return False, text
            try:
                await runtime.hangup(self.stream_id)
            except Exception as error:  # noqa: BLE001
                text = f"QQ 挂断请求失败：{str(error).strip() or type(error).__name__}"
                await self._reply(text)
                return False, text
            text = "已请求挂断 QQ 通话，等待状态确认。"
            await self._reply(text)
            return True, text

        active = await call_state.get_active_call()
        if active is None or active.caller_stream_id != self.stream_id:
            await self._reply("当前流没有进行中的语音通话。")
            return False, "stream mismatch or no call"
        ok, result = await finalize_call(
            stream_id=self.stream_id,
            farewell="通话已被手动挂断。",
            end_reason="manual",
            plugin=self.plugin,
        )
        await self._reply("通话已挂断。" if ok else result)
        return ok, result

    @cmd_route("status")
    async def handle_status(self) -> tuple[bool, str]:
        """查看当前通话及静默超时状态。"""

        runtime = getattr(self.plugin, "qq_calls", None)
        qq_state = runtime.get_state(self.stream_id) if runtime is not None else None
        if qq_state is not None:
            text = "\n".join(
                (
                    f"QQ 通话状态：{qq_state.get('phase', 'unknown')}",
                    f"通话编号：{qq_state.get('call_id') or '尚未确认'}",
                    f"输入模式：{qq_state.get('input_mode', 'unknown')}",
                    f"音频状态：{qq_state.get('audio_error') or '正常'}",
                    f"已收音频：{qq_state.get('capture_chunks', 0)} 块 / {qq_state.get('capture_bytes', 0)} 字节",
                    f"收音电平：RMS={qq_state.get('capture_rms', 0.0):.6f}，峰值={qq_state.get('capture_peak', 0.0):.6f}",
                    f"已提交语句：{qq_state.get('submitted_utterances', 0)}；已排空播音：{qq_state.get('playback_bytes', 0)} 字节",
                )
            )
            await self._reply(text)
            return True, "QQ status reported"

        active = await call_state.get_active_call()
        if active is None:
            await self._reply("当前没有进行中的语音通话。")
            return True, "no active call"
        remaining = await call_state.get_remaining_seconds()
        text = "\n".join(
            (
                f"通话发起 stream：{active.caller_stream_id}",
                f"已持续：{int(time.time() - active.started_at)} 秒",
                f"剩余静默超时：{remaining:.0f} 秒"
                if remaining is not None
                else "剩余静默超时：未知",
                f"通话期间消息数：{len(active.messages_in_call)}",
            )
        )
        await self._reply(text)
        return True, "status reported"
