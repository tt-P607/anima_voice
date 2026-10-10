"""语音通话所需的最小框架内部兼容能力。"""

from __future__ import annotations

from typing import Any


def create_task(coro: Any, name: str) -> Any:
    """通过框架 TaskManager 创建后台任务。"""

    from src.kernel.concurrency import get_task_manager

    return get_task_manager().create_task(
        coro, name=name, daemon=True, metadata={"plugin": "anima_voice"}
    )


def cancel_task(handle: Any) -> None:
    """取消 TaskManager 返回的任务句柄。"""

    from src.kernel.concurrency import get_task_manager

    get_task_manager().cancel_task(handle.task_id)


def feed_watchdog(stream_id: str) -> None:
    """喂一次 stream watchdog。"""

    from src.kernel.concurrency import get_watchdog

    get_watchdog().feed_dog(stream_id)


def default_context_compression_handler() -> Any:
    """返回框架默认的聊天上下文压缩处理器。"""

    from src.core.utils.context_compression import (
        default_chat_context_compression_handler,
    )

    return default_chat_context_compression_handler


def stream_reminder_bucket(stream_id: str, bucket: str) -> str:
    """返回流私有 SystemReminder bucket 名称。"""

    from src.core.prompt import STREAM_BUCKET_PREFIX

    return f"{STREAM_BUCKET_PREFIX}{stream_id}:{bucket}"


async def restart_stream_loop(stream_id: str) -> None:
    """重启目标流循环使 chatter 绑定生效，失败时抛出 RuntimeError。"""

    from src.core.transport.distribution.stream_loop_manager import (
        get_stream_loop_manager,
    )

    if not await get_stream_loop_manager().restart_stream_loop(stream_id):
        raise RuntimeError("流循环重启失败")


def build_notice_message(
    *, message_id: str, content: str, platform: str, stream_id: str, time: float
) -> Any:
    """构造通话边界用的 NOTICE 消息。"""

    from src.core.models.message import Message, MessageType

    return Message(
        message_id=message_id,
        content=content,
        processed_plain_text=content,
        platform=platform,
        stream_id=stream_id,
        sender_id="system",
        sender_name="系统通知",
        message_type=MessageType.NOTICE,
        time=time,
    )


def render_plugin_config(config_type: type, data: dict[str, Any]) -> str:
    """使用框架现有签名 renderer 生成插件配置 TOML。"""

    from src.kernel.config.core import _render_toml_with_signature

    return _render_toml_with_signature(config_type, data)


def get_core_config() -> Any:
    """返回框架核心配置，包括全局人格设置。"""

    from src.core.config import get_core_config as get_config

    return get_config()
