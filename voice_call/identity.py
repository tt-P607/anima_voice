"""从聊天流消息历史解析通话对象身份。"""

from __future__ import annotations

from src.app.plugin_system.types import ChatStream, MessageType


def resolve_caller_identity(chat_stream: ChatStream) -> tuple[str, str]:
    """返回最近一条对方消息中的平台用户 ID 和显示名。"""

    bot_id = (chat_stream.bot_id or "").strip()
    context = chat_stream.context
    for messages in (context.unread_messages, context.history_messages):
        for message in reversed(list(messages)):
            sender_id = (message.sender_id or "").strip()
            if not sender_id or sender_id.lower() == "system" or sender_id == bot_id:
                continue
            if message.message_type == MessageType.NOTICE:
                continue
            return sender_id, (message.sender_name or "").strip() or sender_id
    return "", ""
