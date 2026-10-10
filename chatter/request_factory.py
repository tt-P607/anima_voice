"""保留 anima 模型选择、提醒来源及 NDFC 请求元数据的构造。"""

from __future__ import annotations

from src.app.plugin_system.api.llm_api import (
    get_model_set_by_name,
    get_model_set_by_task,
)
from src.app.plugin_system.types import (
    LLMContextManager,
    LLMRequest,
    ModelSet,
    ReminderSourceSpec,
)

from .._internal_compat import (
    default_context_compression_handler,
    stream_reminder_bucket,
)
from ..config import PluginSection


def resolve_model_set(section: PluginSection, fallback_task: str) -> ModelSet:
    """优先用自定义模型列表，否则按任务名获取模型集。"""

    if section.models:
        selected: ModelSet = []
        for name in section.models:
            selected.extend(
                get_model_set_by_name(
                    name, temperature=section.temperature, max_tokens=section.max_tokens
                )
                or []
            )
        if selected:
            return selected
    task = section.model_task or fallback_task
    models = get_model_set_by_task(task)
    if not models:
        raise ValueError(f"无法解析模型集：任务 '{task}' 未配置")
    return models


def build_reminder_sources(
    bucket: str | None, stream_id: str
) -> list[ReminderSourceSpec] | None:
    """构造全局及流私有 SystemReminder 来源。"""

    if bucket is None:
        return None
    result = [ReminderSourceSpec(bucket=bucket, wrap_with_system_tag=True)]
    if stream_id:
        result.append(
            ReminderSourceSpec(
                bucket=stream_reminder_bucket(stream_id, bucket),
                wrap_with_system_tag=True,
            )
        )
    return result


def create_request(
    *,
    section: PluginSection,
    stream_id: str,
    task: str,
    request_name: str,
    with_reminder: str | None,
) -> LLMRequest:
    """创建 LLMRequest，保留 stream 元数据和 reminder 上下文。"""

    return LLMRequest(
        model_set=resolve_model_set(section, task),
        request_name=request_name,
        meta_data={"stream_id": stream_id, "_ndfc": True},
        context_manager=LLMContextManager(
            context_compression_handler=default_context_compression_handler(),
            reminder_sources=build_reminder_sources(with_reminder, stream_id),
        ),
    )
