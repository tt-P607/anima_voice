"""从公开 TTS service 读取参数描述并注入 Action schema。"""

from __future__ import annotations

from typing import Any

from src.app.plugin_system.api.service_api import get_service

from ..constants import TTS_SERVICE


def inject_tts_params(schema: dict[str, Any]) -> dict[str, Any]:
    """将 TTS service 当前参数能力加入 schema。"""

    service = get_service(TTS_SERVICE)
    if service is None:
        raise RuntimeError(f"TTS service 未加载：{TTS_SERVICE}")
    caps = service.get_capabilities()
    parameters = schema.get("function", {}).get("parameters", {})
    properties = parameters.setdefault("properties", {})
    required = parameters.setdefault("required", [])
    for name in ("style", "language", "speed", "aux_refer_wav_paths"):
        guide = getattr(caps, f"{name}_guide", None)
        if guide is None:
            continue
        property_schema: dict[str, Any] = {
            "type": guide.param_type,
            "description": guide.description,
        }
        for source, target in (
            ("default", "default"),
            ("valid_values", "enum"),
            ("min_value", "minimum"),
            ("max_value", "maximum"),
        ):
            value = getattr(guide, source, None)
            if value is not None:
                property_schema[target] = value
        properties[name] = property_schema
        if guide.required and name not in required:
            required.append(name)
    return schema
