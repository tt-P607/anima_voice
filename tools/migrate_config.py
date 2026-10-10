"""从 anima_chatter 配置离线迁移语音通话配置。"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from .._internal_compat import render_plugin_config
from ..config import AnimaVoiceConfig


def map_legacy_config(source: dict[str, Any]) -> dict[str, Any]:
    """将旧配置中的通话、TTS 和本机音频字段映射到新结构。"""

    old_plugin = source.get("plugin", {})
    old_tts = source.get("tts", {})
    old_vts = source.get("vts", {})
    old_audio_drive = source.get("audio_drive", {})
    plugin = {
        key: old_plugin[key]
        for key in (
            "enabled",
            "model_task",
            "models",
            "temperature",
            "max_tokens",
            "plain_text_retry_limit",
            "enable_action_suspend",
        )
        if key in old_plugin
    }
    prompt_modes = old_plugin.get("custom_prompt_modes", ["voice", "vtb", "vtb_live"])
    if not isinstance(prompt_modes, list):
        raise TypeError("custom_prompt_modes 必须是字符串列表")
    normalized_modes = {str(mode).strip().lower() for mode in prompt_modes}
    if "voice" in normalized_modes and "custom_prompt" in old_plugin:
        plugin["custom_prompt"] = old_plugin["custom_prompt"]
    tts_keys = AnimaVoiceConfig.default()["tts"]
    tts = {key: old_tts[key] for key in tts_keys if key in old_tts}
    audio: dict[str, Any] = {}
    if "audio_output_device" in old_vts:
        audio["output_device"] = old_vts["audio_output_device"]
    if "loudness_target_dbfs" in old_audio_drive:
        audio["loudness_target_dbfs"] = old_audio_drive["loudness_target_dbfs"]
    return {"plugin": plugin, "tts": tts, "audio": audio, "voice_call": {}}


def migrate_config(
    source_path: Path, target_path: Path, *, apply: bool = False
) -> dict[str, Any]:
    """预览迁移结果，或在显式授权后备份并原子写入新配置。"""

    source_bytes = source_path.read_bytes()
    legacy_data = tomllib.loads(source_bytes.decode("utf-8"))
    backup = source_path.with_name(f"{source_path.name}.anima_voice.bak")
    if backup.exists() and "custom_prompt_modes" not in legacy_data.get("plugin", {}):
        legacy_data = tomllib.loads(backup.read_text(encoding="utf-8"))
    mapped = map_legacy_config(legacy_data)
    rendered = _render(mapped)
    if target_path.exists():
        current_data = tomllib.loads(target_path.read_text(encoding="utf-8"))
        expected_data = tomllib.loads(rendered)
        return {
            "written": False,
            "matches": current_data == expected_data,
            "target": target_path,
        }
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if (
        backup.exists()
        and "custom_prompt_modes" in tomllib.loads(source_bytes.decode("utf-8")).get("plugin", {})
        and backup.read_bytes() != source_bytes
    ):
        raise FileExistsError(f"迁移备份与源文件不一致：{backup}")
    if not apply:
        return {
            "written": False,
            "matches": False,
            "target": target_path,
            "preview": mapped,
        }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    backup_created = False
    try:
        if not backup.exists():
            shutil.copy2(source_path, backup)
            backup_created = True
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(rendered)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, target_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if backup_created:
            backup.unlink(missing_ok=True)
        raise
    return {"written": True, "matches": True, "target": target_path, "backup": backup}


def _render(mapped: dict[str, Any]) -> str:
    """复用框架签名感知 renderer 输出标准插件 TOML。"""

    data = AnimaVoiceConfig.default()
    for section_name, values in mapped.items():
        if isinstance(values, dict):
            data.setdefault(section_name, {}).update(values)
    AnimaVoiceConfig.from_dict(data)
    return render_plugin_config(AnimaVoiceConfig, data)


def map_embedded_config(
    voice: dict[str, Any], adapter: dict[str, Any], funasr: dict[str, Any]
) -> dict[str, Any]:
    """合并识别设置到通话配置，保留已有新字段并去掉外部服务选项。"""

    data = AnimaVoiceConfig.default()
    for section, defaults in data.items():
        if isinstance(defaults, dict):
            original = voice.get(section, {})
            defaults.update({key: copy.deepcopy(original[key]) for key in defaults if key in original})
    for section in ("audio", "activation", "bot", "message"):
        legacy = adapter.get(section, {})
        current = voice.get(section, {})
        for key in data[section]:
            if key in legacy and key not in current:
                data[section][key] = copy.deepcopy(legacy[key])
    for key, value in funasr.get("asr", {}).items():
        if key in data["asr"] and key not in voice.get("asr", {}):
            data["asr"][key] = copy.deepcopy(value)
    if "enabled" not in voice.get("asr", {}) and "enabled" in adapter.get("plugin", {}):
        data["asr"]["enabled"] = adapter["plugin"]["enabled"]
    if "output_device" not in voice.get("audio", {}) and "output_device" in adapter.get("playback", {}):
        data["audio"]["output_device"] = adapter["playback"]["output_device"]
    AnimaVoiceConfig.from_dict(data)
    return data


def migrate_embedded_config(
    voice_path: Path, adapter_path: Path, funasr_path: Path, *, apply: bool = False
) -> dict[str, Any]:
    """显式备份并原子升级现有通话配置，旧识别配置保持不变。"""

    original = voice_path.read_bytes()
    voice = tomllib.loads(original.decode("utf-8"))
    adapter = tomllib.loads(adapter_path.read_text(encoding="utf-8"))
    funasr = tomllib.loads(funasr_path.read_text(encoding="utf-8"))
    mapped = map_embedded_config(voice, adapter, funasr)
    rendered = render_plugin_config(AnimaVoiceConfig, mapped)
    if voice == tomllib.loads(rendered):
        return {"written": False, "matches": True}
    if not apply:
        return {"written": False, "matches": False, "preview": mapped}
    backup = voice_path.with_name(f"{voice_path.name}.before_embedded_asr.bak")
    if backup.exists():
        if backup.read_bytes() != original:
            raise FileExistsError("通话配置备份与当前源文件不同，拒绝覆盖")
    else:
        shutil.copy2(voice_path, backup)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=voice_path.parent,
            prefix=f".{voice_path.name}.", suffix=".tmp", delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(rendered)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, voice_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return {"written": True, "matches": True, "backup": backup}


def main() -> int:
    """执行显式离线配置迁移 CLI。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("config/plugins/anima_chatter/config.toml")
    )
    parser.add_argument(
        "--target", type=Path, default=Path("config/plugins/anima_voice/config.toml")
    )
    parser.add_argument("--apply", action="store_true", help="备份旧配置并写入目标配置")
    parser.add_argument("--embed-asr", action="store_true", help="将已有识别配置并入通话配置")
    parser.add_argument(
        "--asr-config", type=Path, default=Path("config/plugins/asr_adapter_anima/config.toml")
    )
    parser.add_argument(
        "--funasr-config", type=Path, default=Path("config/plugins/funasr_asr_provider_anima/config.toml")
    )
    args = parser.parse_args()
    if args.embed_asr:
        result = migrate_embedded_config(
            args.target, args.asr_config, args.funasr_config, apply=args.apply,
        )
        print("内置 ASR 配置已迁移" if result["written"] else "配置已一致" if result["matches"] else "迁移预览完成，未写入")
        return 0
    result = migrate_config(args.source, args.target, apply=args.apply)
    if result["written"]:
        print(f"已迁移到 {result['target']}；旧配置备份：{result['backup']}")
    elif result["matches"]:
        print(f"目标配置已存在且与映射一致：{result['target']}")
    elif "preview" in result:
        print(f"迁移预览完成，未写入文件：{result['target']}")
    else:
        print(f"目标配置已存在且有差异，未覆盖：{result['target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
