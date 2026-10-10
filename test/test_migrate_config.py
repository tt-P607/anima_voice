"""anima_voice 离线配置迁移测试。"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from plugins.anima_voice.tools.migrate_config import (
    _render,
    map_embedded_config,
    map_legacy_config,
    migrate_config,
    migrate_embedded_config,
)


def test_custom_prompt_only_moves_when_voice_mode_is_enabled() -> None:
    """旧 custom_prompt_modes 未含 voice 时不得迁移自定义提示词。"""

    data = map_legacy_config(
        {"plugin": {"custom_prompt": "legacy", "custom_prompt_modes": ["vtb_live"]}}
    )

    assert "custom_prompt" not in data["plugin"]


def test_missing_custom_prompt_modes_uses_legacy_default_scope() -> None:
    """旧配置缺少 scope 字段时沿用包含 voice 的旧默认值。"""

    data = map_legacy_config(
        {"plugin": {"custom_prompt": "legacy", "enable_action_suspend": False}}
    )

    assert data["plugin"]["custom_prompt"] == "legacy"
    assert data["plugin"]["enable_action_suspend"] is False


@pytest.mark.parametrize("modes", ['["vtb_live"]', '[]', '[" Voice "]'])
def test_migration_after_live_load_reads_original_scope_from_backup(
    tmp_path: Path, modes: str,
) -> None:
    """直播先自动更新配置时，语音仍按原备份中的作用域继承提示词。"""

    from plugins.anima_chatter.config import AnimaChatterConfig

    source = tmp_path / "config.toml"
    target = tmp_path / "voice.toml"
    source.write_text(
        f'[plugin]\ncustom_prompt="shared prompt"\ncustom_prompt_modes={modes}\n',
        encoding="utf-8",
    )
    original = source.read_bytes()
    AnimaChatterConfig.load(source, auto_update=True)
    migrate_config(source, target, apply=True)
    migrated = tomllib.loads(target.read_text(encoding="utf-8"))
    assert migrated["plugin"]["custom_prompt"] == (
        "shared prompt" if "Voice" in modes else ""
    )
    assert source.with_name("config.toml.anima_voice.bak").read_bytes() == original


def test_config_migration_maps_fields_and_preserves_source(tmp_path: Path) -> None:
    """迁移映射模型、TTS、设备与响度，并备份且保留旧源文件。"""

    source = tmp_path / "old.toml"
    target = tmp_path / "new" / "config.toml"
    source.write_text(
        '[plugin]\nmodel_task="custom"\ncustom_prompt="voice prompt"\ncustom_prompt_modes=["voice"]\n'
        '[tts]\nprovider="legacy_provider"\nendpoint="http://localhost/tts"\n'
        '[vts]\naudio_output_device="mic@MME"\n'
        "[audio_drive]\nloudness_target_dbfs=-19.0\n",
        encoding="utf-8",
    )

    preview = migrate_config(source, target)
    assert preview["written"] is False
    assert not target.exists()

    result = migrate_config(source, target, apply=True)
    migrated = tomllib.loads(target.read_text(encoding="utf-8"))

    assert result["written"] is True
    assert result["backup"].read_bytes() == source.read_bytes()
    assert source.exists()
    assert migrated["plugin"]["model_task"] == "custom"
    assert migrated["plugin"]["custom_prompt"] == "voice prompt"
    assert "provider" not in migrated["tts"]
    assert "endpoint" not in migrated["tts"]
    assert migrated["audio"]["output_device"] == "mic@MME"
    assert migrated["audio"]["loudness_target_dbfs"] == -19.0


def test_existing_target_is_not_overwritten(tmp_path: Path) -> None:
    """已有目标只报告差异且不改写。"""

    source = tmp_path / "old.toml"
    target = tmp_path / "new.toml"
    source.write_text("[plugin]\n", encoding="utf-8")
    target.write_text("user = true\n", encoding="utf-8")

    result = migrate_config(source, target)

    assert result == {"written": False, "matches": False, "target": target}
    assert target.read_text(encoding="utf-8") == "user = true\n"


def test_existing_target_comparison_uses_toml_data(tmp_path: Path) -> None:
    """注释或排版差异不影响配置数据相等判断。"""

    source = tmp_path / "old.toml"
    target = tmp_path / "new.toml"
    source.write_text("[plugin]\nmodel_task='actor'\n", encoding="utf-8")
    rendered = _render(map_legacy_config({"plugin": {"model_task": "actor"}}))
    target.write_text(f"# local formatting\n{rendered}", encoding="utf-8")

    result = migrate_config(source, target)

    assert result["matches"] is True
    assert target.read_text(encoding="utf-8").startswith("# local formatting")


def test_existing_matching_backup_allows_idempotent_apply(tmp_path: Path) -> None:
    """已有且同源备份允许再次应用迁移。"""

    source = tmp_path / "old.toml"
    target = tmp_path / "new.toml"
    source.write_text("[plugin]\nenabled=true\n", encoding="utf-8")
    backup = source.with_name(f"{source.name}.anima_voice.bak")
    backup.write_bytes(source.read_bytes())

    result = migrate_config(source, target, apply=True)

    assert result["written"] is True
    assert target.exists()
    assert backup.read_bytes() == source.read_bytes()


def test_failed_apply_preserves_preexisting_matching_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """原子写入失败不删除此前验证过的同源备份。"""

    source = tmp_path / "old.toml"
    target = tmp_path / "new.toml"
    source.write_text("[plugin]\nenabled=true\n", encoding="utf-8")
    backup = source.with_name(f"{source.name}.anima_voice.bak")
    backup.write_bytes(source.read_bytes())

    def fail_replace(*_: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("plugins.anima_voice.tools.migrate_config.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        migrate_config(source, target, apply=True)

    assert backup.read_bytes() == source.read_bytes()
    assert not target.exists()


def test_embedded_migration_preserves_devices_identity_model_and_new_overrides() -> None:
    """采音、身份和模型参数映射到内置节，新配置显式值优先。"""

    voice = {
        "plugin": {"custom_prompt": "call prompt"},
        "tts": {"endpoint": "http://localhost/unused", "provider": "old", "max_parallel_segments": 2},
        "audio": {"output_device": "speaker@MME", "loudness_target_dbfs": -19.0},
        "asr": {"device": "cpu"},
    }
    adapter = {
        "plugin": {"enabled": True},
        "audio": {"device": "input", "block_size": 3200},
        "bot": {"speaker_id": "local-user", "bot_id": "local-bot"},
        "activation": {"mode": "push_to_talk", "hotkey": "space"},
        "message": {"min_text_length": 3},
        "playback": {"output_device": "different-speaker"},
    }
    funasr = {"asr": {"model": "example/model", "device": "cuda:0", "endpoint_silence_ms": 700}}
    data = map_embedded_config(voice, adapter, funasr)
    assert data["plugin"]["custom_prompt"] == "call prompt"
    assert data["audio"]["output_device"] == "speaker@MME"
    assert data["audio"]["loudness_target_dbfs"] == -19.0
    assert (data["audio"]["device"], data["audio"]["block_size"]) == ("input", 3200)
    assert (data["bot"]["speaker_id"], data["bot"]["bot_id"]) == ("local-user", "local-bot")
    assert data["activation"]["mode"] == "push_to_talk"
    assert data["message"]["min_text_length"] == 3
    assert data["asr"]["enabled"] is True
    assert data["asr"]["model"] == "example/model"
    assert data["asr"]["device"] == "cpu"
    assert data["asr"]["endpoint_silence_ms"] == 700
    assert not {"endpoint", "provider", "timeout"} & data["tts"].keys()


def test_embedded_apply_is_atomic_backed_up_and_idempotent(tmp_path: Path) -> None:
    """预览不落盘，升级保存完整原文备份，重复执行无副作用。"""

    voice = tmp_path / "voice.toml"
    adapter = tmp_path / "adapter.toml"
    funasr = tmp_path / "funasr.toml"
    voice.write_text('[plugin]\ncustom_prompt="call"\n[tts]\nprovider="old"\n', encoding="utf-8")
    adapter.write_text('[plugin]\nenabled=true\n[audio]\ndevice="mic"\n', encoding="utf-8")
    funasr.write_text('[asr]\nmodel="example/model"\ndevice="cpu"\n', encoding="utf-8")
    original = voice.read_bytes()
    assert migrate_embedded_config(voice, adapter, funasr)["written"] is False
    assert voice.read_bytes() == original
    result = migrate_embedded_config(voice, adapter, funasr, apply=True)
    assert result["written"] is True
    assert result["backup"].read_bytes() == original
    assert tomllib.loads(voice.read_text(encoding="utf-8"))["asr"]["enabled"] is True
    current = voice.read_bytes()
    assert migrate_embedded_config(voice, adapter, funasr, apply=True) == {"written": False, "matches": True}
    assert voice.read_bytes() == current


def test_failed_embedded_apply_keeps_source_and_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写入失败时原配置和恢复备份均保留，不留下半成品临时文件。"""

    voice = tmp_path / "voice.toml"
    adapter = tmp_path / "adapter.toml"
    funasr = tmp_path / "funasr.toml"
    voice.write_text("[plugin]\n", encoding="utf-8")
    adapter.write_text("[plugin]\nenabled=false\n", encoding="utf-8")
    funasr.write_text("[asr]\n", encoding="utf-8")
    original = voice.read_bytes()

    def fail(*_: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("plugins.anima_voice.tools.migrate_config.os.replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        migrate_embedded_config(voice, adapter, funasr, apply=True)
    assert voice.read_bytes() == original
    assert voice.with_name("voice.toml.before_embedded_asr.bak").read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
