"""anima_voice 插件和内置语音识别配置模型。"""

from __future__ import annotations

from typing import ClassVar

from src.app.plugin_system.base import BaseConfig, Field, SectionBase, config_section


@config_section("plugin", title="通话设置")
class PluginSection(SectionBase):
    """通话聊天器行为和模型设置。"""

    enabled: bool = Field(default=True, description="是否启用通话聊天器")
    model_task: str = Field(default="actor", description="默认模型任务名")
    models: list[str] = Field(default_factory=list, description="自定义模型名称列表")
    temperature: float = Field(default=0.7, description="自定义模型温度")
    max_tokens: int = Field(default=8000, description="自定义模型最大输出 token 数")
    enable_action_suspend: bool = Field(
        default=True, description="是否挂起纯 Action 回合"
    )
    plain_text_retry_limit: int = Field(default=1, description="纯文本输出重试次数")
    custom_prompt: str = Field(default="", description="语音通话附加提示词")


@config_section("tts", title="TTS 设置")
class TTSSection(SectionBase):
    """共享 TTS 播放策略。"""

    streaming_enabled: bool = Field(default=True, description="启用共享 TTS 流式合成")
    sentence_split_enabled: bool = Field(default=True, description="按句切分合成")
    max_parallel_segments: int = Field(
        default=4, description="并行合成片段的最大数量，音频仍按原顺序输出"
    )
    emit_text_on_tts_failure: bool = Field(
        default=False, description="合成失败时发送文本，但仍报告语音输出失败"
    )


@config_section("audio", title="本机音频输出")
class AudioSection(SectionBase):
    """本机音频采集和输出配置。"""

    sample_rate: int = Field(default=16000, description="麦克风采样率")
    channels: int = Field(default=1, description="麦克风输入通道数")
    device: str = Field(default="", description="麦克风輸入设备名称或索引")
    block_size: int = Field(default=8000, description="每次麦克风回调采样点数")
    queue_max_chunks: int = Field(default=80, description="采集缓冲区最大音频块数")
    output_device: str = Field(default="", description="设备名或设备名@驱动")
    loudness_target_dbfs: float = Field(
        default=-14.0, description="目标 RMS 响度，0 关闭"
    )


@config_section("activation", title="监听激活")
class ActivationSection(SectionBase):
    """麦克风监听激活方式配置。"""

    mode: str = Field(default="vad", description="vad、按住热键、切换热键或持续监听")
    hotkey: str = Field(default="space", description="按键激活热键名称")
    vad_threshold: float = Field(default=0.003, description="VAD RMS 激活阈值")
    toggle_initially_active: bool = Field(default=False, description="切换模式初始是否激活")
    preroll_ms: int = Field(default=500, description="VAD 触发前保留的音频毫秒数")


@config_section("bot", title="本地语音身份")
class BotSection(SectionBase):
    """本地 ASR 入站和外发 Bot 身份。"""

    bot_id: str = Field(default="local_asr_bot", description="外发历史使用的 Bot ID")
    bot_name: str = Field(default="MoFox", description="外发历史使用的 Bot 名称")
    speaker_id: str = Field(default="local_microphone", description="本地 ASR 用户 ID")
    speaker_name: str = Field(default="Local Microphone", description="本地 ASR 用户名称")


@config_section("message", title="识别消息")
class MessageSection(SectionBase):
    """识别文本质量和提交策略。"""

    min_text_length: int = Field(default=2, description="提交文本最短长度")
    commit_partial_results: bool = Field(default=False, description="是否提交流式中间结果")
    partial_emit_interval: float = Field(default=1.0, description="中间结果提交间隔秒数")
    enable_quality_filter: bool = Field(default=False, description="启用中文文本质量过滤")
    enable_confidence_filter: bool = Field(default=True, description="启用识别置信度过滤")
    min_avg_confidence: float = Field(default=-0.6, description="平均 token 置信度下限")
    min_token_confidence: float = Field(default=-3.0, description="单 token 置信度下限")
    max_ascii_ratio: float = Field(default=0.2, description="英文字母占比上限")
    min_cjk_ratio: float = Field(default=0.65, description="中文字符占比下限")
    min_common_cjk_ratio: float = Field(default=0.45, description="常用汉字占比下限")


@config_section("asr", title="FunASR 识别模型")
class AsrSection(SectionBase):
    """内置 FunASR 模型和推理设置。"""

    enabled: bool = Field(default=False, description="是否随适配器常驻启动本机 ASR")
    model_type: str = Field(default="sensevoice_small", description="FunASR 模型类型")
    model: str = Field(default="FunAudioLLM/SenseVoiceSmall", description="模型 ID 或本地目录")
    hub: str = Field(default="ms", description="模型下载来源")
    device: str = Field(default="auto", description="推理设备")
    trust_remote_code: bool = Field(default=True, description="是否加载模型仓库 Python 代码")
    remote_code: str = Field(default="./model.py", description="模型远程代码路径")
    vad_model: str = Field(default="fsmn-vad", description="FunASR VAD 模型")
    enable_streaming: bool = Field(default=False, description="是否执行流式中间推理")
    max_single_segment_time: int = Field(default=30000, description="VAD 最大单段时长毫秒")
    batch_size_s: int = Field(default=0, description="FunASR 动态 batch 秒数")
    max_length: int = Field(default=32, description="Fun-ASR-Nano 最大生成 token 数")
    llm_dtype: str = Field(default="bf16", description="Fun-ASR-Nano 推理精度")
    ctc_only: bool = Field(default=False, description="Fun-ASR-Nano 仅使用 CTC")
    inference_window_ms: int = Field(default=1600, description="流式推理窗口毫秒")
    max_audio_window_ms: int = Field(default=30000, description="最大缓存音频窗口毫秒")
    endpoint_silence_ms: int = Field(default=900, description="端点静音时长毫秒")
    endpoint_rms_threshold: float = Field(default=0.008, description="端点语音 RMS 阈值")
    enable_endpoint: bool = Field(default=True, description="启用静音端点识别")


@config_section("voice_call", title="语音通话")
class VoiceCallSection(SectionBase):
    """通话静默超时配置。"""

    timeout_seconds: float = Field(default=300.0, description="静默超时秒数")


class AnimaVoiceConfig(BaseConfig):
    """anima_voice 插件配置。"""

    name: ClassVar[str] = "config"
    description: ClassVar[str] = "Anima 独立语音通话配置"
    plugin: PluginSection = Field(default_factory=PluginSection)
    tts: TTSSection = Field(default_factory=TTSSection)
    audio: AudioSection = Field(default_factory=AudioSection)
    voice_call: VoiceCallSection = Field(default_factory=VoiceCallSection)
    activation: ActivationSection = Field(default_factory=ActivationSection)
    bot: BotSection = Field(default_factory=BotSection)
    message: MessageSection = Field(default_factory=MessageSection)
    asr: AsrSection = Field(default_factory=AsrSection)


__all__ = [
    "ActivationSection",
    "AnimaVoiceConfig",
    "AsrSection",
    "AudioSection",
    "BotSection",
    "MessageSection",
    "PluginSection",
    "TTSSection",
    "VoiceCallSection",
]
