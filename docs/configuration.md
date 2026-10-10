# 配置

插件配置由框架管理，默认路径为 `config/plugins/anima_voice/config.toml`。`[plugin]` 控制启用状态、模型任务、提示词和纯文本重试；`[tts]` 控制 PCM 或完整 WAV 播放策略；`[audio]` 设置麦克风、输出设备及 WAV 响度目标；`[voice_call]` 设置私聊通话静默超时。TTS 模型、风格、推理服务地址和合成参数由 TTS 插件管理，不在通话插件重复配置。

迁移必须通过 `tools/migrate_config.py` 显式执行。导入插件不会自动迁移。旧直播配置迁移不覆盖已有目标；`--embed-asr` 模式合并已有通话与独立识别配置，只有追加 `--apply` 才会备份并原子升级目标。

## 迁移顺序

在首次加载通话插件之前执行 [迁移器](../tools/migrate_config.py)，让新配置继承原参数。直播配置自动同步会把 `custom_prompt_modes` 转为 `custom_prompt_enabled` 并保存原始备份；如果直播先加载，通话迁移仍读取该备份中的原作用域。目标已有差异时需要核对，不自动覆盖。

模型列表、任务名、温度、token 上限、纯文本重试和 Action 挂起直接保留。自定义提示词只有在旧作用域包含 `voice` 时迁入；未设置旧作用域沿用旧默认规则。旧 `[vts].audio_output_device` 映射到 `[audio].output_device`，`[audio_drive].loudness_target_dbfs` 映射到 `[audio].loudness_target_dbfs`，原直播设备配置仍留在直播插件。

## TTS 行为

`streaming_enabled` 默认开启。PCM 采用 48000Hz 单声道 s16le，首块即播，只有 `[wait:n]` 切分请求；当前段接收结束即请求下一段，不等待设备播完，静音在同一输出流中插入。不支持音效链，不对整条回复做 RMS 归一化。

关闭流式时，`sentence_split_enabled` 控制句子切分，`max_parallel_segments` 限制完整 WAV 合成并发；音频按原分段顺序输出，取消回收全部预合成任务。`audio.loudness_target_dbfs` 仅作用于完整 WAV，0 关闭归一化。

`emit_text_on_tts_failure` 只在明确启用时发送失败文本，不把文本视为语音播放成功。旧 HTTP endpoint、provider、timeout、重试和熔断字段不再使用。字段定义与默认值以 [配置模型](../config.py) 为准。

## 本机识别

`asr.enabled` 控制常驻监听；关闭时接通平台私聊仍会临时启动麦克风，挂断后释放。`[asr]` 同时保存 FunASR 模型、推理设备、代码路径和端点检测参数，识别模型首次推理才加载。`[activation]` 设置 VAD 或热键激活，平台通话期间强制持续监听；`[message]` 设置中间结果、长度和质量过滤；`[bot]` 保存本地身份，平台重定向使用真实来电身份。

`--embed-asr` 将旧识别插件的 `audio`、`activation`、`message`、`bot` 和 FunASR 的 `asr` 字段并入现有配置；现有通话配置显式字段优先。旧适配器启用状态映射到 `asr.enabled`，已有通话输出设备优先于旧识别播放设备。迁移不修改模型文件、原识别配置或数据库；部署前应确认模型和代码路径不依赖停用的插件目录。