# 配置

插件配置由框架管理，默认路径为 `config/plugins/anima_voice/config.toml`。`[plugin]` 控制启用状态、模型任务、提示词和纯文本重试；`[tts]` 控制 PCM 或完整 WAV 播放策略；`[audio]` 设置本机麦克风、输出设备及 WAV 响度目标；`[voice_call]` 设置 QQ 通话输入模式和静默超时。TTS 模型、风格、推理服务地址和合成参数由 TTS 插件管理，不在通话插件重复配置。

迁移必须通过 `tools/migrate_config.py` 显式执行。导入插件不会自动迁移。旧直播配置迁移不覆盖已有目标；`--embed-asr` 模式合并已有通话与独立识别配置，只有追加 `--apply` 才会备份并原子升级目标。

## 迁移顺序

在首次加载通话插件之前执行 [迁移器](../tools/migrate_config.py)，让新配置继承原参数。直播配置自动同步会把 `custom_prompt_modes` 转为 `custom_prompt_enabled` 并保存原始备份；如果直播先加载，通话迁移仍读取该备份中的原作用域。目标已有差异时需要核对，不自动覆盖。

模型列表、任务名、温度、token 上限、纯文本重试和 Action 挂起直接保留。自定义提示词只有在旧作用域包含 `voice` 时迁入；未设置旧作用域沿用旧默认规则。旧 `[vts].audio_output_device` 映射到 `[audio].output_device`，`[audio_drive].loudness_target_dbfs` 映射到 `[audio].loudness_target_dbfs`，原直播设备配置仍留在直播插件。

## TTS 行为

QQ 原生通话固定消费共享 TTS 的 48000Hz 单声道 s16le PCM 流并发送至 SnowLuma；不使用本机播放器，不把完整音频模式或失败后发文本作为通话音频的替代。以下播放开关控制本机语音功能。

`streaming_enabled` 默认开启。PCM 采用 48000Hz 单声道 s16le，首块即播，只有 `[wait:n]` 切分请求；当前段接收结束即请求下一段，不等待设备播完，静音在同一输出流中插入。不支持音效链，不对整条回复做 RMS 归一化。

关闭流式时，`sentence_split_enabled` 控制句子切分，`max_parallel_segments` 限制完整 WAV 合成并发；音频按原分段顺序输出，取消回收全部预合成任务。`audio.loudness_target_dbfs` 仅作用于完整 WAV，0 关闭归一化。

`emit_text_on_tts_failure` 只在明确启用时发送失败文本，不把文本视为语音播放成功。旧 HTTP endpoint、provider、timeout、重试和熔断字段不再使用。字段定义与默认值以 [配置模型](../config.py) 为准。

## 本机识别

`asr.enabled` 控制本机麦克风常驻监听。该功能独立于 QQ 原生通话，不会用电脑麦克风接管 QQ 电话。`[asr]` 保存 FunASR 模型、推理设备、代码路径和端点检测参数，识别模型首次推理才加载；`[activation]` 设置本机监听的 VAD 或热键激活；`[message]` 设置中间结果、长度和质量过滤；`[bot]` 保存本机语音身份。

## QQ 原生通话

通话只在 `kokoro_flow_chatter` 的 QQ 私聊中使用，默认不自动接听。`voice_call.input_mode` 默认 `asr`，通过 VAD 静音端点分句后使用内置 FunASR；`multimodal` 会将分句 WAV 提交给 KFC 主模型，要求该模型支持 `input_audio`。此流程是分句音频请求，不是 Live API。

入站网络块不直接成为消息。分句器保留前滚，候选起句需要至少 120ms 连续活动；起句后以起句 RMS 阈值的一半维持活动，避免轻声句尾过早计入静音。未确认的短活动记录为 `short_activity`，不提交模型。连续静音达到端点后提交完整 WAV；安全时长上限和采音正常 EOF 也可闭合有效片段。分句日志记录 `silence`、`max_duration` 或 `capture_eof`、`voice_ms`、超起句阈值时长和采样范围。`voice_ms` 是能量活动时长，不是模型判定的人声时长；持续背景声仍可能误触发，不能凭片段长度断定噪声或丢字。

多模态输入在当前未完成模型决策期间保留来源音频。新发言导致 LLM 打断时，KFC 按同一 `call_id`、`generation` 和来源序号把 PCM 重新封装为一个合法 WAV，同时更新待处理快照；重复来源和重叠采样只保留一次。已完成回合和其他代次不合入当前音频。缓冲上限为 120 秒，超出时明确失败，不静默裁剪旧音频；不增加额外的整通等待或 ASR 回退。

接通后使用 `action-kfc_voice_reply` 发送通话语音，`content` 为完整正文字符串，一次请求共享 TTS 并持续播放收到的 PCM，不等待完整文件。QQ 通话不使用打字延迟，不按标点或等待标记切分合成请求；`sentence_split_enabled` 仍只服务于本机完整音频播放。`voice_style` 直接使用共享 TTS 当前可用风格；未指定时使用共享 TTS 的默认风格，不可用的显式风格会报错。通话保持当前 KFC 原会话，普通聊天语音条提醒在通话期间暂停，结束后恢复。

语音输出日志用同一 `trace_id` 关联动作、完整正文的 TTS 请求、PCM 结束与 SL 排空，仅记录字符数、字节数和结束原因，不记录正文或音频数据。`eof` 表示 TTS 音频源正常结束，`drained` 表示 SL 已确认排空；`cancelled`、`error`、`consumer_closed` 独立记录。首块耗时和包含下游等待的总耗时不是纯合成耗时。正文已完整进入请求和字节守恒都不能单独证明模型已朗读全文或手机已播放完毕。

`/voice status` 显示收音块数、字节数、累计 RMS 和峰值、已提交语句数及已排空的播音字节数。收到音频不等于检测到语音，播音排空也不等于对端实际听到。QQ、采音和播音进程必须使用同一账号的音频设备映射；不能把全局设备与账号专属设备混用。

媒体与控制复用 SnowLuma 已有反向消息 WebSocket，不新增地址、端口或 token。取消本地音频消费不会释放上游采音订阅，只有通话结束或原消息连接断开时才释放。SnowLuma 需要包含 `6d8e0224` 或更新提交中的语音接口，并已启用账号语音宿主和音频环境；接口存在不等于部署已就绪。

`--embed-asr` 将旧识别插件的 `audio`、`activation`、`message`、`bot` 和 FunASR 的 `asr` 字段并入现有配置；现有通话配置显式字段优先。旧适配器启用状态映射到 `asr.enabled`，已有通话输出设备优先于旧识别播放设备。迁移不修改模型文件、原识别配置或数据库；部署前应确认模型和代码路径不依赖停用的插件目录。