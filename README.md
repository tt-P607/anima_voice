# Anima 语音通话

Anima 语音通话为 KFC QQ 私聊提供 QQ 原生语音呼叫，并保留独立的本机麦克风语音交互功能。

## 功能

- 在当前 KFC QQ 私聊中发起 QQ 呼叫、接听、拒接和挂断；默认不会自动接听来电。
- 使用 QQ 通话音频，按 VAD 静音端点分句，再以默认 ASR 模式识别并交回原 KFC 会话。
- 可选 multimodal 模式将分句 WAV 输入对话模型；该模型必须支持 `input_audio`。
- 使用 GPT-SoVITS 生成语音并通过 SnowLuma 现有 QQ 通话音频流播放。
- 本机麦克风识别仍可独立使用，支持语音活动检测、热键和持续监听；它不是 QQ 原生通话。

## 使用条件

- Neo-MoFox 1.3.0-alpha.0 或更新版本，以及 Python 3.11 或更新版本。
- 启用 `neo_default_chatter` 和 `tts_voice_plugin-neo`，并配置好对话模型与 GPT-SoVITS。
- 启用 `kokoro_flow_chatter` 与 `snowluma_adapter`；SnowLuma 需要包含 `6d8e0224` 或更新提交中的语音接口，并完成账号语音宿主与音频设备部署。
- TTS 插件须提供本插件所需的语音服务和公共语音规则模板；仅有旧版完整音频接口的版本不能使用。
- 本机麦克风功能需要可用的麦克风、扬声器或耳机，并安装 FunASR 及相应音频依赖。
- `multimodal` 输入要求 KFC 所用主模型支持 OpenAI 兼容的 `input_audio`。

QQ 通话固定使用共享 TTS 的 48 kHz 单声道 S16LE 流，需要支持 V5 PCM 流的 GPT-SoVITS 服务。`tts.streaming_enabled` 仅控制本机播放路径。

## 安装与配置

在 Neo-MoFox 项目根目录执行：

```bash
git clone https://github.com/tt-P607/anima_voice.git plugins/anima_voice
```

首次加载后，配置位于 `config/plugins/anima_voice/config.toml`。主要设置如下：

| 设置 | 用途 |
| --- | --- |
| `plugin` | 对话模型、附加提示词与插件开关 |
| `audio` | 麦克风和输出设备 |
| `asr` | 识别模型、推理设备及是否常驻监听 |
| `activation` | 监听方式和热键 |
| `tts` | 流式或完整音频播放 |
| `voice_call` | 输入模式和静默自动挂断时间 |

`asr.enabled` 默认关闭，只控制本机麦克风常驻监听，不影响 QQ 通话的远端采音。`voice_call.input_mode` 默认 `asr`；显式设置为 `multimodal` 才会直接提交 WAV。语音模型、音色和合成服务地址在 TTS 插件中配置。

更详细的配置与旧配置迁移方法见 [配置说明](docs/configuration.md)。

## 通话与命令

在 KFC QQ 私聊中，Bot 可发起呼叫、接听或拒接来电，并请求挂断。控制 Action 的成功回执表示请求已发送，不代表 QQ 已接通或已挂断；实际状态由 QQ 通话通知更新。`/voice status` 查询当前流状态，`/voice off` 请求挂断。

通话通过 SnowLuma 已有反向消息 WebSocket 处理，不需要新增地址、端口或 token。通话音频采用 VAD 分句处理，不是 Live API。取消本地音频消费不会释放上游采音订阅；订阅会保留到 QQ 通话结束。

| 命令 | 用途 |
| --- | --- |
| `/voice status` | 查看当前通话状态 |
| `/voice off` | 手动挂断当前聊天的通话 |

这两个命令需要 Bot 所有者权限。通话 Action 仅由 `kokoro_flow_chatter` 在 QQ 私聊中使用。

## 注意事项

- QQ 原生通话与本机麦克风语音交互是两项独立功能；本机麦克风不会接管 QQ 电话的媒体链路。
- 通话状态不在 Bot 重启后恢复；拨号或控制请求结果未知时不会自动重拨。
- `asr` 模式在本地识别分句音频；`multimodal` 模式向主模型提交 WAV，模型能力由所选服务决定。
- 安装其他本机麦克风监听插件时，应避免同时监听同一设备。
- 识别文本和语音回复会进入聊天历史，使用前应告知参与者。

## 许可证

本项目采用 [GNU AGPL-3.0](LICENSE)。语音模型和依赖项目的许可证以各自说明为准。