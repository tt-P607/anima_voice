# 架构

语音通话插件只负责实时语音与平台私聊接管，不依赖 VTS、直播排播或提供方实现。`AnimaVoiceChatter` 通过 `neo_default_chatter` 的公开 ChatCore service 创建会话；NDFC event handler 仅在流当前绑定到该 Chatter 时处理，否则返回 `PASS`。

平台私聊接管期间，插件保存此前 Chatter 实例，启动内置麦克风识别后临时绑定 `anima_voice` 并重启流驱动。识别结果使用真实来电用户身份提交到原平台、原流。结束时清除重定向；未开启常驻监听时释放输入设备，只在当前绑定仍是本插件时恢复原实例；其他组件若已替换绑定，本插件不覆盖其变更。local_asr 流直接由本 Chatter 处理。

通话只有一个插件内活动槽位。状态记录通话消息与静默计时，结束快照用于生命周期事件。未读用户消息只入档一次，其他流不刷新该通话超时。请求沿用原流 ID 并携带 NDFC 标记，支持已有快照插件识别。

TTS 通过 `tts_voice_plugin-neo:service:speech` 的 `get_capabilities()`、`synthesize(request)` 和 `open_pcm_stream(request)` 调用共享合成 runtime。消费者不创建客户端、不切换权重锁、不关闭共享服务；请求上下文仅管理本次 PCM 响应。不存在 HTTP 中间层或提供方注册中心。

语音 SYSTEM 构建器异步调用公开 `prompt_api.get_template("tts_voice_plugin-neo.speech_rules")`
获取副本并渲染公共语音规则，每轮重新读取；TTS 插件未启用或模板未注册时明确失败，不复制备用规则。
ASR 理解、`say`、`[wait:n]`、等待与挂断协议仍由通话插件提供。
`AnimaVoiceChatter.speech_rules_template_name` 声明 SYSTEM 中已包含该模板，
TTS 根据当前流的绑定跳过重复提醒，切回普通 Chatter 后按原配置恢复提醒；其他插件提醒保持不变。

内置适配器与平台通话共用插件唯一播放器。PCM 接收任务在设备播放期间缓存当前回复并预取下一段；单个设备流顺序写入音频和等待静音。播放期间麦克风回采被抑制，结束后保留短尾窗。本地完播语音经公开 Message API 进入媒体缓存与消息历史，以消息 ID 避免重复播放；完整 WAV 外发仍沿框架语音消息链。

识别器首次推理才加载 FunASR；未启用常驻时只在接通后打开输入设备。异步工作由框架 TaskManager 管理。取消或卸载等待本插件 PCM 接收任务和音频线程退出，不调用全局音频停止，也不影响直播的 TTS 请求。