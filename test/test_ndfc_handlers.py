"""NDFC 桥接仅接管绑定到 anima_voice 的流。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.anima_chatter.chatter import request_factory as live_request_factory
from plugins.anima_chatter.config import AnimaChatterConfig
from plugins.anima_voice import prompts
from plugins.anima_voice.chatter import ndfc_handlers, request_factory
from plugins.anima_voice.chatter.core import AnimaVoiceChatter
from plugins.anima_voice.config import AnimaVoiceConfig
from plugins.anima_voice.runtime import call_state
from src.app.plugin_system.api import prompt_api
from src.app.plugin_system.types import ROLE, LLMPayload, PromptTemplate, Text
from src.core import prompt as core_prompt
from src.kernel.event import EventDecision


@pytest.fixture(autouse=True)
def _shared_speech_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离语音模板注册并使用可渲染变量验证 Prompt API。"""

    monkeypatch.setattr(core_prompt.get_prompt_manager(), "_templates", {})
    prompt_api.register_template(PromptTemplate(
        name=prompts.SPEECH_RULES_TEMPLATE_NAME,
        template="<speech_rules>{word}</speech_rules>",
    ).set("word", "shared speech guidance"))


@pytest.mark.parametrize("is_voice", [False, True])
@pytest.mark.parametrize("task", ["actor", "sub_actor"])
def test_request_metadata_preserves_snapshot_eligibility(
    monkeypatch: pytest.MonkeyPatch, is_voice: bool, task: str,
) -> None:
    """对话请求保持按流快照识别，直播注意力请求不覆盖对话快照。"""

    factory = request_factory if is_voice else live_request_factory
    section = (AnimaVoiceConfig() if is_voice else AnimaChatterConfig()).plugin
    monkeypatch.setattr(factory, "resolve_model_set", lambda *_: [])
    monkeypatch.setattr(factory, "default_context_compression_handler", lambda: None)
    monkeypatch.setattr(factory, "LLMRequest", lambda **kwargs: SimpleNamespace(**kwargs))
    request = factory.create_request(
        section=section, stream_id="unchanged-stream", task=task,
        request_name="custom-session", with_reminder=None,
    )
    assert request.meta_data["stream_id"] == "unchanged-stream"
    assert bool(request.meta_data.get("_ndfc")) is (is_voice or task != "sub_actor")


@pytest.mark.asyncio
async def test_fetch_unreads_records_only_current_call_user_messages_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重复拉取不重复入档，开始前消息与本方消息不计作用户发言。"""

    active = await call_state.set_active_call("voice-stream")
    plugin = SimpleNamespace(config=AnimaVoiceConfig())
    chatter = AnimaVoiceChatter("voice-stream", plugin)
    messages = [
        SimpleNamespace(message_id="before", time=active.started_at - 1,
                        is_sent=False, processed_plain_text="before", content=""),
        SimpleNamespace(message_id="user", time=active.started_at + 1,
                        is_sent=False, processed_plain_text="user voice", content=""),
        SimpleNamespace(message_id="assistant", time=active.started_at + 2,
                        is_sent=True, processed_plain_text="assistant", content=""),
    ]
    monkeypatch.setattr(
        ndfc_handlers.AnimaVoiceChatter.__bases__[0], "fetch_unreads",
        AsyncMock(return_value=("formatted", messages)),
    )
    assert await chatter.fetch_unreads() == ("formatted", messages)
    await chatter.fetch_unreads()
    assert active.messages_in_call == [
        {"role": "user", "text": "user voice", "ts": active.started_at + 1}
    ]


@pytest.mark.asyncio
async def test_fetch_unreads_does_not_record_another_stream_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本地 ASR 不应刷新另一个平台私聊的通话记录和超时。"""

    active = await call_state.set_active_call("other-stream")
    chatter = AnimaVoiceChatter("voice-stream", SimpleNamespace(config=AnimaVoiceConfig()))
    monkeypatch.setattr(
        AnimaVoiceChatter.__bases__[0], "fetch_unreads",
        AsyncMock(return_value=("formatted", [])),
    )
    await chatter.fetch_unreads()
    assert active.messages_in_call == []


@pytest.mark.asyncio
async def test_ndfc_handler_passes_for_non_voice_chatter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """绑定类型不匹配时不得修改事件参数或截断默认处理。"""

    params = {"stream_id": "stream", "proceed": False}
    monkeypatch.setattr(
        ndfc_handlers.chat_api, "get_chatter_by_stream", lambda _: object()
    )

    handler = ndfc_handlers.AnimaVoicePreprocessHandler(plugin=object())
    decision, returned = await handler.execute("neo_default_chatter:preprocess", params)

    assert decision is EventDecision.PASS
    assert returned is params
    assert params["proceed"] is False


def test_voice_system_prompt_replaces_only_system_payload() -> None:
    """语音提示替换 SYSTEM 内容时保留 NDFC USER 上下文。"""

    system = LLMPayload(ROLE.SYSTEM, Text("default system"))
    user = LLMPayload(ROLE.USER, Text("existing user context"))
    response = type("Response", (), {"payloads": [system, user]})()

    ndfc_handlers._replace_system_prompt(response, "voice system")

    assert system.content == [Text("voice system")]
    assert user.content == [Text("existing user context")]


@pytest.mark.asyncio
async def test_create_request_routes_only_voice_chatter_with_ndfc_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """create_request 转发 NDFC task、request 名与 reminder 参数。"""

    plugin = type("Plugin", (), {"config": AnimaVoiceConfig()})()
    chatter = AnimaVoiceChatter("voice-stream", plugin)
    calls: list[tuple[object, ...]] = []

    def create_request(*args: object) -> object:
        calls.append(args)
        return "voice-request"

    monkeypatch.setattr(chatter, "create_request", create_request)
    monkeypatch.setattr(
        ndfc_handlers.chat_api, "get_chatter_by_stream", lambda _: chatter
    )
    handler = ndfc_handlers.AnimaVoiceCreateRequestHandler(plugin=plugin)
    params = {
        "stream_id": "voice-stream",
        "task_name": "voice-task",
        "request_name": "voice-request-name",
        "with_reminder": "voice-reminder",
    }

    decision, returned = await handler.execute(
        "neo_default_chatter:create_request", params
    )

    assert decision is EventDecision.STOP
    assert returned["request"] == "voice-request"
    assert calls == [("voice-task", "voice-request-name", "voice-reminder")]


@pytest.mark.asyncio
async def test_voice_unread_handler_keeps_user_context_and_voice_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """语音 handler 替换 system prompt、保留已有上下文并附加 custom prompt。"""

    plugin = type("Plugin", (), {"config": AnimaVoiceConfig()})()
    _mock_core_personality(monkeypatch)
    plugin.config.plugin.custom_prompt = "Use concise spoken phrasing."
    chatter = AnimaVoiceChatter("voice-stream", plugin)
    monkeypatch.setattr(
        ndfc_handlers.chat_api, "get_chatter_by_stream", lambda _: chatter
    )
    system = LLMPayload(ROLE.SYSTEM, Text("default system"))
    context = LLMPayload(ROLE.USER, Text("earlier NDFC context"))
    response = type(
        "Response",
        (),
        {
            "payloads": [system, context],
            "add_payload": lambda self, item: self.payloads.append(item),
        },
    )()
    params = {
        "stream_id": "voice-stream",
        "response": response,
        "formatted_text": "new utterance",
    }
    handler = ndfc_handlers.AnimaVoiceInjectUnreadsHandler(plugin=plugin)

    decision, _ = await handler.execute(
        "neo_default_chatter:inject_unread_payload", params
    )

    assert decision is EventDecision.STOP
    assert "所有要说出的内容必须调用 say" in str(system.content)
    assert "Use concise spoken phrasing." in str(system.content)
    assert context.content == [Text("earlier NDFC context")]
    assert response.payloads[-1].content == [Text("new utterance")]
    assert params["skip"] is True


def _mock_core_personality(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离核心人格配置的全局初始化依赖。"""

    personality = type(
        "Personality",
        (),
        {
            "nickname": "测试昵称",
            "alias_names": ["测试别名"],
            "personality_core": "友善",
            "personality_side": "偶尔爱开玩笑",
            "identity": "助手",
            "reply_style": "简洁",
            "background_story": "来自海边",
            "safety_guidelines": ["不泄露隐私"],
            "negative_behaviors": ["不攻击他人"],
        },
    )()
    monkeypatch.setattr(
        prompts,
        "get_core_config",
        lambda: type("Config", (), {"personality": personality})(),
    )


@pytest.mark.asyncio
async def test_system_prompt_preserves_global_personality_and_custom_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通话提示保留全局人格、安全边界及附加提示词的有限作用域。"""

    _mock_core_personality(monkeypatch)
    rendered = await prompts.build_system_prompt(
        custom_prompt="用更活泼的语气",
        enable_action_suspend=True,
        call_status="<call_status>当前通话</call_status>",
    )

    for expected in (
        "测试昵称",
        "测试别名",
        "友善",
        "偶尔爱开玩笑",
        "助手",
        "简洁",
        "来自海边",
        "不泄露隐私",
        "不攻击他人",
        "用更活泼的语气",
        "__SUSPEND__",
        "当前通话",
        "ASR",
        "pass_and_wait",
        "say action",
    ):
        assert expected in rendered
    assert rendered.index("不泄露隐私") < rendered.index("用更活泼的语气")
    assert "只在不违反人格、安全准则及通话工具协议时适用" in rendered
    assert rendered.count("<speech_rules>") == 1
    assert "shared speech guidance" in rendered
    assert AnimaVoiceChatter.speech_rules_template_name == prompts.SPEECH_RULES_TEMPLATE_NAME
    assert not any(name in rendered for name in (
        "tts_segments", "merge_voice", "voice_style", "say_and_perform",
    ))


@pytest.mark.asyncio
async def test_system_prompt_honors_disabled_action_suspend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭 Action 挂起时不要求挂起标记，等待动作语义仍保留。"""

    _mock_core_personality(monkeypatch)
    rendered = await prompts.build_system_prompt(
        custom_prompt="", enable_action_suspend=False, call_status=""
    )

    assert '不要因 Action 回执输出 "__SUSPEND__"' in rendered
    assert "调用 pass_and_wait 仍会进入等待" in rendered
    assert "确认需要挂断时调用 end_voice_call" not in rendered


@pytest.mark.asyncio
async def test_voice_prompt_requires_shared_template_and_rereads_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """每次构建读取当前公共模板，模板缺失时明确失败。"""

    _mock_core_personality(monkeypatch)
    prompt_api.register_template(PromptTemplate(
        name=prompts.SPEECH_RULES_TEMPLATE_NAME,
        template="<speech_rules>updated speech rules</speech_rules>",
    ))
    rendered = await prompts.build_system_prompt(
        custom_prompt="", enable_action_suspend=True, call_status="",
    )
    assert "updated speech rules" in rendered
    assert "shared speech guidance" not in rendered
    assert rendered.count("<speech_rules>") == 1
    prompt_api.unregister_template(prompts.SPEECH_RULES_TEMPLATE_NAME)
    with pytest.raises(RuntimeError, match="公共语音模板未注册"):
        await prompts.build_system_prompt(
            custom_prompt="", enable_action_suspend=True, call_status="",
        )


@pytest.mark.asyncio
async def test_voice_prompt_renders_dynamic_call_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """活动通话提示显示开始时间、持续时长及静默超时提醒。"""

    _mock_core_personality(monkeypatch)
    plugin = type("Plugin", (), {"config": AnimaVoiceConfig()})()
    chatter = AnimaVoiceChatter("voice-stream", plugin)
    active = await call_state.set_active_call("voice-stream", timeout_seconds=125)
    active.started_at -= 3665
    active.last_activity_at -= 20
    monkeypatch.setattr(
        ndfc_handlers.chat_api, "get_chatter_by_stream", lambda _: chatter
    )
    system = LLMPayload(ROLE.SYSTEM, Text("default system"))
    response = type(
        "Response",
        (),
        {
            "payloads": [system],
            "add_payload": lambda self, item: self.payloads.append(item),
        },
    )()
    handler = ndfc_handlers.AnimaVoiceInjectUnreadsHandler(plugin=plugin)

    await handler.execute(
        "neo_default_chatter:inject_unread_payload",
        {"stream_id": "voice-stream", "response": response, "formatted_text": "turn"},
    )

    rendered = str(system.content)
    assert "通话开始于：" in rendered
    assert "已持续：1 小时" in rendered
    assert "距离静默自动挂断还剩：1 分钟" in rendered
    assert "静默超过 2 分钟 5 秒" in rendered


@pytest.mark.asyncio
async def test_other_stream_active_call_omits_status_and_hangup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """活动通话不属于当前流时不显示状态或挂断指令。"""

    _mock_core_personality(monkeypatch)
    await call_state.set_active_call("another-stream")
    plugin = type("Plugin", (), {"config": AnimaVoiceConfig()})()
    chatter = AnimaVoiceChatter("voice-stream", plugin)
    monkeypatch.setattr(
        ndfc_handlers.chat_api, "get_chatter_by_stream", lambda _: chatter
    )
    system = LLMPayload(ROLE.SYSTEM, Text("default system"))
    response = type(
        "Response",
        (),
        {
            "payloads": [system],
            "add_payload": lambda self, item: self.payloads.append(item),
        },
    )()
    handler = ndfc_handlers.AnimaVoiceInjectUnreadsHandler(plugin=plugin)

    await handler.execute(
        "neo_default_chatter:inject_unread_payload",
        {"stream_id": "voice-stream", "response": response, "formatted_text": "turn"},
    )

    rendered = str(system.content)
    assert "<call_status>" not in rendered
    assert "end_voice_call" not in rendered


@pytest.mark.asyncio
async def test_unread_payload_preserves_ndfc_user_context_and_injects_formatted_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """保留 NDFC 既有 USER 上下文，并按事件契约注入 formatted_text。"""

    _mock_core_personality(monkeypatch)
    plugin = type("Plugin", (), {"config": AnimaVoiceConfig()})()
    chatter = AnimaVoiceChatter("voice-stream", plugin)
    monkeypatch.setattr(
        ndfc_handlers.chat_api, "get_chatter_by_stream", lambda _: chatter
    )
    system = LLMPayload(ROLE.SYSTEM, Text("default system"))
    context = LLMPayload(ROLE.USER, Text("history and clear boundary"))
    response = type(
        "Response",
        (),
        {
            "payloads": [system, context],
            "add_payload": lambda self, item: self.payloads.append(item),
        },
    )()
    handler = ndfc_handlers.AnimaVoiceInjectUnreadsHandler(plugin=plugin)

    await handler.execute(
        "neo_default_chatter:inject_unread_payload",
        {
            "stream_id": "voice-stream",
            "response": response,
            "formatted_text": "formatted NDFC turn",
        },
    )

    assert context.content == [Text("history and clear boundary")]
    assert response.payloads[-1].content == [Text("formatted NDFC turn")]
    assert sum(payload.role == ROLE.SYSTEM for payload in response.payloads) == 1


def test_request_factory_preserves_model_parameters_and_reminder_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """请求保留自定义模型参数、流 metadata 及全局和流私有 reminder。"""

    model_calls: list[tuple[object, ...]] = []
    compression_handler = object()

    def get_model(
        name: str, *, temperature: float, max_tokens: int
    ) -> list[dict[str, str]]:
        model_calls.append((name, temperature, max_tokens))
        return [{"model": name}]

    monkeypatch.setattr(request_factory, "get_model_set_by_name", get_model)
    monkeypatch.setattr(
        request_factory,
        "default_context_compression_handler",
        lambda: compression_handler,
    )
    monkeypatch.setattr(
        request_factory,
        "stream_reminder_bucket",
        lambda stream_id, bucket: f"{stream_id}:{bucket}",
    )
    config = AnimaVoiceConfig().plugin
    config.models = ["voice-model"]
    config.temperature = 0.35
    config.max_tokens = 987

    request = request_factory.create_request(
        section=config,
        stream_id="voice-stream",
        task="voice-task",
        request_name="ndfc-voice-request",
        with_reminder="voice-reminder",
    )

    assert model_calls == [("voice-model", 0.35, 987)]
    assert request.model_set == [{"model": "voice-model"}]
    assert request.meta_data == {"stream_id": "voice-stream", "_ndfc": True}
    assert request.request_name == "ndfc-voice-request"
    assert request.context_manager.context_compression_handler is compression_handler
    assert [source.bucket for source in request.context_manager.reminder_sources] == [
        "voice-reminder",
        "voice-stream:voice-reminder",
    ]
