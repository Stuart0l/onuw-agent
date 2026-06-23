import pytest

from onuw.agents.llm_agent import LLMAgent
from onuw.llm.client import LLMClient
from onuw.types import Role

pytestmark = pytest.mark.asyncio


SEAT_ORDER = ["p1", "p2", "p3"]


def _bind(agent: LLMAgent, role: Role = Role.VILLAGER) -> LLMAgent:
    agent.bind(
        name="Alice", seat=0, dealt_role=role,
        persona=None, seat_order=SEAT_ORDER,
    )
    return agent


class _FakeMessage:
    def __init__(self, text: str):
        self.content = text


class _FakeChoice:
    def __init__(self, text: str):
        self.message = _FakeMessage(text)


class _FakeResp:
    def __init__(self, text: str):
        self.choices = [_FakeChoice(text)]


class FakeClient(LLMClient):
    """LLMClient subclass that returns canned strings instead of calling litellm."""

    def __init__(self, responses: list[str]):
        super().__init__(max_retries=0)
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def _acompletion(self, kwargs: dict):
        self.calls.append(kwargs)
        text = self._responses.pop(0) if self._responses else ""
        return _FakeResp(text)


async def test_act_night_parses_json():
    client = FakeClient(['{"action": "rob", "target": "p3"}'])
    agent = _bind(LLMAgent("p1", model="x", client=client), role=Role.ROBBER)
    out = await agent.act_night("robber", valid_targets=["p2", "p3"])
    assert out == {"action": "rob", "target": "p3"}
    assert len(client.calls) == 1


async def test_act_night_retries_on_garbage_and_returns_empty_on_double_fail():
    client = FakeClient(["not json at all", "still not json"])
    agent = _bind(LLMAgent("p1", model="x", client=client), role=Role.ROBBER)
    out = await agent.act_night("robber", valid_targets=["p2", "p3"])
    assert out == {}
    assert len(client.calls) == 2
    retry_user_msg = client.calls[1]["messages"][1]["content"]
    assert "VALID JSON" in retry_user_msg


async def test_act_night_strips_markdown_fences():
    client = FakeClient(['```json\n{"action": "swap_center", "index": 1}\n```'])
    agent = _bind(LLMAgent("p1", model="x", client=client), role=Role.DRUNK)
    out = await agent.act_night("drunk", valid_targets=[])
    assert out == {"action": "swap_center", "index": 1}


async def test_speak_returns_speech_field():
    client = FakeClient(['{"speech": "I think p2 is the wolf."}'])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    text = await agent.speak(round_idx=0, total_rounds=3, max_chars=600)
    assert text == "I think p2 is the wolf."


async def test_speak_returns_empty_on_double_failure_for_engine_localization():
    # Empty string signals the engine to substitute the localized
    # "nothing to add" fallback per state.language.
    client = FakeClient(["garbage", "still garbage"])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    text = await agent.speak(0, 3, 600)
    assert text == ""


async def test_vote_returns_vote_field():
    client = FakeClient(['{"vote": "p3", "rationale": "they\'re sus"}'])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    target = await agent.vote(valid_targets=SEAT_ORDER)
    assert target == "p3"


async def test_vote_falls_back_to_self_on_double_failure():
    client = FakeClient(["nope", "nope"])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    target = await agent.vote(valid_targets=SEAT_ORDER)
    assert target == "p1"


async def test_system_prompt_is_built_at_bind_and_passed_through():
    client = FakeClient(['{"vote": "p2"}'])
    agent = _bind(LLMAgent("p1", model="gpt-4o", client=client), role=Role.ROBBER)
    await agent.vote(valid_targets=SEAT_ORDER)
    msgs = client.calls[0]["messages"]
    assert msgs[0]["role"] == "system"
    # Built from build_system_prompt(role=Robber): contains the role block.
    assert "ROBBER" in msgs[0]["content"].upper()
    assert msgs[1]["role"] == "user"
    # User prompt comes from the agent's internal memory + vote task.
    assert "VOTING PHASE" in msgs[1]["content"]


async def test_json_mode_true_sends_response_format():
    client = FakeClient(['{"vote": "p2"}'])
    agent = _bind(LLMAgent("p1", model="gpt-4o", client=client, json_mode=True))
    await agent.vote(valid_targets=SEAT_ORDER)
    assert client.calls[0]["response_format"] == {"type": "json_object"}


async def test_json_mode_false_omits_response_format():
    client = FakeClient(['{"vote": "p2"}'])
    agent = _bind(LLMAgent("p1", model="gpt-4o", client=client))
    await agent.vote(valid_targets=SEAT_ORDER)
    assert "response_format" not in client.calls[0]


async def test_extra_body_is_forwarded_when_set():
    client = FakeClient(['{"vote": "p2"}'])
    agent = _bind(LLMAgent(
        "p1", model="x", client=client,
        extra_body={"thinking": {"type": "disabled"}},
    ))
    await agent.vote(valid_targets=SEAT_ORDER)
    assert client.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_extra_body_is_omitted_by_default():
    client = FakeClient(['{"vote": "p2"}'])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    await agent.vote(valid_targets=SEAT_ORDER)
    assert "extra_body" not in client.calls[0]


# ----- token usage -----

class _FakeUsage:
    def __init__(self, prompt: int, completion: int):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _FakeRespWithUsage(_FakeResp):
    def __init__(self, text: str, prompt: int = 0, completion: int = 0):
        super().__init__(text)
        self.usage = _FakeUsage(prompt, completion)


class FakeClientWithUsage(LLMClient):
    def __init__(self, responses):
        super().__init__(max_retries=0)
        self._responses = list(responses)

    async def _acompletion(self, kwargs):
        return self._responses.pop(0) if self._responses else _FakeResp("")


async def test_token_usage_accumulates_across_calls():
    client = FakeClientWithUsage([
        _FakeRespWithUsage('{"vote": "p2"}', prompt=100, completion=20),
        _FakeRespWithUsage('{"speech": "hi"}', prompt=150, completion=30),
    ])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    await agent.vote(valid_targets=SEAT_ORDER)
    await agent.speak(0, 3, 600)
    assert agent.token_usage.prompt_tokens == 250
    assert agent.token_usage.completion_tokens == 50
    assert agent.token_usage.total_tokens == 300


async def test_token_usage_stays_zero_when_response_has_no_usage_field():
    client = FakeClient(['{"vote": "p2"}'])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    await agent.vote(valid_targets=SEAT_ORDER)
    assert agent.token_usage.total_tokens == 0


async def test_scripted_agent_has_zero_token_usage():
    from onuw.agents.scripted_agent import ScriptedAgent
    s = ScriptedAgent("p1", vote="p2")
    await s.vote(valid_targets=SEAT_ORDER)
    assert s.token_usage.total_tokens == 0


# ----- reasoning content / streaming -----

class _FakeDelta:
    def __init__(self, content: str = "", reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeStreamChoice:
    def __init__(self, content: str = "", reasoning_content: str = ""):
        self.delta = _FakeDelta(content, reasoning_content)


class _FakeStreamChunk:
    def __init__(
        self,
        content: str = "",
        reasoning_content: str = "",
        usage=None,
    ):
        self.choices = [_FakeStreamChoice(content, reasoning_content)]
        self.usage = usage


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class FakeStreamingClient(LLMClient):
    def __init__(self, chunks):
        super().__init__(max_retries=0)
        self._chunks = chunks
        self.calls: list[dict] = []

    async def _acompletion(self, kwargs):
        self.calls.append(kwargs)
        return _FakeStream(self._chunks)


async def test_reasoning_content_streams_as_chunks():
    from onuw.events.bus import EventBus, ReasoningChunkEvent
    captured: list = []
    bus = EventBus([type("C", (), {"on_event": lambda self, e: captured.append(e)})()])
    client = FakeStreamingClient([
        _FakeStreamChunk(reasoning_content="Hmm, "),
        _FakeStreamChunk(reasoning_content="p2 has been suspicious"),
        _FakeStreamChunk(content='{"vote": "p2"}'),
    ])
    agent = LLMAgent("p1", model="x", client=client)
    agent.bind(
        name="Alice", seat=0, dealt_role=Role.VILLAGER,
        persona=None, seat_order=SEAT_ORDER, bus=bus,
    )
    await agent.vote(valid_targets=SEAT_ORDER)
    chunks = [e for e in captured if isinstance(e, ReasoningChunkEvent)]
    assert "".join(c.delta for c in chunks) == "Hmm, p2 has been suspicious"


async def test_reasoning_falls_back_to_reasoning_attribute_on_delta():
    from onuw.events.bus import EventBus, ReasoningChunkEvent

    class _AltDelta:
        def __init__(self, content="", reasoning=""):
            self.content = content
            self.reasoning = reasoning  # NOT reasoning_content

    class _AltChoice:
        def __init__(self, content="", reasoning=""):
            self.delta = _AltDelta(content, reasoning)

    class _AltChunk:
        def __init__(self, content="", reasoning="", usage=None):
            self.choices = [_AltChoice(content, reasoning)]
            self.usage = usage

    captured: list = []
    bus = EventBus([type("C", (), {"on_event": lambda self, e: captured.append(e)})()])
    client = FakeStreamingClient([
        _AltChunk(reasoning="alt-style trace"),
        _AltChunk(content='{"vote": "p2"}'),
    ])
    agent = LLMAgent("p1", model="x", client=client)
    agent.bind(
        name="Alice", seat=0, dealt_role=Role.VILLAGER,
        persona=None, seat_order=SEAT_ORDER, bus=bus,
    )
    await agent.vote(valid_targets=SEAT_ORDER)
    chunks = [e for e in captured if isinstance(e, ReasoningChunkEvent)]
    assert chunks and "alt-style trace" in "".join(c.delta for c in chunks)


async def test_inline_think_tags_split_into_reasoning_and_content_chunks():
    from onuw.events.bus import ContentChunkEvent, EventBus, ReasoningChunkEvent
    captured: list = []
    bus = EventBus([type("C", (), {"on_event": lambda self, e: captured.append(e)})()])
    client = FakeStreamingClient([
        _FakeStreamChunk(content="<think>p3 said X "),
        _FakeStreamChunk(content="so likely Y"),
        _FakeStreamChunk(content='</think>\n{"vote": "p3"}'),
    ])
    agent = LLMAgent("p1", model="x", client=client)
    agent.bind(
        name="Alice", seat=0, dealt_role=Role.VILLAGER,
        persona=None, seat_order=SEAT_ORDER, bus=bus,
    )
    target = await agent.vote(valid_targets=SEAT_ORDER)
    assert target == "p3"
    reasoning_chunks = [e for e in captured if isinstance(e, ReasoningChunkEvent)]
    assert "".join(c.delta for c in reasoning_chunks) == "p3 said X so likely Y"
    content_chunks = [e for e in captured if isinstance(e, ContentChunkEvent)]
    joined_content = "".join(c.delta for c in content_chunks)
    assert "<think>" not in joined_content
    assert "</think>" not in joined_content
    assert '{"vote": "p3"}' in joined_content


async def test_no_reasoning_means_no_reasoning_chunk():
    from onuw.events.bus import EventBus, ReasoningChunkEvent
    captured: list = []
    bus = EventBus([type("C", (), {"on_event": lambda self, e: captured.append(e)})()])
    client = FakeStreamingClient([
        _FakeStreamChunk(content='{"vote": "p2"}'),
    ])
    agent = LLMAgent("p1", model="x", client=client)
    agent.bind(
        name="Alice", seat=0, dealt_role=Role.VILLAGER,
        persona=None, seat_order=SEAT_ORDER, bus=bus,
    )
    await agent.vote(valid_targets=SEAT_ORDER)
    assert not [e for e in captured if isinstance(e, ReasoningChunkEvent)]


async def test_streaming_emits_reasoning_chunks_in_order():
    from onuw.events.bus import EventBus, ReasoningChunkEvent
    captured: list = []
    bus = EventBus([type("C", (), {"on_event": lambda self, e: captured.append(e)})()])
    client = FakeStreamingClient([
        _FakeStreamChunk(reasoning_content="First "),
        _FakeStreamChunk(reasoning_content="second "),
        _FakeStreamChunk(reasoning_content="third."),
        _FakeStreamChunk(content='{"vote": "p3"}', usage=_FakeUsage(120, 30)),
    ])
    agent = LLMAgent("p1", model="x", client=client)
    agent.bind(
        name="Alice", seat=0, dealt_role=Role.VILLAGER,
        persona=None, seat_order=SEAT_ORDER, bus=bus,
    )
    target = await agent.vote(valid_targets=SEAT_ORDER)
    assert target == "p3"
    assert client.calls[0]["stream"] is True
    assert client.calls[0]["stream_options"] == {"include_usage": True}
    chunks = [e for e in captured if isinstance(e, ReasoningChunkEvent)]
    assert [c.delta for c in chunks] == ["First ", "second ", "third."]
    assert agent.token_usage.total_tokens == 150


async def test_non_streaming_path_when_no_bus_is_bound():
    client = FakeClient(['{"vote": "p2"}'])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    target = await agent.vote(valid_targets=SEAT_ORDER)
    assert target == "p2"


# ----- belief state extraction -----

async def test_speak_extracts_belief_state_into_memory():
    client = FakeClient([
        '{"belief_state": {"p2": "likely Werewolf", "p3": "Mason"}, '
        '"speech": "I think p2 is the wolf."}'
    ])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    text = await agent.speak(round_idx=0, total_rounds=3, max_chars=600)
    assert text == "I think p2 is the wolf."
    assert agent.memory is not None
    assert agent.memory.belief_state == {
        "p2": "likely Werewolf",
        "p3": "Mason",
    }


async def test_speak_without_belief_state_in_response_preserves_prior_state():
    # First call sets beliefs; second call's response has no belief_state.
    client = FakeClient([
        '{"belief_state": {"p2": "wolf"}, "speech": "round 1 speech"}',
        '{"speech": "round 2 speech"}',
    ])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    await agent.speak(0, 3, 600)
    await agent.speak(1, 3, 600)
    assert agent.memory.belief_state == {"p2": "wolf"}


async def test_belief_state_renders_into_next_prompt():
    # First speak() seeds beliefs. The user prompt sent on the SECOND
    # speak() should include the previously-stored belief.
    client = FakeClient([
        '{"belief_state": {"p2": "bluffing Robber"}, "speech": "hi"}',
        '{"speech": "round 2 reply"}',
    ])
    agent = _bind(LLMAgent("p1", model="x", client=client))
    await agent.speak(0, 3, 600)
    await agent.speak(1, 3, 600)
    second_user_msg = client.calls[1]["messages"][1]["content"]
    assert "== YOUR PRIVATE BELIEFS" in second_user_msg
    assert "bluffing Robber" in second_user_msg
