import pytest

from onuw.agents.llm_agent import LLMAgent
from onuw.llm.client import LLMClient

pytestmark = pytest.mark.asyncio


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
    agent = LLMAgent("p1", model="x", client=client)
    out = await agent.act_night("robber", "user prompt")
    assert out == {"action": "rob", "target": "p3"}
    assert len(client.calls) == 1


async def test_act_night_retries_on_garbage_and_returns_empty_on_double_fail():
    client = FakeClient(["not json at all", "still not json"])
    agent = LLMAgent("p1", model="x", client=client)
    out = await agent.act_night("robber", "user prompt")
    assert out == {}
    assert len(client.calls) == 2
    # Retry includes the "IMPORTANT" reminder
    retry_user_msg = client.calls[1]["messages"][1]["content"]
    assert "VALID JSON" in retry_user_msg


async def test_act_night_strips_markdown_fences():
    client = FakeClient(['```json\n{"action": "swap_center", "index": 1}\n```'])
    agent = LLMAgent("p1", model="x", client=client)
    out = await agent.act_night("drunk", "u")
    assert out == {"action": "swap_center", "index": 1}


async def test_speak_returns_speech_field():
    client = FakeClient(['{"speech": "I think p2 is the wolf."}'])
    agent = LLMAgent("p1", model="x", client=client)
    text = await agent.speak(round_idx=0, user_prompt="u")
    assert text == "I think p2 is the wolf."


async def test_speak_falls_back_to_safe_string_on_double_failure():
    client = FakeClient(["garbage", "still garbage"])
    agent = LLMAgent("p1", model="x", client=client)
    text = await agent.speak(0, "u")
    assert text == "I have nothing to add."


async def test_vote_returns_vote_field():
    client = FakeClient(['{"vote": "p3", "rationale": "they\'re sus"}'])
    agent = LLMAgent("p1", model="x", client=client)
    target = await agent.vote("u")
    assert target == "p3"


async def test_vote_falls_back_to_self_on_double_failure():
    client = FakeClient(["nope", "nope"])
    agent = LLMAgent("p1", model="x", client=client)
    target = await agent.vote("u")
    assert target == "p1"


async def test_system_prompt_is_passed_through():
    client = FakeClient(['{"vote": "p2"}'])
    agent = LLMAgent("p1", model="gpt-4o", client=client)
    agent.system_prompt = "YOU ARE THE ROBBER."
    await agent.vote("vote prompt")
    msgs = client.calls[0]["messages"]
    assert msgs[0]["role"] == "system"
    assert "ROBBER" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "vote prompt"


async def test_json_mode_true_sends_response_format():
    client = FakeClient(['{"vote": "p2"}'])
    agent = LLMAgent("p1", model="gpt-4o", client=client, json_mode=True)
    await agent.vote("u")
    assert client.calls[0]["response_format"] == {"type": "json_object"}


async def test_json_mode_false_omits_response_format():
    # Default mode: response_format is NOT sent. Required for cross-
    # provider compatibility — LM Studio rejects {"type":"json_object"}.
    client = FakeClient(['{"vote": "p2"}'])
    agent = LLMAgent("p1", model="gpt-4o", client=client)
    await agent.vote("u")
    assert "response_format" not in client.calls[0]


async def test_extra_body_is_forwarded_when_set():
    client = FakeClient(['{"vote": "p2"}'])
    agent = LLMAgent(
        "p1", model="x", client=client,
        extra_body={"thinking": {"type": "disabled"}},
    )
    await agent.vote("u")
    assert client.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_extra_body_is_omitted_by_default():
    client = FakeClient(['{"vote": "p2"}'])
    agent = LLMAgent("p1", model="x", client=client)
    await agent.vote("u")
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
    agent = LLMAgent("p1", model="x", client=client)
    await agent.vote("u1")
    await agent.speak(0, "u2")
    assert agent.token_usage.prompt_tokens == 250
    assert agent.token_usage.completion_tokens == 50
    assert agent.token_usage.total_tokens == 300


async def test_token_usage_stays_zero_when_response_has_no_usage_field():
    client = FakeClient(['{"vote": "p2"}'])
    agent = LLMAgent("p1", model="x", client=client)
    await agent.vote("u")
    assert agent.token_usage.prompt_tokens == 0
    assert agent.token_usage.completion_tokens == 0
    assert agent.token_usage.total_tokens == 0


async def test_scripted_agent_has_zero_token_usage():
    from onuw.agents.scripted_agent import ScriptedAgent
    s = ScriptedAgent("p1", vote="p2")
    await s.vote("u")
    assert s.token_usage.total_tokens == 0


# ----- reasoning content -----

class _FakeMessageWithReasoning:
    def __init__(self, content: str, reasoning_content: str = "", reasoning: str = ""):
        self.content = content
        if reasoning_content:
            self.reasoning_content = reasoning_content
        if reasoning:
            self.reasoning = reasoning


class _FakeChoiceWithReasoning:
    def __init__(self, content: str, reasoning_content: str = "", reasoning: str = ""):
        self.message = _FakeMessageWithReasoning(content, reasoning_content, reasoning)


class _FakeRespWithReasoning:
    def __init__(self, content: str, reasoning_content: str = "", reasoning: str = ""):
        self.choices = [_FakeChoiceWithReasoning(content, reasoning_content, reasoning)]


class FakeClientCapturing(LLMClient):
    def __init__(self, responses):
        super().__init__(max_retries=0)
        self._responses = list(responses)

    async def _acompletion(self, kwargs):
        return self._responses.pop(0) if self._responses else _FakeResp("")


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
    agent.bus = bus
    await agent.vote("u")
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
    agent.bus = bus
    await agent.vote("u")
    chunks = [e for e in captured if isinstance(e, ReasoningChunkEvent)]
    assert chunks and "alt-style trace" in "".join(c.delta for c in chunks)


async def test_inline_think_tags_split_into_reasoning_and_content_chunks():
    # The splitter strips <think>...</think> from content deltas live
    # and routes the wrapped bytes to reasoning chunks instead, so the
    # user never sees the tag text in the displayed stream.
    from onuw.events.bus import ContentChunkEvent, EventBus, ReasoningChunkEvent
    captured: list = []
    bus = EventBus([type("C", (), {"on_event": lambda self, e: captured.append(e)})()])
    client = FakeStreamingClient([
        _FakeStreamChunk(content="<think>p3 said X "),
        _FakeStreamChunk(content="so likely Y"),
        _FakeStreamChunk(content='</think>\n{"vote": "p3"}'),
    ])
    agent = LLMAgent("p1", model="x", client=client)
    agent.bus = bus
    target = await agent.vote("u")
    assert target == "p3"
    reasoning_chunks = [e for e in captured if isinstance(e, ReasoningChunkEvent)]
    assert "".join(c.delta for c in reasoning_chunks) == "p3 said X so likely Y"
    content_chunks = [e for e in captured if isinstance(e, ContentChunkEvent)]
    # Only the post-</think> bytes reach the content stream — no tags.
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
    agent.bus = bus
    await agent.vote("u")
    assert not [e for e in captured if isinstance(e, ReasoningChunkEvent)]


# ----- streaming -----

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
        usage: "_FakeUsage | None" = None,
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
    agent.bus = bus
    target = await agent.vote("u")
    assert target == "p3"
    assert client.calls[0]["stream"] is True
    assert client.calls[0]["stream_options"] == {"include_usage": True}
    chunks = [e for e in captured if isinstance(e, ReasoningChunkEvent)]
    assert [c.delta for c in chunks] == ["First ", "second ", "third."]
    # Usage from the terminal chunk should accumulate.
    assert agent.token_usage.total_tokens == 150


async def test_non_streaming_path_when_no_bus_is_bound():
    # No bus => no observer => streaming is not enabled.
    client = FakeClientCapturing([
        _FakeRespWithReasoning('{"vote": "p2"}', reasoning_content="quick"),
    ])
    agent = LLMAgent("p1", model="x", client=client)
    target = await agent.vote("u")
    assert target == "p2"
    # We never passed stream=True down to the client when bus is absent.
    # FakeClientCapturing doesn't record calls, so just confirm behavior:
    # the agent succeeded without raising — no streaming machinery used.