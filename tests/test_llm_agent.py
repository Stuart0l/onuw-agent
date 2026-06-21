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