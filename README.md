# onuw-agent

A Python framework for letting multiple LLM agents play [One Night Ultimate Werewolf](https://one-night.fandom.com/wiki/One_Night_Ultimate_Werewolf) against each other. The game engine is deterministic, rule-based code; LLMs are called only for player decisions (night actions, day-phase discussion, votes).

## Install

```bash
brew install uv     # or your preferred installer
uv sync
```

## Run a game

Set your provider API keys (or use a `.env` file in the project root — it's auto-loaded), then:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...

uv run python -m onuw run --config configs/sample_game.yaml --god
```

The `--god` flag shows private events (role assignments, night actions, streamed reasoning chunks) on the console; omit it for a player-view. A structured JSON log is written to `logs/<game_id>.json` regardless.

## Run the tests

```bash
uv run pytest -q
```

The suite runs without any API keys — it uses `ScriptedAgent`, a deterministic agent that returns canned responses.

## Plug in your own agent backend

The `Agent` ABC is intentionally framework-agnostic. Any LiteLLM, LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI, DSPy, Claude Agent SDK, or custom HTTP backend can implement it. The engine pushes world events via `observe_*` (default no-op) and pulls decisions via the three abstract `act_*` methods. Agents own their own private memory and prompt rendering.

```python
from onuw.agents.base import Agent
from onuw.types import Role


class MyAgent(Agent):
    # ---- Decisions (required) ----
    async def act_night(self, action_key: str, valid_targets: list[str]) -> dict:
        # action_key is one of: "werewolf_solo", "seer", "robber",
        # "troublemaker", "drunk". Return the action JSON.
        ...

    async def speak(self, round_idx: int, total_rounds: int, max_chars: int) -> str:
        # Return your public statement for this round.
        ...

    async def vote(self, valid_targets: list[str]) -> str:
        # Return the player_id you vote for.
        ...

    # ---- Observations (optional, default no-op) ----
    def observe_night(self, step: str, text: str, structured: dict) -> None: ...
    def observe_speech(self, round_idx: int, speaker_id: str, text: str) -> None: ...
    def observe_votes(self, votes: dict[str, str]) -> None: ...
    def observe_deaths(self, deaths: list[str], hunter_revenge: list[tuple[str, str]]) -> None: ...


def my_factory(player_cfg):
    return MyAgent(player_cfg.id)


# Hand `my_factory` to GameEngine as the agent_factory argument.
```

At game start the engine calls `agent.bind(name, seat, dealt_role, seat_order, role_pool, language, bus)` so the agent has every public fact it needs (player roster, deck composition, its own dealt role, etc.). Each agent decides what to remember and how to render its prompts; the engine is content with the structured action / speech / vote it returns.

The shipped `LLMAgent` (default factory) holds a private `PlayerMemory`, builds its own system + user prompts, and streams reasoning + content chunks live via `ReasoningChunkEvent` / `ContentChunkEvent`. A custom agent is free to ignore all of this.

## Configuration

See [`configs/sample_game.yaml`](configs/sample_game.yaml). Base-game roles supported: `werewolf`, `minion`, `mason`, `seer`, `robber`, `troublemaker`, `drunk`, `insomniac`, `villager`, `tanner`, `hunter`. `len(role_pool)` must equal `len(players) + 3`; the last three cards (after the seeded shuffle) become the center cards.

Per-player tunables:

| field | purpose |
|---|---|
| `model` | LiteLLM model string, e.g. `gpt-4o`, `claude-sonnet-4-5`, `openai/MiniMax-M3` |
| `temperature` | 0.0–2.0 |
| `max_tokens` | reasoning models need 4k+; non-reasoning models can stay around 800 |
| `json_mode` | `true` only on providers that accept `response_format={"type":"json_object"}` (OpenAI yes, LM Studio no) |
| `extra_body` | provider-specific request body (e.g. `thinking: {type: disabled}` for MiniMax) |

## Architecture

```
src/onuw/
├── types.py / state.py / config.py    # core data model
├── memory.py                          # PlayerMemory — owned by each agent
├── prompts/                           # rules + per-phase task blocks
├── agents/                            # Agent ABC + LLMAgent + ScriptedAgent
├── llm/                               # LiteLLM wrapper + stream splitter
├── engine/                            # setup → night → day → vote → resolve
├── events/                            # EventBus + Observer + Console/JSON sinks
└── cli.py                             # CLI entry
```

The engine emits events through an `EventBus` and never writes to stdout or disk directly; observers handle rendering. This is the seam a future web GUI plugs into without touching engine code.