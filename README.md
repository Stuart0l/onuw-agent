# onuw-agent

A Python framework for letting multiple LLM agents play [One Night Ultimate Werewolf](https://one-night.fandom.com/wiki/One_Night_Ultimate_Werewolf) against each other. The game engine is deterministic, rule-based code; LLMs are called only for player decisions (night actions, day-phase discussion, votes).

## Install

```bash
brew install uv     # or your preferred installer
uv sync
```

## Run a game

Set your provider API keys, then:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-...

uv run python -m onuw run --config configs/sample_game.yaml --god
```

The `--god` flag shows private events (role assignments, night actions) on the console; omit it for a player-view. A structured JSON log is written to `logs/<game_id>.json` regardless.

## Run the tests

```bash
uv run pytest -q
```

The suite runs without any API keys — it uses `ScriptedAgent`, a deterministic agent that returns canned responses.

## Plug in your own agent backend

The `Agent` ABC is intentionally framework-agnostic. Any LiteLLM, LangChain, LangGraph, LlamaIndex, AutoGen, CrewAI, DSPy, Claude Agent SDK, or custom HTTP backend can implement the same three async methods:

```python
from onuw.agents.base import Agent

class MyLangChainAgent(Agent):
    async def act_night(self, action_key, user_prompt) -> dict: ...
    async def speak(self, round_idx, user_prompt) -> str: ...
    async def vote(self, user_prompt) -> str: ...

def my_factory(player_cfg):
    return MyLangChainAgent(player_cfg.id, ...)

# Pass `my_factory` as the `agent_factory` argument to GameEngine.
```

Each agent receives a per-player `PlayerMemory` via `bind()` containing the public discussion log and that player's private night observations, so custom agents can either use the engine's rendered prompts or build their own templates from the memory object.

## Configuration

See [`configs/sample_game.yaml`](configs/sample_game.yaml). Base-game roles supported: `werewolf`, `minion`, `mason`, `seer`, `robber`, `troublemaker`, `drunk`, `insomniac`, `villager`, `tanner`, `hunter`. `len(role_pool)` must equal `len(players) + 3`; the last three cards (after the seeded shuffle) become the center cards.

## Architecture

```
src/onuw/
├── types.py / state.py / config.py    # core data model
├── memory.py                          # per-player view (the LLM's context)
├── prompts/                           # rules + per-phase task blocks
├── agents/                            # Agent ABC + LLMAgent + ScriptedAgent
├── llm/                               # LiteLLM wrapper
├── engine/                            # setup → night → day → vote → resolve
├── events/                            # EventBus + Observer + console/JSON sinks
└── cli.py                             # CLI entry
```

The engine emits events through an `EventBus` and never writes to stdout or disk directly; observers handle rendering. This is the seam a future web GUI plugs into without touching engine code.
