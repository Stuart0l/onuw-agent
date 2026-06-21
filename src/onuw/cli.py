import argparse
import asyncio
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .agents import default_factory
from .config import GameConfig
from .engine.engine import GameEngine
from .events.bus import EventBus
from .events.console import ConsoleObserver
from .events.json_log import JsonObserver
from .llm.client import LLMClient


def main(argv: list[str] | None = None) -> int:
    # Load .env from the current working directory (if present) so users
    # can keep OPENAI_API_KEY / OPENAI_API_BASE / etc. out of their shell.
    load_dotenv()
    parser = argparse.ArgumentParser(prog="onuw")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Play one game from a config file")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument(
        "--god",
        action="store_true",
        help="Print private events (role assignments, night actions) "
        "to the console.",
    )
    run.add_argument(
        "--no-console",
        action="store_true",
        help="Suppress console output (JSON log is still written).",
    )
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = GameConfig.model_validate(yaml.safe_load(args.config.read_text()))

    observers = []
    if not args.no_console:
        observers.append(ConsoleObserver(god=args.god))
    observers.append(JsonObserver(cfg.log_dir))
    bus = EventBus(observers)

    client = LLMClient()
    engine = GameEngine(cfg, bus, agent_factory=default_factory(client))
    asyncio.run(engine.run())
    return 0
