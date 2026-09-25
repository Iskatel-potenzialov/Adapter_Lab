"""Backend configuration loaded from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ENV_FILE = Path(__file__).with_name(".env")


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs without overriding process environment."""
    if not path.is_file():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _integer(name: str, default: int, minimum: int = 0) -> int:
    value = os.getenv(name, str(default))
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if number < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return number


@dataclass(frozen=True)
class Settings:
    model_path: str
    host: str
    port: int
    frontend_origin: str
    idle_unload_seconds: int


def load_settings() -> Settings:
    load_env_file()
    return Settings(
        model_path=os.getenv("MODEL_PATH", ""),
        host=os.getenv("HOST", "0.0.0.0"),
        port=_integer("PORT", 8000, minimum=1),
        frontend_origin=os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
        idle_unload_seconds=_integer("IDLE_UNLOAD_SECONDS", 60),
    )


settings = load_settings()


def model_path_readiness() -> dict[str, bool]:
    path = Path(settings.model_path) if settings.model_path else None
    return {
        "model_path_configured": path is not None,
        "model_path_exists": bool(path and path.is_dir()),
    }
