from __future__ import annotations

try:
    from openenv.core.env_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError("Install dependencies with `uv sync` before starting the server.") from exc

from models import ChaosAgentAction, ChaosAgentObservation
from server.environment import ChaosAgentEnvironment


app = create_app(
    ChaosAgentEnvironment,
    ChaosAgentAction,
    ChaosAgentObservation,
    env_name="chaosagent",
    max_concurrent_envs=4,
)


def main() -> None:
    """Run the OpenEnv FastAPI server; local validation looks for main()."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
