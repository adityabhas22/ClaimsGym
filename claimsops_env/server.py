from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from claimsops_env.environment import ClaimsOpsEnv


class ResetRequest(BaseModel):
    seed: int | None = None
    scenario_family: str | None = None


class StepRequest(BaseModel):
    action: dict[str, Any] | str = Field(..., description="Strict JSON tool action")


def create_app() -> FastAPI:
    app = FastAPI(title="ClaimsOps Gym", version="0.1.0")
    env = ClaimsOpsEnv()

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"name": "claimsops-gym", "docs": "/docs", "metadata": "/metadata"}

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        return env.get_metadata()

    @app.post("/reset")
    def reset(request: ResetRequest | None = None) -> dict[str, Any]:
        request = request or ResetRequest()
        observation = env.reset(seed=request.seed, scenario_family=request.scenario_family)
        return {"observation": observation.model_dump(mode="json")}

    @app.post("/step")
    def step(request: StepRequest) -> dict[str, Any]:
        result = env.step(request.action)
        return result.model_dump(mode="json")

    @app.get("/state")
    def state() -> dict[str, Any]:
        return env.state()

    return app


app = create_app()


def main() -> None:
    uvicorn.run("claimsops_env.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
