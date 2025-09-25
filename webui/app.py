from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .session import MinesweeperSession


class ClickRequest(BaseModel):
    row: int
    col: int


class FlagRequest(BaseModel):
    row: int
    col: int


class NewGameRequest(BaseModel):
    seed: Optional[int] = None


def _default_checkpoint() -> Path:
    for env_var in ("MINESWEEPER_CKPT_16", "MINESWEEPER_CKPT"):
        env_path = os.environ.get(env_var)
        if env_path:
            return Path(env_path)
    return Path("runs/scaling16_medium_u4000/ckpt_final.pt")


app = FastAPI()
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.on_event("startup")
def _load_session() -> None:
    ckpt_path = _default_checkpoint()
    if not ckpt_path.exists():
        raise RuntimeError(
            f"Checkpoint path not found: {ckpt_path}. Set MINESWEEPER_CKPT_16 env var."
        )
    app.state.session = MinesweeperSession(ckpt_path)


def _get_session() -> MinesweeperSession:
    session = getattr(app.state, "session", None)
    if session is None:
        raise HTTPException(status_code=503, detail="Session not ready")
    return session


@app.get("/")
def index() -> FileResponse:
    index_path = _static_dir / "index.html"
    return FileResponse(index_path)


@app.get("/api/state")
def get_state() -> dict:
    state = _get_session().current_state()
    return asdict(state)


@app.post("/api/new-game")
def new_game(req: NewGameRequest) -> dict:
    state = _get_session().reset(seed=req.seed)
    return asdict(state)


@app.post("/api/click")
def click_cell(req: ClickRequest) -> dict:
    state = _get_session().click(req.row, req.col)
    return asdict(state)


@app.post("/api/flag")
def flag_cell(req: FlagRequest) -> dict:
    try:
        state = _get_session().toggle_flag(req.row, req.col)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return asdict(state)
