"""
Lightweight smoke test for the Minesweeper WebUI backend.

Usage:
    PYTHONPATH=. python scripts/test_webui.py

Relies on the default checkpoint path (or `MINESWEEPER_CKPT_16`).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from webui.app import app


def run_smoke() -> None:
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        assert state["rows"] == 16 and state["cols"] == 16
        assert state["mine_count"] == 40
        assert len(state["mine_probabilities"]) == state["rows"]
        assert state.get("next_move") is not None

        flagged = client.post("/api/flag", json={"row": 0, "col": 1}).json()
        assert flagged["flags"][0][1] is True
        unflagged = client.post("/api/flag", json={"row": 0, "col": 1}).json()
        assert unflagged["flags"][0][1] is False

        move = client.post("/api/click", json={"row": 0, "col": 0}).json()
        assert move["revealed"][0][0] is True
        assert move.get("next_move") is not None

        reset_state = client.post("/api/new-game", json={}).json()
        assert reset_state["step"] == 0
        assert reset_state["done"] is False


if __name__ == "__main__":
    run_smoke()
    print("WebUI smoke test passed.")
