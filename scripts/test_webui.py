"""
Lightweight smoke test for the Minesweeper WebUI backend.

Usage:
    PYTHONPATH=. python scripts/test_webui.py

Relies on the default checkpoint path (or `MINESWEEPER_CKPT`).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from webui.app import app


def run_smoke() -> None:
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        assert state["rows"] > 0 and state["cols"] > 0
        assert len(state["safe_probabilities"]) == state["rows"]

        move = client.post("/api/click", json={"row": 0, "col": 0}).json()
        assert move["revealed"][0][0] is True

        reset_state = client.post("/api/new-game", json={}).json()
        assert reset_state["step"] == 0
        assert reset_state["done"] is False

        resized_state = client.post("/api/new-game", json={"preset": "16x16"}).json()
        assert resized_state["rows"] == 16
        assert resized_state["cols"] == 16
        assert resized_state["mine_count"] == 40

        reverted_state = client.post("/api/new-game", json={"preset": "8x8"}).json()
        assert reverted_state["rows"] == 8
        assert reverted_state["cols"] == 8


if __name__ == "__main__":
    run_smoke()
    print("WebUI smoke test passed.")
