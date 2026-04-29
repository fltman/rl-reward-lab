"""Demo server for the LLM-reward pipeline.

Single-user FastAPI app. The /generate endpoint spawns web/_runner.py as a
subprocess (so MuJoCo's renderer gets a proper main thread on macOS) and
forwards each JSON event from its stdout to the browser as SSE.

Run: uv run uvicorn web.server:app --host 127.0.0.1 --port 8765
Open: http://127.0.0.1:8765/
"""
from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent
RUNS = WEB / "runs"
RUNS.mkdir(exist_ok=True)
RUNNER = WEB / "_runner.py"

app = FastAPI()
busy_lock = threading.Lock()
app.mount("/runs", StaticFiles(directory=RUNS), name="runs")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


def _validate_goal(goal: str) -> str:
    g = goal.strip()
    if not g:
        raise HTTPException(400, "goal is empty")
    if len(g) > 240:
        raise HTTPException(400, "goal too long (max 240 chars)")
    return g


@app.get("/generate")
def generate(goal: str):
    """SSE stream of one full pipeline run. One client at a time."""
    goal = _validate_goal(goal)
    if not busy_lock.acquire(blocking=False):
        # Return a real SSE stream so the browser shows a useful message
        # (EventSource can't read HTTP status codes).
        def busy_stream():
            yield ('event: error\n'
                   'data: {"message": "Another run is already in progress. '
                   'Wait for it to finish, then try again."}\n\n')
            yield 'event: end\ndata: {}\n\n'
        return StreamingResponse(busy_stream(), media_type="text/event-stream")

    q: queue.Queue = queue.Queue()
    run_id = uuid.uuid4().hex[:8]

    def reader():
        proc = None
        stderr_lines: list[str] = []
        try:
            proc = subprocess.Popen(
                [sys.executable, str(RUNNER), goal, run_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(ROOT),
            )

            def drain_stderr():
                assert proc is not None and proc.stderr is not None
                for line in proc.stderr:
                    stderr_lines.append(line)

            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()

            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    q.put(json.loads(line))
                except json.JSONDecodeError:
                    q.put({"event": "log", "data": {"line": line}})

            proc.wait()
            stderr_thread.join(timeout=2)

            if proc.returncode != 0:
                tail = "".join(stderr_lines)[-1500:].strip()
                q.put({"event": "error", "data": {
                    "message": f"runner exited {proc.returncode}\n{tail or '(no stderr)'}"
                }})
        except Exception as exc:  # noqa: BLE001
            q.put({"event": "error", "data": {"message": f"server: {exc}"}})
        finally:
            q.put(None)
            busy_lock.release()

    threading.Thread(target=reader, daemon=True).start()

    def stream():
        yield f"event: run\ndata: {json.dumps({'run_id': run_id, 'goal': goal})}\n\n"
        while True:
            msg = q.get()
            if msg is None:
                yield "event: end\ndata: {}\n\n"
                return
            yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
