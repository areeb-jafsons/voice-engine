"""
server.py — Abdion Voice Engine · FastAPI Control Plane
=======================================================
Control plane for the Voice Engine.  Receives SIP webhook calls,
generates LiveKit access tokens, and spawns isolated `bot.py`
subprocesses — one per concurrent call.

Endpoints
---------
  POST /on-call              SIP webhook → spawn bot
  GET  /health               Service liveness
  GET  /agents               List running agents
  POST /agents/{room}/stop   Gracefully terminate a specific bot

Concurrency model
-----------------
Each call runs as an independent OS process (asyncio.create_subprocess_exec).
Processes share NO memory, queues, or event loops.  The `active_agents`
registry tracks PIDs and asyncio tasks that auto-clean on exit.

This design targets 100k calls/day:
  • No shared state = no locking across calls
  • Process crash is fully isolated — one bad call never kills another
  • OS scheduler handles CPU distribution across all bots
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Set

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# LiveKit Server SDK for JWT token generation
from livekit.api import AccessToken, VideoGrants

load_dotenv()

# ---------------------------------------------------------------------------
# WebSocket log broadcaster (defined early — referenced by logging handler)
# ---------------------------------------------------------------------------

class LogBroadcaster:
    """Fan-out hub for real-time log events to all connected WebSocket clients."""
    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._history: list[dict] = []  # last 500 events replayed to late-joiners

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        for event in self._history[-500:]:
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                break

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, event: dict) -> None:
        self._history.append(event)
        if len(self._history) > 500:
            self._history = self._history[-500:]
        dead: Set[WebSocket] = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                dead.add(ws)
        self._clients -= dead


broadcaster = LogBroadcaster()


class _WsBroadcastHandler(logging.Handler):
    """Logging handler that forwards every server log record to all WS clients."""
    def emit(self, record: logging.LogRecord) -> None:
        event = {
            "source": "server",
            "level":  record.levelname,
            "logger": record.name,
            "msg":    self.format(record),
            "ts":     record.created,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(
                lambda: loop.create_task(broadcaster.broadcast(event))
            )
        except RuntimeError:
            pass  # no event loop yet during startup


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if os.getenv("VOICE_DEBUG") else logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice_engine.server")

# Attach WS broadcast handler AFTER basicConfig has run
_ws_handler = _WsBroadcastHandler()
_ws_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
))
logging.root.addHandler(_ws_handler)

# ---------------------------------------------------------------------------
# Config (from environment)
# ---------------------------------------------------------------------------
LIVEKIT_URL        = os.environ["LIVEKIT_URL"]            # wss://your.livekit.cloud
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

BOT_SCRIPT  = os.path.join(os.path.dirname(__file__), "bot.py")
PYTHON_EXE  = sys.executable   # same venv python that runs this server

MAX_CONCURRENT_CALLS = int(os.getenv("MAX_CONCURRENT_CALLS", "200"))
BOT_TOKEN_TTL_SEC    = int(os.getenv("BOT_TOKEN_TTL_SEC",    "3600"))   # 1 hour
GRACEFUL_STOP_SEC    = float(os.getenv("GRACEFUL_STOP_SEC",  "5.0"))    # SIGTERM→SIGKILL


# ---------------------------------------------------------------------------
# Active agent registry
# ---------------------------------------------------------------------------

@dataclass
class AgentRecord:
    room_name:    str
    pid:          int
    started_at:   float   = field(default_factory=time.monotonic)
    process:      asyncio.subprocess.Process = field(repr=False, default=None)  # type: ignore[assignment]
    _watch_task:  Optional[asyncio.Task] = field(repr=False, default=None)


# room_name → AgentRecord
active_agents: dict[str, AgentRecord] = {}


# ---------------------------------------------------------------------------
# LiveKit token factory
# ---------------------------------------------------------------------------

def _make_bot_token(room_name: str, participant_identity: str) -> str:
    """Generate a scoped LiveKit JWT for a bot participant."""
    grants = VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
    )
    token = (
        AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity(participant_identity)
        .with_name(f"voice-bot-{room_name}")
        .with_grants(grants)
        .with_ttl(timedelta(seconds=BOT_TOKEN_TTL_SEC))
        .to_jwt()
    )
    return token


# ---------------------------------------------------------------------------
# Process lifecycle helpers
# ---------------------------------------------------------------------------

async def _relay_stream(
    stream: asyncio.StreamReader,
    room_name: str,
    pid: int,
    stream_name: str,
) -> None:
    """Read bot stdout/stderr line-by-line and broadcast + echo to server log."""
    while True:
        try:
            raw = await stream.readline()
        except Exception:
            break
        if not raw:
            break
        line = raw.decode(errors="replace").rstrip()
        if not line:
            continue
        # echo to server terminal too
        print(f"[bot:{room_name}:{stream_name}] {line}", flush=True)
        await broadcaster.broadcast({
            "source": "bot",
            "room":   room_name,
            "pid":    pid,
            "stream": stream_name,
            "level":  "ERROR" if stream_name == "stderr" else "INFO",
            "msg":    line,
            "ts":     time.time(),
        })


async def _spawn_bot(room_name: str, token: str) -> AgentRecord:
    """Spawn an isolated bot.py subprocess with captured output."""
    log.info("event=spawning_bot room=%s", room_name)

    proc = await asyncio.create_subprocess_exec(
        PYTHON_EXE, BOT_SCRIPT, room_name, token,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,   # capture → relay to WS clients
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),
    )

    record = AgentRecord(
        room_name=room_name,
        pid=proc.pid,
        process=proc,
    )
    active_agents[room_name] = record

    # Relay stdout + stderr to WebSocket broadcaster
    asyncio.create_task(
        _relay_stream(proc.stdout, room_name, proc.pid, "stdout"),
        name=f"relay_stdout_{room_name}",
    )
    asyncio.create_task(
        _relay_stream(proc.stderr, room_name, proc.pid, "stderr"),
        name=f"relay_stderr_{room_name}",
    )

    # Watcher auto-removes registry when process exits
    record._watch_task = asyncio.create_task(
        _watch_process(record),
        name=f"watch_bot_{room_name}",
    )

    log.info("event=bot_spawned room=%s pid=%d", room_name, proc.pid)
    return record


async def _watch_process(record: AgentRecord) -> None:
    """Await process exit, then clean up registry."""
    exit_code = await record.process.wait()
    elapsed   = time.monotonic() - record.started_at
    log.info(
        "event=bot_exited room=%s pid=%d exit_code=%d duration_sec=%.1f",
        record.room_name, record.pid, exit_code, elapsed,
    )
    active_agents.pop(record.room_name, None)


async def _stop_bot(record: AgentRecord) -> None:
    """Send SIGTERM; escalate to SIGKILL after GRACEFUL_STOP_SEC."""
    proc = record.process
    if proc.returncode is not None:
        return   # already exited

    log.info("event=stopping_bot room=%s pid=%d", record.room_name, record.pid)
    try:
        proc.terminate()   # SIGTERM on POSIX, TerminateProcess on Windows
    except ProcessLookupError:
        return

    try:
        await asyncio.wait_for(proc.wait(), timeout=GRACEFUL_STOP_SEC)
    except asyncio.TimeoutError:
        log.warning("event=kill_bot room=%s pid=%d reason=graceful_timeout", record.room_name, record.pid)
        try:
            proc.kill()
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class OnCallRequest(BaseModel):
    """
    Flexible SIP webhook body.  Use whichever fields your SIP provider sends.

    Twilio SIP:  room_name derived from `To` header (the DID / SIP URI).
    Telnyx SIP:  room_name derived from `to` field in the payload.
    Custom:      set `room_name` directly.
    """
    room_name:  Optional[str] = None   # explicit override
    to:         Optional[str] = None   # Telnyx / generic "To" number/URI
    To:         Optional[str] = None   # Twilio capitalisation variant
    caller_id:  Optional[str] = "unknown"
    call_sid:   Optional[str] = None   # Twilio CallSid / Telnyx call_control_id

    def resolved_room_name(self) -> str:
        """Derive a safe room name from whichever field is present."""
        raw = self.room_name or self.to or self.To or "default"
        # Sanitise: keep only alphanum + dash + underscore, max 64 chars
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in raw)
        return safe[:64].strip("-")


class AgentInfo(BaseModel):
    room_name:   str
    pid:         int
    uptime_sec:  float


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    log.info("event=server_start max_concurrent=%d", MAX_CONCURRENT_CALLS)
    yield
    # On shutdown: gracefully stop all running bots
    log.info("event=server_shutdown stopping %d active agents", len(active_agents))
    stop_tasks = [_stop_bot(r) for r in list(active_agents.values())]
    if stop_tasks:
        await asyncio.gather(*stop_tasks, return_exceptions=True)
    log.info("event=server_stopped")


app = FastAPI(
    title="Abdion Voice Engine",
    version="1.0.0",
    description="SIP-to-Pipecat bot dispatcher",
    lifespan=_lifespan,
)

# Allow file:// and any local origin to reach the API (dev only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/on-call", status_code=status.HTTP_202_ACCEPTED)
async def on_call(request: Request) -> JSONResponse:
    """
    SIP webhook handler.

    Expected call flow:
      1. SIP provider (Twilio/Telnyx/FreeSWITCH/Kamailio) receives an inbound
         call and sends an HTTP POST to this endpoint.
      2. This endpoint generates a LiveKit token, spawns `bot.py`, and returns
         202 immediately (SIP provider should not wait for the call to finish).
      3. The bot connects to the LiveKit room.  The SIP provider's bridge module
         (LiveKit SIP gateway) bridges the RTP stream into the same room.

    For raw JSON body (testing):
        { "room_name": "call-123", "caller_id": "+12125550100" }

    For Twilio form-encoded:
        To=+12125550100&From=+13055550199&CallSid=CA...
    """
    # -- Parse body (JSON or form-encoded) -----------------------------------
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = OnCallRequest(**(await request.json()))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    else:
        # Form-encoded (Twilio, Telnyx, FreeSWITCH ESL, etc.)
        form = await request.form()
        body = OnCallRequest(**{k: v for k, v in form.items()})

    room_name = body.resolved_room_name()
    log.info("event=on_call room=%s caller=%s", room_name, body.caller_id)

    # -- Concurrency limit ---------------------------------------------------
    if len(active_agents) >= MAX_CONCURRENT_CALLS:
        log.warning("event=concurrency_limit_reached current=%d max=%d", len(active_agents), MAX_CONCURRENT_CALLS)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Concurrency limit reached ({MAX_CONCURRENT_CALLS} active calls).",
        )

    # -- Idempotency guard --------------------------------------------------
    if room_name in active_agents:
        existing = active_agents[room_name]
        log.warning("event=duplicate_room room=%s pid=%d", room_name, existing.pid)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"status": "already_active", "room": room_name, "pid": existing.pid},
        )

    # -- Generate bot token -------------------------------------------------
    participant_id = f"bot-{room_name}"
    try:
        token = _make_bot_token(room_name, participant_id)
    except Exception as exc:
        log.error("event=token_generation_failed room=%s error=%s", room_name, exc)
        raise HTTPException(status_code=500, detail="Token generation failed.") from exc

    # -- Spawn bot subprocess -----------------------------------------------
    try:
        record = await _spawn_bot(room_name, token)
    except Exception as exc:
        log.error("event=spawn_failed room=%s error=%s", room_name, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to spawn bot process.") from exc

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status":    "dispatched",
            "room":      room_name,
            "pid":       record.pid,
            "livekit":   LIVEKIT_URL,
        },
    )


@app.get("/join-token")
async def join_token(room: str, identity: str = "human-tester") -> JSONResponse:
    """
    Mint a LiveKit access token for a *human* participant (browser client).
    Used exclusively by the local test client — not needed in production
    where the SIP gateway handles participant auth.

    Example:
        GET /join-token?room=quality-test&identity=alice
    """
    grants = VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
    )
    try:
        token = (
            AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name(identity)
            .with_grants(grants)
            .with_ttl(timedelta(seconds=3600))
            .to_jwt()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Token error: {exc}") from exc

    return JSONResponse({
        "token":       token,
        "url":         LIVEKIT_URL,
        "room":        room,
        "identity":    identity,
    })


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe."""
    return JSONResponse({
        "status":       "ok",
        "active_calls": len(active_agents),
        "max_calls":    MAX_CONCURRENT_CALLS,
        "capacity_pct": round(len(active_agents) / MAX_CONCURRENT_CALLS * 100, 1),
    })


@app.get("/agents", response_model=list[AgentInfo])
async def list_agents() -> list[AgentInfo]:
    """Returns a snapshot of all running bot processes."""
    now = time.monotonic()
    return [
        AgentInfo(
            room_name=r.room_name,
            pid=r.pid,
            uptime_sec=round(now - r.started_at, 1),
        )
        for r in active_agents.values()
    ]


@app.post("/agents/{room_name}/stop", status_code=status.HTTP_202_ACCEPTED)
async def stop_agent(room_name: str) -> JSONResponse:
    """Gracefully terminate a specific bot (SIGTERM → wait 5s → SIGKILL)."""
    record = active_agents.get(room_name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No active agent for room '{room_name}'.")

    asyncio.create_task(_stop_bot(record), name=f"stop_{room_name}")
    return JSONResponse({"status": "stopping", "room": room_name, "pid": record.pid})


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket) -> None:
    """
    Real-time log stream for the test client.
    Sends every server + bot log event as a JSON object:
    {
      "source": "server" | "bot",
      "level":  "INFO" | "WARNING" | "ERROR" | ...,
      "logger": "<logger name>",   # server only
      "room":   "<room>",          # bot only
      "pid":    <int>,             # bot only
      "stream": "stdout"|"stderr", # bot only
      "msg":    "<line>",
      "ts":     <unix float>
    }
    """
    await broadcaster.connect(ws)
    log.info("event=ws_log_client_connected remote=%s", ws.client)
    try:
        while True:
            # Keep the connection alive; we only send, never receive
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.disconnect(ws)
        log.info("event=ws_log_client_disconnected remote=%s", ws.client)
