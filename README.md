# Abdion Voice Engine

Production-grade, process-isolated voice AI microservice built on **Pipecat-ai 0.0.102** + **LiveKit WebRTC**.

## Architecture

```
SIP Call ──► POST /on-call (FastAPI)
                │
                ├─ Generate LiveKit JWT
                ├─ asyncio.create_subprocess_exec(bot.py, room, token)
                └─► bot.py (isolated OS process per call)
                          │
                          └─ Pipecat Pipeline:
                             LiveKit mic
                             → Deepgram STT (nova-3, multi)
                             → HybridTurnGate ← Hybrid VAD
                             → Groq LLM (llama-3.3-70b-versatile)
                             → ElevenLabs TTS (eleven_turbo_v2_5)
                             → LiveKit speaker
```

## Quick Start

### 1. Install Dependencies

```powershell
# Create / activate your virtualenv first, e.g.:
# python -m venv .venv && .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
# CPU-only torch (faster install if no GPU):
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. Configure Environment

```powershell
copy .env.example .env
# Edit .env and fill in all API keys
notepad .env
```

### 3. Start the Control Plane

```powershell
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note**: Use `--workers 1`. The `active_agents` dict is in-process; multiple workers would each have their own isolated registry.

### 4. Verify Health

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/health" | ConvertFrom-Json
```

Expected output:
```json
{"status": "ok", "active_calls": 0, "max_calls": 200, "capacity_pct": 0.0}
```

---

## SIP Trunk Setup

The `/on-call` endpoint acts as your **SIP Action URL** (Twilio) or **webhook** (Telnyx / FreeSWITCH). The LiveKit SIP Gateway bridges the RTP audio stream into the LiveKit room that the bot joins.

### Architecture Overview

```
PSTN / SIP Client
     │
     ▼ RTP
LiveKit SIP Gateway (cloud or self-hosted)
     │
     ├─ creates LiveKit room: "call-<DID>"
     ├─ sends HTTP POST to your /on-call endpoint
     │
     ▼ HTTP POST /on-call
FastAPI server.py
     │
     └─ spawns bot.py which joins the same LiveKit room
```

### Option A — Twilio Elastic SIP Trunking

1. **Create a Twilio SIP Trunk** in the [Twilio Console](https://console.twilio.com/us1/develop/voice/manage/sip-trunking).
2. Set **Voice Configuration → Request URL**:
   ```
   https://your-server.example.com/on-call
   ```
   Method: `HTTP POST`
3. Assign a Phone Number (DID) to the trunk.
4. In your `/on-call` handler the `To` field contains the dialled DID — used as the `room_name`.

Twilio will POST form-encoded fields including `To`, `From`, `CallSid`. `server.py` handles this automatically.

### Option B — Telnyx SIP Connection

1. In the [Telnyx Mission Control portal](https://portal.telnyx.com/), go to **Voice → SIP Connections**.
2. Create a connection with **Webhook URL**:
   ```
   https://your-server.example.com/on-call
   ```
3. Set **Webhook API Version**: API v2.
4. Assign a DID to the SIP connection.

Telnyx POSTs JSON. The `to` field in the payload becomes the `room_name`.

### Option C — LiveKit SIP Gateway (Recommended)

The cleanest approach is to use **LiveKit's native SIP integration**, which terminates SIP directly and fires the webhook:

1. Deploy or use LiveKit Cloud.
2. Create a **SIP Trunk** in the LiveKit dashboard.
3. Set the **Dispatch Rule Webhook** to:
   ```
   https://your-server.example.com/on-call
   ```
4. LiveKit will call your webhook with a JSON body containing `room_name`.

### Webhook Body (generic / testing)

For manual testing or custom SIP integrations, POST JSON:
```powershell
Invoke-WebRequest -Method POST `
  -Uri "http://localhost:8000/on-call" `
  -ContentType "application/json" `
  -Body '{"room_name": "test-call-001", "caller_id": "+12125550100"}'
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/on-call` | SIP webhook → spawn bot |
| `GET` | `/health` | Liveness probe |
| `GET` | `/agents` | List running bots (room, PID, uptime) |
| `POST` | `/agents/{room}/stop` | Gracefully terminate a bot |

---

## HybridTurnGate — Hybrid VAD Logic

The `HybridTurnGate` FrameProcessor inside `bot.py` combines two VAD layers:

| Layer | Technology | Role |
|---|---|---|
| **Acoustic VAD** | Silero (via LiveKit transport) | Detects speech start/end from raw audio energy |
| **Semantic VAD** | Regex (English + Roman-Urdu) | Extends the commit window if the transcript trails off on a connector word |

### Semantic Extension Timer

The extension uses a **monotonic deadline float** (`time.monotonic() + 1.2`), checked by a 100ms async poll loop. This prevents the "instant double-trigger" bug present in the original monolith where a `continue` inside a tight loop could re-evaluate a stale deadline within the same tick.

- Max 2 extensions of 1.2s each (2.4s total bonus window)
- If both extensions are exhausted, the turn commits unconditionally
- If new speech starts, the pending timer task is cancelled immediately

### Urdu / English Connectors Detected

| Language | Examples |
|---|---|
| English | `and`, `or`, `but`, `with`, `because`, `that`, `which`, `in`, `for`, … |
| Roman-Urdu | `ke`, `ki`, `aur`, `lekin`, `magar`, `ya`, `phir`, `toh`, `agar`, `kyunke`, … |

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `LIVEKIT_URL` | ✅ | — | LiveKit WebSocket URL |
| `LIVEKIT_API_KEY` | ✅ | — | LiveKit server API key |
| `LIVEKIT_API_SECRET` | ✅ | — | LiveKit server API secret |
| `DEEPGRAM_API_KEY` | ✅ | — | Deepgram API key |
| `GROQ_API_KEY` | ✅ | — | Groq API key |
| `ELEVENLABS_API_KEY` | ✅ | — | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | ❌ | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice (Rachel) |
| `GROQ_LLM_MODEL` | ❌ | `llama-3.3-70b-versatile` | Override Groq model |
| `MAX_CONCURRENT_CALLS` | ❌ | `200` | Concurrency cap |
| `BOT_TOKEN_TTL_SEC` | ❌ | `3600` | LiveKit JWT TTL (seconds) |
| `GRACEFUL_STOP_SEC` | ❌ | `5.0` | SIGTERM→SIGKILL timeout |
| `VOICE_DEBUG` | ❌ | `false` | Enable DEBUG logging |

---

## Scaling

For **100k calls/day** (~1.15 calls/second average, with spikes):

1. **Horizontal scale** `server.py` behind a load balancer (nginx / AWS ALB).
   - Each instance tracks its own `active_agents` — use a shared Redis or Postgres counter for global limit enforcement.
2. **Run `bot.py` on separate worker nodes** by changing the `asyncio.create_subprocess_exec` call to post a job to a task queue (e.g., Celery / RQ) instead of spawning locally.
3. **Use LiveKit Cloud autoscaling** for the media layer — it handles the SFU load.
