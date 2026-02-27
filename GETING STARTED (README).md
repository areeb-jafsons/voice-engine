# GETING STARTED (README) — Abdion Voice Engine

Welcome to the **Abdion Voice Engine**. This repository contains a production-grade, high-concurrency microservice for real-time conversational AI. It uses Pipecat to orchestrate streaming audio, VAD, STT, LLM, and TTS models within an isolated process-per-call architecture.

---

## 🏗 Architecture Blueprint & Scope

The Voice Engine is designed to handle 100k+ calls per day without single points of failure. It consists of two main architectural layers:

### 1. Control Plane (`server.py`)
A fast, non-blocking HTTP/WebSocket API built with **FastAPI**.
- **Role:** Receives webhook events from your SIP provider (e.g., Twilio, Telnyx).
- **Concurrency Model:** For every incoming call, it mints a LiveKit token and spawns a completely isolated `bot.py` subprocess. 
- **Resilience:** Processes share no memory or locks. If one bot crashes due to an unhandled exception, it only affects that single call; the remaining concurrent calls are untouched.
- **Dynamic Config:** Publishes global configuration state dynamically to bots via stdin during the spawn phase.

### 2. Live Pipeline (`bot.py`)
The worker process executed per live call, built on **Pipecat (v0.0.102)**.
- **Input/Output Layer:** `LiveKitTransport` handles WebRTC audio streaming to and from the SIP gateway.
- **Acoustic VAD:** `SileroVADAnalyzer` runs locally to detect speech boundaries.
- **Transcription (STT):** `DeepgramSTTService` (Nova-3, multilingual) provides ultra-fast interim and final streaming transcripts.
- **Semantic Turn Gate:** A custom `HybridTurnGate` FrameProcessor that combines acoustic VAD silence with linguistic regex rules (English & Urdu) to prevent interrupting the user mid-thought.
- **Intelligence (LLM):** `GroqLLMService` (Llama 3.3 70B Versatile) streams conversational completions in real-time. System prompts enforce strict bilingual rules.
- **Synthesis (TTS):** `ElevenLabsTTSService` (Turbo v2.5) streams the synthesized audio back to the user with ultra-low latency.

---

## 📁 Project Structure

The project has been flattened and cleaned for production deployment:

```text
abdion-voice-engine/
├── server.py              # FastAPI control plane & API endpoints
├── bot.py                 # Pipecat agent worker (1 process per live call)
├── config.py              # Pydantic data models for dynamic runtime configuration
├── config.json            # Persisted state for system configurables
├── test_client.html       # Web-based Dev Console for testing and config management
├── requirements.txt       # Pinned Python dependencies
├── .env                   # Environment variables (API keys, URLs)
├── .env.example           # Sanitized template for environment variables
├── .gitignore             # Ignored files, logs, and caches
├── infra/                 # Infrastructure configuration
│   └── docker-compose.yml # Local LiveKit + Redis deployment spec
└── docs/                  # Project documentation
    └── VAD_UPGRADE_PLAN   # Future roadmap for Voice Activity Detection
```

---

## ⚙️ Dynamic Configuration

The engine supports **zero-downtime configuration**. You do not need to restart the server to change LLM instructions, models, or voices.

1. Open `test_client.html` in your browser.
2. Expand the **⚙ Configuration** panel.
3. Adjust LLM Temperature, ElevenLabs Voice ID, Deepgram settings, Semantic VAD timeout, or the System Prompt.
4. Click **Save Default**. The changes are instantly persisted to `config.json` and will be applied to the very next call that rings in.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.12 or 3.13
- Git
- Docker & Docker Compose (for running a local LiveKit server)

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd abdion-voice-engine
```

### Step 2: Set Up Environment Variables
Copy the template file to create your active `.env`:

**Mac/Linux:**
```bash
cp .env.example .env
```
**Windows (PowerShell):**
```powershell
Copy-Item .env.example -Destination .env
```

Open `.env` in your editor and fill in your API keys:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `DEEPGRAM_API_KEY`
- `GROQ_API_KEY`
- `ELEVENLABS_API_KEY`

### Step 3: Start Local Infrastructure (LiveKit)
If you don't use a hosted LiveKit Cloud instance, run the local dev server using Docker:
```bash
cd infra
docker-compose up -d
cd ..
```

### Step 4: Install Python Dependencies

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

*(Note: On Windows, if execution of scripts is disabled, run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser` as Administrator first).*

### Step 5: Run the Server
With your virtual environment activated, start the FastAPI control plane:

```bash
uvicorn server:app --reload --port 8000
```
*(The server will log `event=server_start max_concurrent=200` indicating it is ready).*

### Step 6: Test Locally
1. Open the file `test_client.html` directly in your web browser (Chrome/Edge recommended).
2. The UI will automatically connect to the server log stream.
3. Adjust any settings in the **Configuration** panel if desired.
4. Click **🔌 Connect & Talk**. Ensure you grant browser microphone permissions.
5. You can now speak to the bot locally to test latency and behavior before hooking up a real phone number!

---

## 🚢 Production Deployment

For deploying the `server.py` to production (e.g., an AWS EC2 instance, DigitalOcean Droplet, or railway.app):

1. **Do not use `--reload`**. Run via standard uvicorn workers:
   `uvicorn server:app --host 0.0.0.0 --port 8000`
2. **Reverse Proxy:** Put the FastAPI server behind Nginx or Caddy with SSL enabled.
3. **SIP Trunking:** Point your Twilio/Telnyx SIP webhook POST URL to `https://your-domain.com/on-call`.
4. Ensure your server has enough CPU cores, as each concurrent call spawns a unique Python process. 
