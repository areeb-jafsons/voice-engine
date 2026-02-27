# VAD Upgrade Implementation Plan
## Abdion Voice Engine — Streaming Neural VAD + SmartTurn-v2

**Date:** 2026-02-24  
**Target file:** `apps/pipeline/flux_groq_bridge.py`  
**Status:** PENDING IMPLEMENTATION

---

## TABLE OF CONTENTS

0. Current System Snapshot  
1. Upgrade Goals & Latency Targets  
2. New Components — Specs & APIs  
   2a. Silero VAD 5.1 — VADIterator (streaming mode)  
   2b. SmartTurn-v2 — Context-Aware Turn Detector  
3. What to REMOVE from the Current System  
4. What to ADD — Exact Code  
5. Updated `requirements.txt`  
6. New Pipeline Topology  
7. State Machine Changes  
8. Step-by-Step Implementation Sequence (Steps 1–11)  
9. Latency Budget Comparison  
10. Invariants & Regression Tests  
11. Modified Files Summary  
12. Rollback Plan  

---

## 0. CURRENT SYSTEM SNAPSHOT

### 0.1 Component Inventory

| Layer | Component | Library | Version (pinned) |
|-------|-----------|---------|-----------------|
| ASR | Deepgram Nova-2 | `deepgram-sdk` | 3.5.0 |
| LLM | Groq (llama3-70b) | `groq` | 0.9.0 |
| TTS | Kokoro ONNX | `kokoro-onnx` | 0.1.3 |
| VAD | WebRTC VAD (batch, 500 ms window) | `webrtcvad` | 2.0.10 |
| AEC | WebRTC AudioProcessor | `webrtc-noise-gain` | custom wrapper |
| Playback | sounddevice StreamingPlayback | `sounddevice` | 0.4.x |
| Dialog | TurnState FSM (IDLE/LISTENING/THINKING/SPEAKING) | stdlib | — |


### 0.2 Current VAD Parameters (lines 1103–1112 of flux_groq_bridge.py)

```python
_VAD_RATE        = 16_000          # Hz — VAD operates at 16 kHz
_VAD_FRAME_MS    = 30              # ms — webrtcvad frame size
_VAD_FRAME       = int(_VAD_RATE * _VAD_FRAME_MS / 1000)  # = 480 samples
_VAD_WINDOW_MS   = 500            # ms — rolling window for majority-vote
_VAD_WINDOW_FRAMES = _VAD_WINDOW_MS // _VAD_FRAME_MS      # = 16 frames
_VAD_SPEECH_THRESH = 0.6          # fraction of frames that must be speech
_VAD_SILENCE_MS  = 800            # ms of silence before end-of-turn commit
_VAD_SILENCE_FRAMES = _VAD_SILENCE_MS // _VAD_FRAME_MS    # = 26 frames
_VAD_BARGE_IN_MS = 200            # ms of speech during SPEAKING → interrupt
_VAD_BARGE_IN_FRAMES = _VAD_BARGE_IN_MS // _VAD_FRAME_MS  # = 6 frames
```

### 0.3 Current Audio Pipeline Order (audio_callback → workers)

```
Mic (sounddevice InputStream, blocksize=1280, sr=16kHz, dtype=int16)
  │
  ▼
audio_callback()
  ├─► AEC forward path  (webrtc-noise-gain, 160-sample frames)
  ├─► vad_queue         → vad_worker()  [batch webrtcvad, 500ms window]
  └─► audio_queue       → deepgram_sender()  [gated: IDLE/LISTENING only]
```

### 0.4 Known Problems with Current VAD

1. **Latency**: 500 ms rolling window → speech onset detected ≥ 500 ms after first phoneme.
2. **Majority vote is coarse**: a 60 % threshold over 16 frames is far less accurate than a neural model.
3. **No turn-completion intelligence**: 800 ms hard silence timer fires on every breath/pause, causing false commits mid-sentence.
4. **`blocksize=1280`** is not aligned to VADIterator's required 512-sample chunks.

---

## 1. UPGRADE GOALS & LATENCY TARGETS

| Metric | Current | Target after upgrade |
|--------|---------|---------------------|
| Speech onset latency | 500–550 ms | 32–96 ms (1–3 Silero frames) |
| False barge-in rate | High (majority vote) | Low (neural confidence) |
| False turn-commit rate | High (hard 800 ms timer) | Low (SmartTurn gate) |
| End-of-turn latency | 800 ms silence flat | 200–400 ms (SmartTurn fast-path) |


---

## 2. NEW COMPONENTS — SPECS & APIs

### 2a. Silero VAD 5.1 — VADIterator (Streaming Mode)

**Package:** `silero-vad==5.1.2`  
**Install:** `pip install silero-vad==5.1.2`  
**PyPI:** https://pypi.org/project/silero-vad/  
**GitHub:** https://github.com/snakers4/silero-vad  
**Model checkpoint:** downloaded automatically on first use via `torch.hub`; cached at `~/.cache/torch/hub/snakers4_silero-vad_master/`

#### Why VADIterator (not get_speech_timestamps)?

`get_speech_timestamps()` requires a complete audio tensor — it cannot run on a streaming mic. `VADIterator` maintains an internal GRU state machine and processes exactly **512 samples (32 ms at 16 kHz)** per call. It yields events as they happen, giving 32–96 ms onset latency.

#### Constructor

```python
from silero_vad import load_silero_vad, VADIterator

model = load_silero_vad()   # loads ONNX model; ~6 MB; no GPU needed

vad_iterator = VADIterator(
    model=model,
    threshold=0.5,           # float — confidence to call a frame "speech"
                             #   lower → more sensitive, more false positives
                             #   higher → less sensitive, misses quiet speech
                             #   recommended range: 0.45–0.60
    sampling_rate=16000,     # must match mic sample rate
    min_silence_duration_ms=100,   # ms of silence before speech_end fires
                                   # keep short; SmartTurn adds the real gate
    speech_pad_ms=30,        # ms to pad around speech events (smooths edges)
)
```

#### Call API (per 512-sample chunk)

```python
chunk = np.array([...], dtype=np.float32)  # shape (512,), values in [-1, 1]
result = vad_iterator(chunk, return_seconds=False)
# Returns one of:
#   {'start': sample_index}   — speech onset detected
#   {'end':   sample_index}   — speech ended (silence >= min_silence_duration_ms)
#   {}                        — no event this frame
```

#### Reset between utterances

```python
vad_iterator.reset_states()   # call after each committed turn
```

#### Input dtype requirement

The model expects `float32` in the range `[-1.0, 1.0]`.  
Current mic captures `int16` → must normalize before each call:

```python
chunk_f32 = chunk_int16.astype(np.float32) / 32768.0
```

#### Frame size alignment

- VADIterator requires exactly **512 samples** per call.
- Current `blocksize=1280` delivers 1280 samples per callback → 2 full chunks of 512 (1024 samples) + 256-sample remainder.
- **Fix:** Change `blocksize` to **512** so each callback delivers exactly one VADIterator chunk. This eliminates the remainder problem entirely.


---

### 2b. SmartTurn-v2 — Context-Aware Turn Detector

**Model:** `lllchak/smart-turn-v2`  
**HuggingFace Hub:** https://huggingface.co/lllchak/smart-turn-v2  
**Library:** `transformers==4.44.2`  
**Task:** Binary audio classification — predicts whether a VAD silence event is a genuine end-of-turn (`1`) or a mid-utterance pause (`0`).

#### How it works

SmartTurn-v2 is a fine-tuned audio transformer. When Silero fires `speech_end`, instead of immediately committing the turn, we pass the buffered audio segment to SmartTurn. It returns a probability score:

- Score ≥ `SMART_TURN_THRESHOLD` (0.65) → **commit the turn** → send to ASR/LLM
- Score < threshold → **hold** → continue listening (reset silence timer, await next Silero event)
- If no commit after `SMART_TURN_MAX_WAIT_MS` (1200 ms) → **force commit** (avoids hanging forever)

#### Load & Inference API

```python
from transformers import pipeline as hf_pipeline

smart_turn = hf_pipeline(
    task="audio-classification",
    model="lllchak/smart-turn-v2",
    device=-1,   # CPU; set to 0 for GPU
)

def is_end_of_turn(audio_segment_np: np.ndarray, sr: int = 16000) -> bool:
    """
    audio_segment_np: float32 numpy array, shape (N,), values [-1.0, 1.0]
    sr: sample rate (must be 16000)
    Returns True if SmartTurn predicts genuine end-of-turn.
    """
    results = smart_turn({"array": audio_segment_np, "sampling_rate": sr})
    # results is a list of dicts: [{"label": "LABEL_1", "score": 0.87}, ...]
    # LABEL_1 = end-of-turn, LABEL_0 = mid-utterance pause
    score = next(r["score"] for r in results if r["label"] == "LABEL_1")
    return score >= SMART_TURN_THRESHOLD
```

#### Thresholds & Timing Constants

```python
SMART_TURN_THRESHOLD   = 0.65    # confidence to commit a turn
SMART_TURN_MAX_WAIT_MS = 1200    # force-commit timeout (ms) after first speech_end
```

#### Latency Profile

SmartTurn inference on CPU with a 1–3 second audio clip takes approximately **80–150 ms**.  
This is acceptable because it replaces the fixed 800 ms silence timer, giving a net saving of 500–700 ms on typical utterances.

#### Audio buffer for SmartTurn

Maintain a rolling `bytearray` (or `collections.deque`) of PCM samples from the moment Silero fires `speech_start`. On `speech_end`, slice this buffer and pass to SmartTurn.


---

## 3. WHAT TO REMOVE FROM THE CURRENT SYSTEM

### 3.1 Remove: webrtcvad import and instantiation

**Location:** Top of `flux_groq_bridge.py`, imports section  
**Remove this line:**
```python
import webrtcvad
```
And anywhere `webrtcvad.Vad()` is instantiated (search for `webrtcvad.Vad`).

### 3.2 Remove: Batch VAD constants

**Location:** Lines 1103–1112 (current VAD constants block)  
**Remove ALL of:**
```python
_VAD_RATE            = 16_000
_VAD_FRAME_MS        = 30
_VAD_FRAME           = int(_VAD_RATE * _VAD_FRAME_MS / 1000)
_VAD_WINDOW_MS       = 500
_VAD_WINDOW_FRAMES   = _VAD_WINDOW_MS // _VAD_FRAME_MS
_VAD_SPEECH_THRESH   = 0.6
_VAD_SILENCE_MS      = 800
_VAD_SILENCE_FRAMES  = _VAD_SILENCE_MS // _VAD_FRAME_MS
```
The barge-in constants below are kept but will be re-expressed in Silero frame units:
```python
_VAD_BARGE_IN_MS     = 200
_VAD_BARGE_IN_FRAMES = _VAD_BARGE_IN_MS // _VAD_FRAME_MS   # REMOVE this line too
```

### 3.3 Remove: Rolling window majority-vote logic inside vad_worker

The entire body of the current `vad_worker` coroutine uses a `deque` of webrtcvad frame decisions and computes a majority vote. **Delete this entire logic.** The new `vad_worker` is specified in Section 4.

### 3.4 Remove: blocksize=1280 from InputStream

**Location:** `sounddevice.InputStream(...)` constructor call  
**Change:**
```python
# REMOVE:
blocksize=1280,
```
This will be replaced with `blocksize=512` (see Section 4.3).

### 3.5 Remove: get_speech_timestamps usage (if any)

Search for `get_speech_timestamps` in the file. If present anywhere outside comments, delete it. This batch API cannot be used in a streaming pipeline.

### 3.6 Remove: _VAD_FRAME_MS / _VAD_FRAME references in audio_callback

The AEC loop uses `_AEC_FRAME = 160` (separate constant, keep this). Any references to `_VAD_FRAME` in `audio_callback` should be deleted — VAD is no longer called from within `audio_callback`; it consumes from `vad_queue`.


---

## 4. WHAT TO ADD — EXACT CODE

### 4.1 New imports (add to top of file, with other imports)

```python
# --- Silero VAD streaming ---
from silero_vad import load_silero_vad, VADIterator

# --- SmartTurn-v2 ---
from transformers import pipeline as hf_pipeline
```

### 4.2 New constants (replace removed block from Section 3.2)

```python
# ── Silero VAD (streaming) ───────────────────────────────────────────────────
_VAD_RATE            = 16_000        # Hz
_SILERO_CHUNK        = 512           # samples — VADIterator hard requirement
_SILERO_CHUNK_MS     = int(_SILERO_CHUNK / _VAD_RATE * 1000)  # = 32 ms
_SILERO_THRESHOLD    = 0.5           # speech confidence gate
_SILERO_MIN_SIL_MS   = 100          # ms silence before speech_end event fires
_SILERO_PAD_MS       = 30           # ms padding around speech segments

# Barge-in: N consecutive Silero speech frames required to trigger interrupt
_VAD_BARGE_IN_MS     = 200
_VAD_BARGE_IN_FRAMES = _VAD_BARGE_IN_MS // _SILERO_CHUNK_MS  # = 6 frames

# ── SmartTurn-v2 ─────────────────────────────────────────────────────────────
SMART_TURN_THRESHOLD   = 0.65   # end-of-turn confidence gate
SMART_TURN_MAX_WAIT_MS = 1200   # force-commit timeout after first speech_end
```

### 4.3 Model initialisation (add near top of main / startup section)

```python
# Load Silero VAD model (ONNX, ~6 MB, CPU-only, cached after first run)
_silero_model   = load_silero_vad()
_vad_iterator   = VADIterator(
    model=_silero_model,
    threshold=_SILERO_THRESHOLD,
    sampling_rate=_VAD_RATE,
    min_silence_duration_ms=_SILERO_MIN_SIL_MS,
    speech_pad_ms=_SILERO_PAD_MS,
)
log.info("event=silero_vad_loaded")

# Load SmartTurn-v2 (downloads ~90 MB on first run; cached in HF hub cache)
_smart_turn = hf_pipeline(
    task="audio-classification",
    model="lllchak/smart-turn-v2",
    device=-1,   # CPU; change to 0 if CUDA available
)
log.info("event=smart_turn_loaded")
```


### 4.4 blocksize change in InputStream

Find the `sounddevice.InputStream(...)` constructor and change:

```python
# BEFORE:
blocksize=1280,

# AFTER:
blocksize=512,    # = _SILERO_CHUNK — one exact VADIterator frame per callback
```

### 4.5 SmartTurn helper function (add as module-level function)

```python
def _check_smart_turn(audio_buf: np.ndarray) -> bool:
    """
    Run SmartTurn-v2 on buffered speech audio.

    Parameters
    ----------
    audio_buf : np.ndarray
        dtype=float32, shape (N,), values in [-1.0, 1.0].
        Should contain all audio since speech_start up to speech_end.

    Returns
    -------
    bool
        True  → genuine end-of-turn; commit the utterance.
        False → mid-utterance pause; keep listening.
    """
    results = _smart_turn({"array": audio_buf, "sampling_rate": _VAD_RATE})
    score = next((r["score"] for r in results if r["label"] == "LABEL_1"), 0.0)
    log.info("event=smart_turn_score score=%.3f threshold=%.2f", score, SMART_TURN_THRESHOLD)
    return score >= SMART_TURN_THRESHOLD
```

### 4.6 New vad_worker (complete replacement)

Replace the entire body of the existing `vad_worker` coroutine with:

```python
async def vad_worker():
    """
    Streaming VAD worker — Silero VADIterator + SmartTurn-v2.

    Consumes int16 frames from vad_queue (blocksize=512, one chunk per frame).
    Fires speech_start / speech_end events.
    On speech_end, SmartTurn decides whether to commit.
    On speech during SPEAKING, counts frames for barge-in.
    """
    loop = asyncio.get_event_loop()

    # Rolling buffer of float32 samples since last speech_start
    speech_buf: list[np.ndarray] = []
    in_speech   = False
    barge_frames = 0
    smart_turn_deadline: float | None = None   # wall-clock deadline for force-commit

    log.info("event=vad_worker_start engine=silero+smartturn")

    while True:
        # --- Pull one 512-sample frame from the queue -----------------------
        shaped = await loop.run_in_executor(None, vad_queue.get)   # int16 (512,1)
        chunk_i16 = shaped[:, 0]                                    # int16 (512,)
        chunk_f32 = chunk_i16.astype(np.float32) / 32768.0         # float32 [-1,1]

        # --- Run Silero VADIterator -----------------------------------------
        result = _vad_iterator(chunk_f32, return_seconds=False)

        sess = _active_session
        state = sess.state if sess else TurnState.IDLE
```


```python
        # --- SPEECH START --------------------------------------------------
        if "start" in result:
            in_speech = True
            speech_buf.clear()
            barge_frames = 0
            smart_turn_deadline = None
            log.info("event=vad_speech_start")
            if sess and state in (TurnState.IDLE, TurnState.LISTENING):
                sess._set_state(TurnState.LISTENING)

        # --- Accumulate samples while in speech ----------------------------
        if in_speech:
            speech_buf.append(chunk_f32)

        # --- Barge-in detection during SPEAKING ----------------------------
        if state == TurnState.SPEAKING and "start" in result:
            barge_frames += 1
            log.info("event=vad_barge_frame count=%d/%d", barge_frames, _VAD_BARGE_IN_FRAMES)
            if barge_frames >= _VAD_BARGE_IN_FRAMES:
                log.info("event=barge_in_triggered")
                if sess:
                    sess.stop_playback()
                    sess._set_state(TurnState.LISTENING)
                barge_frames = 0
        elif state != TurnState.SPEAKING:
            barge_frames = 0   # reset counter when not in playback

        # --- SPEECH END — run SmartTurn ------------------------------------
        if "end" in result and in_speech:
            in_speech = False
            log.info("event=vad_speech_end")

            # Concatenate buffered audio
            audio_segment = np.concatenate(speech_buf) if speech_buf else np.zeros(512, dtype=np.float32)
            speech_buf.clear()

            # Set force-commit deadline on first speech_end
            if smart_turn_deadline is None:
                smart_turn_deadline = time.perf_counter() + SMART_TURN_MAX_WAIT_MS / 1000.0

            # Run SmartTurn in executor (avoids blocking the event loop)
            is_eot = await loop.run_in_executor(None, _check_smart_turn, audio_segment)
            timed_out = time.perf_counter() >= smart_turn_deadline

            if is_eot or timed_out:
                reason = "smart_turn" if is_eot else "timeout"
                log.info("event=turn_commit reason=%s", reason)
                smart_turn_deadline = None
                _vad_iterator.reset_states()
                if sess and state in (TurnState.LISTENING,):
                    sess._set_state(TurnState.THINKING)
                    # Signal Deepgram to finalize transcript
                    await sess.finalize_transcript()
            else:
                log.info("event=smart_turn_hold waiting_for_more_speech")
                # Resume listening; Silero will fire another speech_end later
```

> **Note:** `finalize_transcript()` is the existing session method that sends a Deepgram close-stream signal and awaits the final transcript result. If the method has a different name in the current codebase, substitute accordingly.


---

## 5. UPDATED `requirements.txt`

Replace the entire contents of `apps/pipeline/requirements.txt` with:

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
sqlalchemy==2.0.30
asyncpg==0.29.0
redis==5.0.4
livekit==0.12.3
deepgram-sdk==3.5.0
groq==0.9.0
kokoro-onnx==0.1.3
numpy==1.26.4
soundfile==0.12.1

# ── VAD upgrade ──────────────────────────────────────────────────────────────
silero-vad==5.1.2          # streaming neural VAD (VADIterator)
torch==2.3.1               # required by silero-vad (CPU-only wheel is fine)
transformers==4.44.2       # SmartTurn-v2 HuggingFace pipeline
```

**Notes:**
- `webrtcvad` is intentionally removed.
- `torch==2.3.1` CPU-only wheel: install with  
  `pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu`  
  (saves ~2 GB vs the CUDA build; SmartTurn runs fine on CPU for this workload).
- `transformers==4.44.2` pins the version tested with `lllchak/smart-turn-v2`.

---

## 6. NEW PIPELINE TOPOLOGY

```
Mic  (sounddevice InputStream)
  blocksize = 512  (= one Silero frame = 32 ms @ 16 kHz)
  channels  = 1
  dtype     = int16
  samplerate = 16000
       │
       ▼
audio_callback()
  │
  ├─► AEC forward path        (webrtc-noise-gain, 160-sample sub-frames)
  │       always runs — keeps echo model in sync with reverse path
  │
  ├─► vad_queue  ──────────────► vad_worker()
  │       always fed                  │
  │       (even during SPEAKING)      ├─ Silero VADIterator (512 samples/frame)
  │                                   │     speech_start → set LISTENING
  │                                   │     speech during SPEAKING → barge-in counter
  │                                   │     speech_end → SmartTurn inference
  │                                   │          ├─ score ≥ 0.65 → THINKING → finalize
  │                                   │          └─ score < 0.65 → hold, keep listening
  │                                   └─ force-commit after 1200 ms (safety net)
  │
  └─► audio_queue ─────────────► deepgram_sender()
          gated: IDLE / LISTENING only
          (suppressed during THINKING / SPEAKING)
```


---

## 7. STATE MACHINE CHANGES

The four-state FSM (`IDLE → LISTENING → THINKING → SPEAKING → IDLE`) is unchanged in structure. The transitions triggered by VAD are re-mapped as follows:

| Event | Old trigger | New trigger |
|-------|-------------|-------------|
| `IDLE → LISTENING` | webrtcvad majority vote (500 ms window) | Silero `speech_start` event |
| `LISTENING → THINKING` | 800 ms hard silence timer | SmartTurn score ≥ 0.65 on Silero `speech_end` |
| `SPEAKING → LISTENING` (barge-in) | 6 webrtcvad frames during SPEAKING | 6 Silero speech frames during SPEAKING |
| `THINKING → SPEAKING` | unchanged (TTS first chunk ready) | unchanged |
| `SPEAKING → IDLE` | unchanged (drain wait + guarded transition) | unchanged |

**Additions to state entry actions:**

- On entering `IDLE` from `SPEAKING`: call `_vad_iterator.reset_states()` to flush any residual Silero state from the previous turn's audio.
- On entering `LISTENING` from barge-in: call `_vad_iterator.reset_states()` to clear state accumulated during TTS playback noise.

---

## 8. STEP-BY-STEP IMPLEMENTATION SEQUENCE

### Step 1 — Install dependencies

```bash
pip install silero-vad==5.1.2
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.44.2
pip install huggingface_hub   # usually already present via transformers
```

Verify Silero loads:
```python
from silero_vad import load_silero_vad, VADIterator
m = load_silero_vad(); print("ok")
```

Verify SmartTurn loads (will download ~90 MB model on first run):
```python
from transformers import pipeline as hf_pipeline
st = hf_pipeline("audio-classification", model="lllchak/smart-turn-v2", device=-1)
print("ok")
```

### Step 2 — Update requirements.txt

Apply the full replacement from Section 5. Commit this change separately before touching Python files.

### Step 3 — Remove webrtcvad

In `flux_groq_bridge.py`:
1. Delete `import webrtcvad` line.
2. Delete the VAD constants block (Section 3.2).
3. Delete the `webrtcvad.Vad()` instantiation.

Run `python -c "import flux_groq_bridge"` — it will fail on missing `_VAD_FRAME` references; that is expected. Proceed to Step 4.

### Step 4 — Add new imports

Add the two import lines from Section 4.1 directly below the existing `import sounddevice` line.


### Step 5 — Add new constants

Replace the deleted constants block with the new block from Section 4.2. Place it at the same location in the file (the VAD constants section near the top of the module-level constants).

### Step 6 — Change blocksize

Find `blocksize=1280` in the `sounddevice.InputStream(...)` call. Change it to `blocksize=512`. This is a one-character edit; double-check no other code path assumes a 1280-sample callback (search the file for `1280`).

### Step 7 — Add model initialisation

Add the Silero + SmartTurn init block from Section 4.3 in the startup/init section, after `aec = AudioProcessor(...)` and before any worker coroutines are defined. These are module-level globals used by `vad_worker`.

### Step 8 — Add _check_smart_turn helper

Add the `_check_smart_turn()` function from Section 4.5 as a module-level function, directly above the `vad_worker` coroutine definition.

### Step 9 — Replace vad_worker body

Delete the entire existing body of `vad_worker`. Replace it with the new body from Sections 4.6 (both code blocks, concatenated in order — the outer `while True` block and the inner event-handling block).

### Step 10 — Add _vad_iterator.reset_states() calls

1. In `_set_state()` or the IDLE entry logic: call `_vad_iterator.reset_states()` when transitioning TO `IDLE`.
2. In barge-in handling: call `_vad_iterator.reset_states()` when transitioning TO `LISTENING` via barge-in.

Exact locations to insert:
```python
# In _set_state(), case TurnState.IDLE:
_vad_iterator.reset_states()
log.info("event=silero_reset reason=idle_entry")

# In vad_worker, after barge-in fires:
_vad_iterator.reset_states()
log.info("event=silero_reset reason=barge_in")
```

### Step 11 — Smoke test

Start the pipeline and verify the following log sequence for a normal utterance:

```
event=silero_vad_loaded
event=smart_turn_loaded
event=vad_worker_start engine=silero+smartturn
... (mic running) ...
event=vad_speech_start
event=vad_speech_end
event=smart_turn_score score=0.82 threshold=0.65
event=turn_commit reason=smart_turn
```

And for a barge-in during SPEAKING:
```
event=vad_barge_frame count=1/6
event=vad_barge_frame count=2/6
...
event=vad_barge_frame count=6/6
event=barge_in_triggered
event=silero_reset reason=barge_in
event=playback_stop
```


---

## 9. LATENCY BUDGET COMPARISON

All times are worst-case estimates at 16 kHz, CPU-only.

| Stage | Old system | New system | Delta |
|-------|-----------|-----------|-------|
| Speech onset detection | 500 ms (rolling window) | 32–96 ms (1–3 Silero frames) | **−450 ms** |
| End-of-turn decision | 800 ms (hard silence timer) | 80–150 ms (SmartTurn inference) | **−650 ms** |
| Barge-in detection | Dead (frames dropped) | 192 ms (6 × 32 ms frames) | Fixed |
| AEC forward path | Always (unchanged) | Always (unchanged) | 0 ms |
| Deepgram echo leakage | Yes (frames fed during SPEAKING) | No (gated by state) | Fixed |
| **Total per-turn latency saving** | — | — | **~1100 ms** |

---

## 10. INVARIANTS & REGRESSION TESTS

After implementation, verify all of the following manually:

| # | Invariant | How to verify |
|---|-----------|---------------|
| 1 | AEC always runs (never skipped) | Check logs: `event=aec_process_frame` fires on every callback even during SPEAKING |
| 2 | VAD always fed (even during SPEAKING) | Check logs: `event=vad_queue_put` fires on every callback |
| 3 | Deepgram suppressed during THINKING/SPEAKING | Check logs: `event=audio_queue_put` absent during those states |
| 4 | Barge-in fires within 200 ms of speech onset | Say something during TTS playback; measure `vad_speech_start` → `barge_in_triggered` |
| 5 | No audio join after barge-in | Listen: old TTS must stop, silence, then new response starts cleanly |
| 6 | SmartTurn holds on mid-sentence pause | Pause mid-sentence for ~600 ms; turn must NOT commit |
| 7 | Force-commit fires after 1200 ms | Deliberately mumble quietly; after 1200 ms turn must commit regardless |
| 8 | Silero reset on IDLE entry | Check logs: `event=silero_reset reason=idle_entry` after each turn completes |
| 9 | State guard: no IDLE override after barge-in | Barge-in mid-TTS; confirm state stays LISTENING not IDLE |
| 10 | blocksize=512 confirmed | Add temporary log in audio_callback: `log.debug("frames=%d", frames)` → must print 512 |

---

## 11. MODIFIED FILES SUMMARY

| File | Change type | Summary |
|------|-------------|---------|
| `apps/pipeline/flux_groq_bridge.py` | Heavy edit | Remove webrtcvad, add Silero+SmartTurn, new vad_worker, blocksize=512, reset_states calls |
| `apps/pipeline/requirements.txt` | Full replace | Remove webrtcvad, add silero-vad, torch, transformers |

No other files require modification. The upgrade is fully contained in the pipeline layer.

---

## 12. ROLLBACK PLAN

If the upgrade introduces regressions:

1. `git stash` or `git checkout apps/pipeline/flux_groq_bridge.py apps/pipeline/requirements.txt`
2. `pip install webrtcvad==2.0.10`
3. `pip uninstall silero-vad torch transformers -y`

The old webrtcvad-based `vad_worker` is preserved in git history. No database migrations or external service changes are needed — the upgrade is purely local to the pipeline process.

---

## APPENDIX A — Silero VADIterator Internal State Machine

```
SILENT ──(confidence ≥ threshold)──► SPEECH_ONSET
SPEECH_ONSET ──(N frames speech)──► IN_SPEECH
IN_SPEECH ──(silence ≥ min_silence_duration_ms)──► SPEECH_ENDED  [fires {'end': N}]
SPEECH_ENDED ──(reset_states())──► SILENT
```

The internal GRU hidden state accumulates temporal context across calls. **Never** skip frames or feed silence-padded arrays — always feed the real mic audio to keep the GRU synchronized.

---

## APPENDIX B — SmartTurn-v2 Label Schema

| Label | Meaning |
|-------|---------|
| `LABEL_0` | Mid-utterance pause — keep listening |
| `LABEL_1` | Genuine end-of-turn — commit utterance |

The pipeline checks `LABEL_1` score against `SMART_TURN_THRESHOLD = 0.65`. If `LABEL_1` is absent from results (model error), `next(..., 0.0)` returns 0.0, which is below threshold → safe fallback to force-commit after timeout.

---

*End of VAD_UPGRADE_PLAN.md*
