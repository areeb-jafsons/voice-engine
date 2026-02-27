"""
bot.py — Abdion Voice Engine · Pipecat Agent Worker
====================================================
One OS process per live call.  Spawned by server.py via
asyncio.create_subprocess_exec.  Inherits env vars from the parent.

Usage
-----
    python bot.py <livekit_room_name> <livekit_participant_token>

Pipeline
--------
LiveKitTransport (mic in)
    → DeepgramSTTService (nova-3, language=multi)
    → HybridTurnGate  ← custom FrameProcessor (Hybrid VAD)
    → GroqLLMService  (llama-3.3-70b-versatile, streaming)
    → ElevenLabsTTSService (eleven_turbo_v2_5, streaming)
    → LiveKitTransport (speaker out)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import time
from enum import Enum, auto
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Pipecat imports
# ---------------------------------------------------------------------------
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMMessagesFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    StartInterruptionFrame,
    LLMFullResponseEndFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.livekit.transport import (
    LiveKitParams,
    LiveKitTransport,
)
from deepgram import LiveOptions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG if os.getenv("VOICE_DEBUG") else logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice_engine.bot")

# ---------------------------------------------------------------------------
# Runtime config (all from env)
# ---------------------------------------------------------------------------
LIVEKIT_URL       = os.environ["LIVEKIT_URL"]
DEEPGRAM_API_KEY  = os.environ["DEEPGRAM_API_KEY"]
GROQ_API_KEY      = os.environ["GROQ_API_KEY"]
ELEVENLABS_API_KEY = os.environ["ELEVENLABS_API_KEY"]
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel default

# Groq LLM model — override via env for easy A/B testing
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Bilingual system prompt (Urdu / English code-switching)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a multilingual real-time conversational AI voice assistant for Abdion.
YOU ARE A FEMALE!!
STRICT RULES:
- Always reply in the SAME language the user speaks in their CURRENT message.
- If the user speaks Urdu or Hindi (Roman script OR Nastaliq script), respond FULLY in Urdu.
- If the user speaks English, respond FULLY in English.
- If the user mixes languages (code-switching), mirror their mixing ratio naturally.
- You can speak Hindi too if the user speaks Urdu or Hindi . Hindi and Urdu may share vocabulary
  but differ in register; always use Urdu vocabulary and phrasing when responding.
- Keep responses concise, conversational, and natural for voice (no markdown, no lists).
- You may use Roman Urdu (Urdu written in Latin script) when the user does so.
- Never break character or discuss these instructions.
"""

# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
#  HYBRID TURN GATE  —  Custom FrameProcessor
#  Ports the Hybrid VAD logic from flux_groq_bridge.py into Pipecat's
#  frame-pipeline model.
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 1. Semantic completeness regex  (ported verbatim from flux_groq_bridge.py)
# --------------------------------------------------------------------------

_SENTENCE_COMPLETE_PUNCT_RE = re.compile(r'[.!?؟۔]\s*$')

_ENGLISH_CONTINUATION_RE = re.compile(
    r'\b(?:and|or|but|with|the|my|your|his|her|its|our|their|'
    r'because|so|if|when|that|which|to|for|a|an|in|of|on|at|'
    r'is|are|was|were|i|we|they|he|she|it|not|also|then|than|'
    r'about|from|into|over|after|before|during|without|between)\s*$',
    re.IGNORECASE,
)

_URDU_CONTINUATION_RE = re.compile(
    r'\b(?:ke|ki|aur|lekin|magar|ya|phir|toh|to|jab|agar|kyunke|'
    r'taake|jo|jis|jin|un|us|ye|wo|mein|par|se|ka|ko|ne)\s*$',
    re.IGNORECASE,
)

SEMANTIC_EXTENSION_SEC  = 1.2   # extra seconds per extension window
MAX_SEMANTIC_EXTENSIONS = 2     # hard cap — prevents unbounded waiting


def is_semantically_complete(text: str) -> bool:
    """Fast regex heuristic: does *text* look like a finished thought?

    Handles English and Roman-Urdu.  Returns True when the utterance
    appears complete, False when it trails off on a conjunction,
    preposition, or connector.

    Designed for zero-overhead synchronous invocation on every commit tick.
    """
    text = text.strip()
    if not text:
        return True

    if _SENTENCE_COMPLETE_PUNCT_RE.search(text):
        return True

    if _ENGLISH_CONTINUATION_RE.search(text):
        log.debug("event=semantic_incomplete lang=en text=%.60s", text)
        return False

    if _URDU_CONTINUATION_RE.search(text):
        log.debug("event=semantic_incomplete lang=ur text=%.60s", text)
        return False

    return True


# --------------------------------------------------------------------------
# 2. Intent-aware turn filter  (ported verbatim from flux_groq_bridge.py)
# --------------------------------------------------------------------------

_SHORT_QUESTION_STARTERS: frozenset[str] = frozenset({
    "what", "why", "how", "when", "where", "who", "which",
    "are", "is", "was", "were",
    "do", "does", "did",
    "can", "could", "would", "should",
    "will", "have", "has", "had",
})

_SHORT_IMPERATIVES: frozenset[str] = frozenset({
    "tell", "explain", "show", "define", "describe",
    "list", "compare", "give",
})

_FILLER_WORDS: frozenset[str] = frozenset({
    "uh", "um", "hmm", "huh", "yeah", "okay", "ok",
    "right", "alright",
})


def is_meaningful_turn(transcript: str, consecutive_repeats: int) -> bool:
    """Return True when the transcript deserves an LLM response.

    Rules (strict priority):
        Rule 1 — empty → reject
        Rule 2 — consecutive_repeats >= 3 → reject
        Rule 3 — single filler word (English only) → reject
        Rule 4 — 2+ non-filler words → always accept (handles Hindi/Urdu short replies)
        Rule 5 — single non-filler word that ends with '?' → accept
    """
    normalised = transcript.strip().lower().rstrip(".,!?;:")
    words      = normalised.split()
    word_count = len(words)

    if word_count == 0:
        log.info("event=turn_filtered rule=empty")
        return False

    if consecutive_repeats >= 3:
        log.info("event=turn_filtered rule=consecutive_repeat count=%d", consecutive_repeats)
        return False

    if word_count == 1 and words[0] in _FILLER_WORDS:
        log.info("event=turn_filtered rule=filler transcript=%.40r", transcript)
        return False

    # Accept any 2+ word utterance — short Hindi/Urdu phrases like "हाں یار", "تو وہ" are valid
    if word_count >= 2:
        return True

    # Single non-filler word: only accept if it's a question
    if transcript.strip().endswith("?"):
        return True

    log.info("event=turn_filtered rule=single_word_not_question transcript=%.40r", transcript)
    return False


# --------------------------------------------------------------------------
# 3. HybridTurnGate FrameProcessor
# --------------------------------------------------------------------------

class _Phase(Enum):
    IDLE       = auto()   # waiting for speech
    LISTENING  = auto()   # user is speaking (accumulating transcript)
    WAITING    = auto()   # VAD silence detected; waiting for semantic completeness
    COMMITTED  = auto()   # LLM dispatched; ignoring further audio until bot finishes


class HybridTurnGate(FrameProcessor):
    """Hybrid VAD gate: acoustic (Silero, via LiveKitTransport) + semantic (regex).

    Frame contract
    ──────────────
    Upstream frames consumed (not forwarded):
      • UserStartedSpeakingFrame   → enter LISTENING phase
      • UserStoppedSpeakingFrame   → enter WAITING phase (start semantic timer)
      • TranscriptionFrame         → accumulate transcript text

    Frames forwarded downstream:
      • LLMMessagesFrame           → built from committed transcript
      • BotStartedSpeakingFrame    → reset to IDLE when bot finishes speaking
      • All other frames           → passed through transparent

    Semantic extension timer (FIX for double-trigger bug)
    ─────────────────────────────────────────────────────
    The old flux_groq_bridge.py used `continue` inside the VAD worker loop
    which could re-evaluate the semantic gate within the same 50ms tick after
    extending the deadline, causing an instant double-trigger.

    Here we use a monotonic *deadline float* and an asyncio periodic checker
    (`_semantic_check_task`).  The checker fires every 100ms and compares
    `time.monotonic() >= self._semantic_deadline`.  Assigning a new deadline
    moves the gate forward atomically — there is no window where both the
    old and new deadline evaluate as expired simultaneously.
    """

    def __init__(
        self,
        *,
        context_messages: list[dict],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context_messages = context_messages

        # Transcript accumulation
        self._finals: list[str]  = []
        self._latest_interim: str = ""

        # Phase state
        self._phase = _Phase.IDLE

        # Repeat suppression
        self._last_turn_norm: str = ""
        self._consecutive_repeats: int = 0

        # Semantic extension timer
        self._semantic_deadline: float = 0.0
        self._semantic_extensions: int = 0
        self._semantic_check_task: Optional[asyncio.Task] = None

        # Bot-speaking interlock (prevents commit while bot is mid-sentence)
        self._bot_speaking: bool = False

    # -----------------------------------------------------------------------
    # Pipecat FrameProcessor entry point
    # -----------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        # ── Acoustic VAD: user starts speaking ──────────────────────────────
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._on_user_started()
            return  # consumed

        # ── Acoustic VAD: user stops speaking ───────────────────────────────
        if isinstance(frame, UserStoppedSpeakingFrame):
            await self._on_user_stopped()
            return  # consumed

        # ── Deepgram transcript ──────────────────────────────────────────────
        if isinstance(frame, TranscriptionFrame):
            await self._on_transcription(frame)
            return  # consumed

        # ── Bot started / stopped speaking (barge-in interlock) ─────────────
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._phase == _Phase.COMMITTED:
                self._phase = _Phase.IDLE
                log.info("event=bot_finished → IDLE")
            await self.push_frame(frame, direction)
            return

        # ── Everything else passes through ───────────────────────────────────
        await self.push_frame(frame, direction)

    # -----------------------------------------------------------------------
    # VAD handlers
    # -----------------------------------------------------------------------

    async def _on_user_started(self) -> None:
        log.info("event=user_started_speaking phase=%s", self._phase.name)

        # Cancel any pending semantic wait — user is still talking
        self._cancel_semantic_timer()

        if self._phase not in (_Phase.COMMITTED,):
            self._phase = _Phase.LISTENING
            self._finals.clear()
            self._latest_interim = ""
            log.info("event=phase_change new=LISTENING")

        # Barge-in: if bot is speaking, interrupt it
        if self._bot_speaking:
            log.info("event=barge_in_triggered")
            await self.push_frame(StartInterruptionFrame())

    async def _on_user_stopped(self) -> None:
        log.info("event=user_stopped_speaking phase=%s", self._phase.name)
        if self._phase != _Phase.LISTENING:
            return

        self._phase = _Phase.WAITING
        self._semantic_extensions = 0
        # Set initial deadline: wait up to 3s for a transcript + semantic check
        self._semantic_deadline = time.monotonic() + 3.0
        log.info("event=phase_change new=WAITING deadline_sec=3.0")

        self._start_semantic_timer()

    # -----------------------------------------------------------------------
    # Transcript handler
    # -----------------------------------------------------------------------

    async def _on_transcription(self, frame: TranscriptionFrame) -> None:
        text = (frame.text or "").strip()
        if not text:
            return

        is_final = getattr(frame, "is_final", True)

        # ── Barge-in: fire as soon as any transcript arrives while bot speaks ──
        # This is transcript-driven, so it works even without VAD events.
        # InterruptionFrame is the current API (StartInterruptionFrame is deprecated).
        if self._bot_speaking:
            log.info("event=barge_in_triggered reason=transcript text=%.40s", text)
            await self.push_frame(InterruptionFrame())
            # Reset: cancel pending timer and start fresh LISTENING for this turn
            self._cancel_semantic_timer()
            self._phase = _Phase.LISTENING
            self._finals.clear()
            self._latest_interim = ""

        # If stuck in IDLE (VAD frames never arrived), auto-advance to LISTENING
        elif self._phase == _Phase.IDLE:
            log.info("event=phase_change new=LISTENING reason=transcript_auto_advance")
            self._phase = _Phase.LISTENING
            self._finals.clear()
            self._latest_interim = ""

        # Only accumulate while we care about the current utterance
        if self._phase not in (_Phase.LISTENING, _Phase.WAITING):
            return

        if is_final:
            log.info("event=transcript_final text=%.80s", text)
            self._finals.append(text)
            # If VAD never sent UserStoppedSpeakingFrame, use Deepgram's own
            # endpointing as the turn-end signal: start the semantic timer
            # on every final result while in LISTENING phase.
            if self._phase == _Phase.LISTENING:
                self._phase = _Phase.WAITING
                self._semantic_extensions = 0
                self._semantic_deadline = time.monotonic() + 2.5  # 2.5s — enough for Hindi/Urdu multi-clause sentences
                log.info("event=phase_change new=WAITING reason=deepgram_final deadline_sec=1.5")
                self._start_semantic_timer()
        else:
            self._latest_interim = text
            log.debug("event=transcript_interim text=%.80s", text)

    # -----------------------------------------------------------------------
    # Semantic extension timer  — FIX for double-trigger bug
    # -----------------------------------------------------------------------
    # Implementation:
    #   • A single asyncio.Task (`_semantic_check_task`) runs a 100ms poll loop.
    #   • On each tick it compares `time.monotonic() >= self._semantic_deadline`.
    #   • When deadline expires it evaluates semantic completeness ONCE and
    #     either extends the deadline (atomically, via a single float assignment)
    #     or commits.
    #   • A new UserStartedSpeakingFrame cancels the task — no possibility of
    #     evaluating a stale deadline after a new utterance begins.

    def _start_semantic_timer(self) -> None:
        if self._semantic_check_task and not self._semantic_check_task.done():
            return
        self._semantic_check_task = asyncio.create_task(
            self._semantic_check_loop(),
            name="hybrid_turn_gate_semantic_timer",
        )

    def _cancel_semantic_timer(self) -> None:
        if self._semantic_check_task and not self._semantic_check_task.done():
            self._semantic_check_task.cancel()
            self._semantic_check_task = None

    async def _semantic_check_loop(self) -> None:
        """Poll every 100ms.  Evaluate and extend/commit when deadline passes."""
        try:
            while True:
                await asyncio.sleep(0.1)  # 100ms tick — not blocking

                if self._phase != _Phase.WAITING:
                    return  # phase changed externally (e.g. new speech started)

                now = time.monotonic()
                if now < self._semantic_deadline:
                    continue  # deadline not yet reached; keep waiting

                # ── Deadline reached — evaluate ─────────────────────────────
                has_finals  = bool(self._finals)
                has_interim = bool(self._latest_interim.strip())

                if not has_finals and not has_interim:
                    log.warning("event=transcript_timeout no_text_available")
                    self._phase = _Phase.IDLE
                    self._reset_accumulation()
                    return

                committed_text = (
                    " ".join(self._finals) if has_finals else self._latest_interim
                )
                source = "finals" if has_finals else "interim_fallback"

                # ── Semantic gate ───────────────────────────────────────────
                if (
                    not is_semantically_complete(committed_text)
                    and self._semantic_extensions < MAX_SEMANTIC_EXTENSIONS
                ):
                    # Atomically shift deadline forward — the single float
                    # write is the ONLY state mutation; no re-entry possible
                    self._semantic_extensions += 1
                    self._semantic_deadline = time.monotonic() + SEMANTIC_EXTENSION_SEC
                    log.info(
                        "event=semantic_extension ext=%d/%d added_sec=%.1f text=%.60s",
                        self._semantic_extensions,
                        MAX_SEMANTIC_EXTENSIONS,
                        SEMANTIC_EXTENSION_SEC,
                        committed_text,
                    )
                    continue  # go back to sleep; new deadline is in the future

                # ── Commit ──────────────────────────────────────────────────
                log.info(
                    "event=commit source=%s text=%.80s", source, committed_text
                )
                self._reset_accumulation()
                await self._commit_turn(committed_text)
                return

        except asyncio.CancelledError:
            log.debug("event=semantic_timer_cancelled")
            raise

    # -----------------------------------------------------------------------
    # Turn commit
    # -----------------------------------------------------------------------

    async def _commit_turn(self, transcript: str) -> None:
        """Apply intent filter and push LLMMessagesFrame downstream."""
        if self._phase != _Phase.WAITING:
            return  # race guard

        # Update repeat counter
        norm = transcript.strip().lower().rstrip(".,!?;:")
        if norm == self._last_turn_norm:
            self._consecutive_repeats += 1
        else:
            self._consecutive_repeats = 0
            self._last_turn_norm = norm

        # Intent-aware filter
        if not is_meaningful_turn(transcript, self._consecutive_repeats):
            log.info("event=turn_filtered_at_commit")
            self._phase = _Phase.IDLE
            return

        self._phase = _Phase.COMMITTED
        log.info("event=turn_committed transcript_len=%d phase=COMMITTED", len(transcript))

        # Build messages for the LLM (system prompt + user turn)
        messages = list(self._context_messages) + [
            {"role": "user", "content": transcript}
        ]
        await self.push_frame(LLMMessagesFrame(messages))

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _reset_accumulation(self) -> None:
        self._finals.clear()
        self._latest_interim = ""


# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
#  EVENT LOOP STALL MONITOR
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

async def _stall_monitor() -> None:
    """Log a warning whenever the event loop blocks for > 150ms."""
    TICK_MS   = 100.0
    WARN_MS   = 150.0
    prev = time.perf_counter() * 1000.0
    while True:
        await asyncio.sleep(TICK_MS / 1000.0)
        now   = time.perf_counter() * 1000.0
        drift = now - prev - TICK_MS
        if drift > WARN_MS:
            log.warning("event=event_loop_stall stall_ms=%.1f", drift)
        prev = now


# ---------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════════
#  MAIN  —  bot entrypoint
# ══════════════════════════════════════════════════════════════════════════
# ---------------------------------------------------------------------------

async def main(room_name: str, token: str) -> None:
    log.info("event=bot_start room=%s", room_name)

    # ── Context messages (system prompt carried through every LLM call) ──────
    context_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── LiveKit Transport ────────────────────────────────────────────────────
    transport = LiveKitTransport(
        url=LIVEKIT_URL,
        token=token,
        room_name=room_name,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # ── STT: Deepgram Nova-3 (multilingual) ──────────────────────────────────
    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        live_options=LiveOptions(
            model="nova-3",
            language="multi",         # Urdu/English code-switching
            smart_format=True,
            punctuate=True,
            interim_results=True,
            endpointing=800,          # ms; VAD handles real turn detection
            encoding="linear16",
            sample_rate=16000,
        ),
    )

    # ── HybridTurnGate ───────────────────────────────────────────────────────
    gate = HybridTurnGate(context_messages=context_messages)

    # ── LLM: Groq (Llama-3.3) ───────────────────────────────────────────────
    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        model=GROQ_LLM_MODEL,
    )

    # ── TTS: ElevenLabs (turbo, multilingual) ────────────────────────────────
    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ELEVENLABS_VOICE_ID,
        model="eleven_turbo_v2_5",
    )

    # ── Pipeline assembly ────────────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),   # LiveKit mic frames + VAD events
        stt,                 # Audio → TranscriptionFrame
        gate,                # HybridTurnGate → LLMMessagesFrame
        llm,                 # LLMMessagesFrame → TextFrame (streaming tokens)
        tts,                 # TextFrame → Audio frames
        transport.output(),  # Audio → LiveKit speaker
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    # ── Transport event handlers ─────────────────────────────────────────────

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport_, participant):
        log.info("event=first_participant_joined participant=%s", participant)
        # Kick off: push system prompt as initial greeting trigger
        await task.queue_frame(
            LLMMessagesFrame([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "Hello! Greet the caller warmly and briefly in both Urdu and English, then ask how you can help."},
            ])
        )

    @transport.event_handler("on_participant_disconnected")
    async def on_participant_disconnected(transport_, participant):
        log.info("event=participant_disconnected participant=%s", participant)
        await task.queue_frame(EndFrame())

    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport_, state):
        log.info("event=call_state state=%s", state)
        if state == "ended":
            await task.queue_frame(EndFrame())

    # ── Run ──────────────────────────────────────────────────────────────────
    monitor = asyncio.create_task(_stall_monitor())
    try:
        runner = PipelineRunner()
        await runner.run(task)
    finally:
        monitor.cancel()
        try:
            await monitor
        except asyncio.CancelledError:
            pass
        log.info("event=bot_shutdown room=%s", room_name)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python bot.py <room_name> <token>", file=sys.stderr)
        sys.exit(1)

    room_name = sys.argv[1]
    token     = sys.argv[2]
    asyncio.run(main(room_name, token))
