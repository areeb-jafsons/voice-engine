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
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Voice engine config
from config import VoiceEngineConfig, DEFAULT_SYSTEM_PROMPT

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
    TextFrame,
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

# System prompt is now in config.py (DEFAULT_SYSTEM_PROMPT)
# It can be overridden at runtime via the /config API.

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

SEMANTIC_EXTENSION_SEC  = 1.2   # overridden at runtime from config
MAX_SEMANTIC_EXTENSIONS = 2     # overridden at runtime from config


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
        max_history_pairs: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._system_message = context_messages[0]   # always the system prompt
        self._context_messages = context_messages
        self._max_history_pairs = max_history_pairs  # keep last N user+assistant pairs
        self._history: list[dict] = []               # rolling [user, assistant, user, assistant...]

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

        # Bot-speaking interlock
        self._bot_speaking: bool = False

        # Partial assistant text accumulator (dead field, kept for _reset_accumulation compat)
        self._pending_assistant_text: list[str] = []

        # PipelineTask reference — set after task is created via set_task()
        # Used to queue InterruptionFrame directly, bypassing in-pipeline frame-holds.
        self._task: Optional[Any] = None

    def set_task(self, task: Any) -> None:
        """Wire in the PipelineTask so barge-in can queue frames directly."""
        self._task = task
        log.debug("event=gate_task_wired")

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

    async def _fire_interruption(self) -> None:
        """Interrupt the bot's speech via the fastest available path.

        1. task.queue_frame() — bypasses Pipecat's in-pipeline frame-holds
           (the safest path when allow_interruptions=True holds STT frames)
        2. Fallback: push_frame() through the pipeline (works when not blocked).
        """
        if self._task is not None:
            log.info("event=barge_in_queued_to_task")
            await self._task.queue_frame(InterruptionFrame())
        else:
            log.info("event=barge_in_pushed_to_pipeline")
            await self.push_frame(InterruptionFrame())

    async def _on_user_started(self) -> None:
        log.info("event=user_started_speaking phase=%s", self._phase.name)

        # Cancel any pending semantic wait — user is still talking
        self._cancel_semantic_timer()

        if self._phase not in (_Phase.COMMITTED,):
            self._phase = _Phase.LISTENING
            self._finals.clear()
            self._latest_interim = ""
            log.info("event=phase_change new=LISTENING")

        # Barge-in: if bot is speaking, fire InterruptionFrame IMMEDIATELY on VAD onset.
        # This fires *before* any transcript arrives — typically <50ms after speech begins.
        if self._bot_speaking:
            log.info("event=vad_barge_in phase=%s", self._phase.name)
            await self._fire_interruption()

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

        # ── Barge-in while bot is speaking ──────────────────────────────────────────
        if self._bot_speaking and len(text) >= 2:
            log.info("event=barge_in_triggered reason=transcript text=%.40s", text)
            self._cancel_semantic_timer()
            self._phase = _Phase.LISTENING
            self._finals.clear()
            self._latest_interim = ""
            await self._fire_interruption()
            # Continue below to accumulate the transcript for the next turn

        # ── Transcript during COMMITTED gap (bot about to speak, not yet speaking) ──
        # If the user speaks in the ~1-2s gap between commit and BotStartedSpeakingFrame,
        # don't drop the transcript silently — treat it as a new LISTENING turn.
        elif self._phase == _Phase.COMMITTED and not self._bot_speaking:
            log.info("event=committed_gap_transcript text=%.40s — treating as new turn", text)
            self._phase = _Phase.LISTENING
            self._finals.clear()
            self._latest_interim = ""

        # ── Auto-advance from IDLE if VAD frames never arrived ───────────────────
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
    # -----------------------------------------------------------------------
    # Turn commit
    # -----------------------------------------------------------------------

    def add_assistant_response(self, text: str) -> None:
        """Called after the bot finishes speaking to record its response in history."""
        self._history.append({"role": "assistant", "content": text})
        # Trim: keep at most max_history_pairs x 2 messages (user+assistant each)
        max_msgs = self._max_history_pairs * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]
        log.debug("event=history_updated history_len=%d", len(self._history))

    def _build_messages(self, user_transcript: str) -> list[dict]:
        """Build full message list: system + rolling history + current user turn."""
        return [self._system_message] + self._history + [{"role": "user", "content": user_transcript}]

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
        log.info("event=turn_committed transcript_len=%d phase=COMMITTED history_pairs=%d",
                 len(transcript), len(self._history) // 2)

        # Add this user message to history immediately (assistant response added on BotStopped)
        self._history.append({"role": "user", "content": transcript})

        # Build messages: system + rolling history (includes current user msg)
        messages = self._build_messages_with_history()
        await self.push_frame(LLMMessagesFrame(messages))

    def _build_messages_with_history(self) -> list[dict]:
        """Build [system] + [last N history] — current user is already appended in history."""
        max_msgs = self._max_history_pairs * 2
        trimmed_history = self._history[-max_msgs:] if len(self._history) > max_msgs else list(self._history)
        return [self._system_message] + trimmed_history

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _reset_accumulation(self) -> None:
        self._finals.clear()
        self._latest_interim = ""
        self._pending_assistant_text.clear()


# ---------------------------------------------------------------------------
# 4. AssistantHistoryCapture FrameProcessor
# ---------------------------------------------------------------------------

class AssistantHistoryCapture(FrameProcessor):
    """Sits between LLM and TTS to capture the assistant response text.

    TextFrame tokens flow DOWNSTREAM (LLM → TTS), so this processor
    must be placed AFTER the LLM in the pipeline to intercept them.
    On LLMFullResponseEndFrame it commits the accumulated text to the
    HybridTurnGate's rolling history via gate.add_assistant_response().
    """

    def __init__(self, gate: HybridTurnGate, **kwargs):
        super().__init__(**kwargs)
        self._gate = gate
        self._buffer: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            self._buffer.append(frame.text)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._buffer:
                full_text = "".join(self._buffer).strip()
                self._buffer.clear()
                if full_text:
                    log.debug("event=assistant_response_captured len=%d", len(full_text))
                    self._gate.add_assistant_response(full_text)
            await self.push_frame(frame, direction)
            return

        # On interruption: discard the partial buffer — response was cut short
        if isinstance(frame, InterruptionFrame):
            self._buffer.clear()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


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

async def main(room_name: str, token: str, config: VoiceEngineConfig) -> None:
    log.info("event=bot_start room=%s", room_name)

    # Apply turn gate config to module-level constants
    global SEMANTIC_EXTENSION_SEC, MAX_SEMANTIC_EXTENSIONS
    SEMANTIC_EXTENSION_SEC  = config.turn_gate.semantic_extension_sec
    MAX_SEMANTIC_EXTENSIONS = config.turn_gate.max_semantic_extensions
    log.info(
        "event=turn_gate_config semantic_ext=%.1fs max_ext=%d",
        SEMANTIC_EXTENSION_SEC, MAX_SEMANTIC_EXTENSIONS,
    )

    # System prompt from config
    system_prompt = config.system_prompt

    # ── Context messages (system prompt carried through every LLM call) ──────
    context_messages: list[dict] = [{"role": "system", "content": system_prompt}]

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
    dgcfg = config.deepgram
    live_opts_kwargs = {
        "model": dgcfg.model,
        "language": dgcfg.language,
        "smart_format": dgcfg.smart_format,
        "punctuate": dgcfg.punctuate,
        "interim_results": dgcfg.interim_results,
        "endpointing": dgcfg.endpointing,
        "encoding": "linear16",
        "sample_rate": 16000,
    }
    # Add optional Deepgram parameters only if explicitly set
    if dgcfg.utterance_end_ms is not None:
        live_opts_kwargs["utterance_end_ms"] = str(dgcfg.utterance_end_ms)
    if dgcfg.filler_words is not None:
        live_opts_kwargs["filler_words"] = dgcfg.filler_words
    if dgcfg.keywords is not None:
        live_opts_kwargs["keywords"] = dgcfg.keywords
    if dgcfg.diarize is not None:
        live_opts_kwargs["diarize"] = dgcfg.diarize
    if dgcfg.numerals is not None:
        live_opts_kwargs["numerals"] = dgcfg.numerals
    if dgcfg.profanity_filter is not None:
        live_opts_kwargs["profanity_filter"] = dgcfg.profanity_filter

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        live_options=LiveOptions(**live_opts_kwargs),
    )
    log.info("event=stt_config model=%s language=%s endpointing=%d", dgcfg.model, dgcfg.language, dgcfg.endpointing)

    # ── HybridTurnGate ───────────────────────────────────────────────────────
    gate = HybridTurnGate(context_messages=context_messages)

    # ── LLM: Groq ────────────────────────────────────────────────────────────
    gcfg = config.groq
    llm_input_params_kwargs = {}
    if gcfg.temperature is not None:
        llm_input_params_kwargs["temperature"] = gcfg.temperature
    if gcfg.top_p is not None:
        llm_input_params_kwargs["top_p"] = gcfg.top_p
    if gcfg.max_tokens is not None:
        llm_input_params_kwargs["max_tokens"] = gcfg.max_tokens
    if gcfg.frequency_penalty is not None:
        llm_input_params_kwargs["frequency_penalty"] = gcfg.frequency_penalty
    if gcfg.presence_penalty is not None:
        llm_input_params_kwargs["presence_penalty"] = gcfg.presence_penalty
    if gcfg.seed is not None:
        llm_input_params_kwargs["seed"] = gcfg.seed

    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        model=gcfg.model,
        **llm_input_params_kwargs,   # temperature, top_p, max_tokens, etc. → forwarded to Groq client
    )
    log.info("event=llm_config model=%s params=%s", gcfg.model, llm_input_params_kwargs or 'defaults')

    # ── TTS: ElevenLabs (turbo, multilingual) ────────────────────────────────
    ecfg = config.elevenlabs
    tts_input_params_kwargs = {}
    if ecfg.stability is not None:
        tts_input_params_kwargs["stability"] = ecfg.stability
    if ecfg.similarity_boost is not None:
        tts_input_params_kwargs["similarity_boost"] = ecfg.similarity_boost
    if ecfg.style is not None:
        tts_input_params_kwargs["style"] = ecfg.style
    if ecfg.use_speaker_boost is not None:
        tts_input_params_kwargs["use_speaker_boost"] = ecfg.use_speaker_boost
    if ecfg.speed is not None:
        tts_input_params_kwargs["speed"] = ecfg.speed
    if ecfg.language is not None:
        tts_input_params_kwargs["language"] = ecfg.language

    tts = ElevenLabsTTSService(
        api_key=ELEVENLABS_API_KEY,
        voice_id=ecfg.voice_id,
        model=ecfg.model,
        params=ElevenLabsTTSService.InputParams(**tts_input_params_kwargs) if tts_input_params_kwargs else None,
    )
    log.info("event=tts_config voice=%s model=%s params=%s", ecfg.voice_id, ecfg.model, tts_input_params_kwargs or 'defaults')

    # ── Pipeline assembly ────────────────────────────────────────────────────
    history_capture = AssistantHistoryCapture(gate=gate)

    pipeline = Pipeline([
        transport.input(),   # LiveKit mic frames + VAD events
        stt,                 # Audio → TranscriptionFrame
        gate,                # HybridTurnGate → LLMMessagesFrame
        llm,                 # LLMMessagesFrame → TextFrame (streaming tokens)
        history_capture,     # Captures TextFrames → gate.add_assistant_response()
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

    # Wire the task into the gate so barge-in can bypass the in-pipeline frame-hold
    gate.set_task(task)

    # ── Transport event handlers ─────────────────────────────────────────────

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport_, participant):
        log.info("event=first_participant_joined participant=%s", participant)
        # Kick off: push system prompt as initial greeting trigger
        await task.queue_frame(
            LLMMessagesFrame([
                {"role": "system", "content": system_prompt},
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

    # Read config from stdin (one JSON line sent by server.py)
    config_line = ""
    try:
        import select
        # On Windows, select doesn't work on stdin, so just try readline
        config_line = sys.stdin.readline().strip()
    except Exception:
        pass

    if config_line:
        try:
            _config = VoiceEngineConfig.model_validate_json(config_line)
            log.info("event=config_received_from_server")
        except Exception as exc:
            log.warning("event=config_parse_error error=%s — using defaults", exc)
            _config = VoiceEngineConfig()
    else:
        log.info("event=no_config_on_stdin — using defaults")
        _config = VoiceEngineConfig()

    asyncio.run(main(room_name, token, _config))
