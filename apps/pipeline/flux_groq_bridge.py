
import os as _os
DEEPGRAM_API_KEY = _os.environ["DEEPGRAM_API_KEY"]
GROQ_API_KEY     = _os.environ["GROQ_API_KEY"]


import asyncio
import websockets
import json
import time
import io


def now_ms() -> float:
    """Return a high-resolution monotonic timestamp in milliseconds."""
    return time.perf_counter() * 1000.0
import re
import logging
import soundfile as sf
import numpy as np
import scipy.signal
from enum import Enum
from groq import Groq
import sounddevice as sd
import threading
import queue
import os
import random
from dotenv import load_dotenv
from semantic_validator import validate_turn_semantics

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------
log = logging.getLogger("voice_engine")

_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
))
log.addHandler(_log_handler)
log.setLevel(logging.DEBUG if os.environ.get("VOICE_DEBUG") else logging.INFO)

SENTENCE_END_RE = re.compile(r'[.!?](?:\s|$)')
COMMA_SPLIT_RE = re.compile(r',\s')
MIN_CHUNK_CHARS = 20
LONG_CLAUSE_CHARS = 60
TOKEN_IDLE_TIMEOUT = 0.4  # seconds
LLM_FIRST_TOKEN_TIMEOUT = 3.0  # seconds
LLM_TOTAL_TIMEOUT = 10.0  # seconds
TTS_SENTENCE_TIMEOUT = 5.0  # seconds
WS_RECONNECT_ATTEMPTS = 5
WS_RECONNECT_DELAY = 1.0  # seconds

# ---------------------------------------------------------------------------
# Echo suppression — mic-to-STT gate during SPEAKING
# ---------------------------------------------------------------------------
# During SPEAKING, mic audio is replaced with silence before being sent to
# Deepgram, preventing TTS playback from being misidentified as user speech.
#
# Exception: if the raw RMS of a chunk exceeds BARGE_IN_RMS_THRESHOLD, the
# real audio is forwarded anyway — a genuinely loud human voice can still
# interrupt.  This threshold is deliberately high (well above typical TTS
# bleed-through) so only confident human speech breaks through.
#
# The mic InputStream keeps running at all times so there is zero restart
# latency when the gate reopens after SPEAKING ends.
BARGE_IN_RMS_THRESHOLD = 3000   # int16 RMS — loud human voice during TTS playback

# ---------------------------------------------------------------------------
# SpeechIntentAnalyzer constants
# ---------------------------------------------------------------------------
GRACE_WINDOW_SEC = 0.3           # 300 ms base soft-commit window (syntactically complete turns)
EXTENDED_GRACE_SEC = 0.7         # 700 ms grace window for syntactically incomplete turns
ADAPTIVE_GRACE_SHORT_SEC = 0.2   # 200 ms fast-commit for very short turns (< 5 words)
ADAPTIVE_GRACE_LONG_SEC  = 0.7   # 700 ms extended commit for complex / long turns
ANALYZER_HOLD_FALLBACK_SEC = 0.8  # fallback commit if analyzer holds and no resume arrives
ANALYZER_HOLD_TIMEOUT_SEC  = 1.2  # safety ceiling: force-commit if analyzer holds > 1.2 s

# ---------------------------------------------------------------------------
# Semantic gate constants
# ---------------------------------------------------------------------------
SEMANTIC_GATE_EXTEND_INCOMPLETE_SEC = 1.2  # re-arm hold when validator returns INCOMPLETE
SEMANTIC_GATE_EXTEND_CONTINUE_SEC   = 0.8  # extend hold when validator returns CONTINUE_LIKELY
# Invocation criteria: call validator when ANY of these is true.
SEMANTIC_GATE_MAX_CHARS  = 40   # transcript shorter than this
SEMANTIC_GATE_MAX_WORDS  = 6    # word count at or below this

# ---------------------------------------------------------------------------
# Turn classifier constants
# ---------------------------------------------------------------------------
# The micro-classifier makes a single Groq API call to label each turn.
# It races against CLASSIFIER_TIMEOUT_SEC; if it loses, the heuristic
# (compute_adaptive_grace) is used instead — so this is always best-effort.
CLASSIFIER_TIMEOUT_SEC      = 0.10  # 100 ms hard ceiling for classifier round-trip
#CLASSIFIER_MODEL            = "llama-3.1-8b-instant"   # fastest available on Groq
CLASSIFIER_MODEL            = "openai/gpt-oss-20b"

# Grace delays keyed by classifier label.  COMPLETE uses compute_adaptive_grace
# (heuristic), so it has no fixed entry here — the caller falls through to it.
CLASSIFIER_GRACE_CONTINUATION = 0.80   # 800 ms — user is mid-thought
CLASSIFIER_GRACE_HESITATION   = 0.50   # 500 ms — filler/pause, give more time
# REPAIR → no grace timer; triggers transition_to_listening() immediately.
# COMPLETE → falls through to compute_adaptive_grace() (heuristic default).

# Words that signal a subordinate clause — used by compute_adaptive_grace()
# to detect complex sentences that need a longer commit window.
# Kept separate from _SC_* sets so each function can evolve independently.
_SUBORDINATE_CLAUSE_WORDS: frozenset[str] = frozenset({
    "because", "although", "though", "even", "since", "while",
    "unless", "until", "if", "when", "whereas", "after", "before",
    "whether", "however", "therefore", "nevertheless",
})

# Word-sets used exclusively by is_syntactically_complete().
# Deliberately separate from CONTINUATION_WORDS / AUXILIARY_VERBS so each
# function's vocabulary can evolve independently.
_SC_TRAILING_CONJUNCTIONS: frozenset[str] = frozenset({
    "and", "but", "so", "because", "if", "when", "although", "though",
    "while", "whereas", "unless", "until", "since", "after", "before",
})
_SC_TRAILING_PREPOSITIONS: frozenset[str] = frozenset({
    "to", "at", "for", "with", "about", "into", "from", "on",
    "in", "of", "by", "over", "under", "through", "between",
})
_SC_CONTINUATION_MARKERS: frozenset[str] = frozenset({
    "actually", "well", "then", "basically", "literally",
    "honestly", "anyway", "regardless",
})
_SC_TRAILING_AUXILIARIES: frozenset[str] = frozenset({
    "is", "are", "was", "were", "do", "does", "did",
    "have", "has", "had", "will", "would", "could", "should",
})
# Open conditional openers: if these two-token sequences appear without a
# following main clause, the utterance is likely mid-sentence.
_SC_OPEN_CONDITIONALS: tuple[tuple[str, str], ...] = (
    ("if", "you"), ("if", "i"), ("if", "we"), ("if", "they"),
    ("when", "i"), ("when", "you"), ("when", "we"),
    ("whenever", "i"), ("whenever", "you"),
)
MIN_PAUSE_THRESHOLD_MS = 500
MAX_PAUSE_THRESHOLD_MS = 1200
ROLLING_WINDOW = 5  # history size for tempo / pause averages
CONFIDENCE_HOLD_THRESHOLD = 0.75
IMPLICIT_QUESTION_PAUSE_MS = 500
LENGTH_FALLBACK_WORDS = 8
LENGTH_FALLBACK_PAUSE_MS = 600

CONTINUATION_WORDS = frozenset({
    "and", "but", "so", "because", "if", "when", "while",
    "although", "to", "for", "with", "that", "which",
    "who", "whose", "where", "uh", "um", "like",
})

AUXILIARY_VERBS = frozenset({
    "are", "do", "does", "did", "can", "could", "would",
    "should", "is", "was", "were", "have", "has",
})

# ---------------------------------------------------------------------------
# Production hardening constants
# ---------------------------------------------------------------------------
MAX_HISTORY_TOKENS = 2000
HEALTH_CHECK_INTERVAL = 60.0  # seconds
HEALTH_STALE_THRESHOLD = 30.0  # seconds

# ---------------------------------------------------------------------------
# Barge-in confidence gate
# ---------------------------------------------------------------------------
MIN_INTERRUPT_WORDS    = 4      # transcript must have ≥ N meaningful words
MIN_INTERRUPT_AUDIO_MS = 180    # OR sustained speech must be ≥ N ms

# Repair phrases: utterances that begin with any of these signal the user is
# correcting themselves or asking the engine to stop and reset.  Multi-word
# entries are stored as-is; matching is prefix-based (see is_repair_phrase).
# Single-word entries cover the common one-word case ("wait", "no", etc.).
REPAIR_PHRASES: frozenset[str] = frozenset({
    "wait",
    "no",
    "sorry",
    "actually",
    "hold on",
    "let me rephrase",
    "i mean",
    "correction",
})


def interrupt_is_strong(transcript: str, speech_duration_ms: float) -> bool:
    """Return True only when a StartOfTurn during SPEAKING is intentional speech.

    Two independent pass conditions (OR logic — either alone is sufficient):
      1. Word count  — transcript already has ≥ MIN_INTERRUPT_WORDS words.
         Filters out single-word echoes and sub-phoneme noise bursts.
      2. Duration    — the audio segment has run for ≥ MIN_INTERRUPT_AUDIO_MS.
         Catches cases where ASR hasn't returned words yet but the mic has
         clearly been active long enough to be deliberate speech.

    Stopwords / filler words are intentionally counted; "uh can you stop" is
    four words and a legitimate barge-in.  Only pure whitespace is stripped.

    Args:
        transcript:        Partial transcript received at StartOfTurn (may be "").
        speech_duration_ms: Milliseconds elapsed since the current audio segment
                           started (caller computes: (now - turn_start_time)*1000).

    Returns:
        True  → interrupt is confident; proceed to transition_to_listening().
        False → likely echo / noise; silently discard the StartOfTurn.
    """
    word_count = len(transcript.split())
    if word_count >= MIN_INTERRUPT_WORDS:
        log.debug(
            "event=barge_in_gate decision=allow reason=word_count "
            "words=%d min=%d duration_ms=%.0f",
            word_count, MIN_INTERRUPT_WORDS, speech_duration_ms,
        )
        return True

    if speech_duration_ms >= MIN_INTERRUPT_AUDIO_MS:
        log.debug(
            "event=barge_in_gate decision=allow reason=duration "
            "words=%d duration_ms=%.0f min_ms=%d",
            word_count, speech_duration_ms, MIN_INTERRUPT_AUDIO_MS,
        )
        return True

    log.debug(
        "event=barge_in_gate decision=reject "
        "words=%d min_words=%d duration_ms=%.0f min_ms=%d",
        word_count, MIN_INTERRUPT_WORDS, speech_duration_ms, MIN_INTERRUPT_AUDIO_MS,
    )
    return False


# ---------------------------------------------------------------------------
# Syntactic completion detector
# ---------------------------------------------------------------------------

def is_syntactically_complete(text: str) -> bool:
    """Return False when the utterance is structurally mid-sentence.

    Evaluated in priority order.  The first matching rule wins.

    Rule 1 — Trailing conjunction
        Last token is a subordinating/coordinating conjunction.
        "I want to go and" → False.

    Rule 2 — Trailing preposition
        Last token is a preposition.
        "I was looking at" → False.

    Rule 3 — Trailing continuation marker
        Last token is a discourse marker that signals more is coming.
        "I mean actually" → False.

    Rule 4 — Trailing auxiliary verb
        Last token is a bare auxiliary with no following predicate.
        "The answer is" / "They were" → False.

    Rule 5 — Open conditional clause without completion
        The bigram sequence ("if"/"when" + pronoun) appears but no
        main-clause verb follows the conditional opener.
        "if you" / "when I" → False.
        "if you want I can help" → True (main clause present).

    Rule 6 — Ellipsis / trailing fragment
        Text ends with "..." or a lone em-dash "—", indicating the
        speaker trailed off mid-thought.

    Short utterances (< 2 words) fall through to True — they are
    handled upstream by is_meaningful_turn() before this is called.

    Args:
        text: Raw transcript string (may contain punctuation).

    Returns:
        True  → syntactically complete; use normal GRACE_WINDOW_SEC.
        False → mid-sentence; extend to EXTENDED_GRACE_SEC.
    """
    stripped = text.strip()
    if not stripped:
        return True

    lower = stripped.lower()
    # Remove trailing punctuation for token analysis, but keep "..." detector first
    # Rule 6 — ellipsis / trailing dash (check before stripping)
    if lower.endswith("...") or lower.endswith("…") or lower.endswith(" —") or lower.endswith("—"):
        log.debug("event=syntactic_check result=incomplete rule=ellipsis text=%.50r", text)
        return False

    # Normalise: strip sentence-end punctuation for clean last-token check
    normalised = lower.rstrip(".,!?;:")
    tokens = normalised.split()
    if not tokens:
        return True

    last = tokens[-1]

    # Rule 1 — trailing conjunction
    if last in _SC_TRAILING_CONJUNCTIONS:
        log.debug("event=syntactic_check result=incomplete rule=trailing_conjunction token=%r text=%.50r", last, text)
        return False

    # Rule 2 — trailing preposition
    if last in _SC_TRAILING_PREPOSITIONS:
        log.debug("event=syntactic_check result=incomplete rule=trailing_preposition token=%r text=%.50r", last, text)
        return False

    # Rule 3 — trailing continuation marker
    if last in _SC_CONTINUATION_MARKERS:
        log.debug("event=syntactic_check result=incomplete rule=continuation_marker token=%r text=%.50r", last, text)
        return False

    # Rule 4 — trailing auxiliary verb
    if last in _SC_TRAILING_AUXILIARIES:
        log.debug("event=syntactic_check result=incomplete rule=trailing_auxiliary token=%r text=%.50r", last, text)
        return False

    # Rule 5 — open conditional clause
    # Detect ("if"/"when" + pronoun) bigram anywhere in the token list.
    # Then verify whether a main-clause verb follows the opener bigram.
    # If the conditional opener is the last two tokens, no main clause exists.
    if len(tokens) >= 2:
        for opener, pronoun in _SC_OPEN_CONDITIONALS:
            try:
                idx = tokens.index(opener)
            except ValueError:
                continue
            # opener found — check if pronoun immediately follows
            if idx + 1 < len(tokens) and tokens[idx + 1] == pronoun:
                # Is this bigram at the very tail? (no tokens after pronoun)
                if idx + 1 == len(tokens) - 1:
                    log.debug(
                        "event=syntactic_check result=incomplete rule=open_conditional "
                        "opener=%r pronoun=%r text=%.50r", opener, pronoun, text,
                    )
                    return False
                # Bigram exists but tokens follow — treat as complete
                break

    log.debug("event=syntactic_check result=complete text=%.50r", text)
    return True


# ---------------------------------------------------------------------------
# Adaptive grace window
# ---------------------------------------------------------------------------

def compute_adaptive_grace(transcript: str) -> float:
    """Return the soft-commit delay (seconds) tuned to the transcript's complexity.

    Decision tiers (evaluated in priority order — first match wins):

    Tier 0 — Syntactically incomplete (is_syntactically_complete → False)
        Always use ADAPTIVE_GRACE_LONG_SEC regardless of other factors.
        A mid-sentence utterance needs the maximum window; the speaker is
        demonstrably still forming the thought.

    Tier 1 — Long / complex turn
        word_count > 12  OR  transcript contains a comma  OR  any token in
        _SUBORDINATE_CLAUSE_WORDS appears in the normalised token list.
        → ADAPTIVE_GRACE_LONG_SEC (700 ms)
        Rationale: dense information needs time for a natural breath pause
        before the user truly stops.

    Tier 2 — Very short turn
        word_count < 5
        → ADAPTIVE_GRACE_SHORT_SEC (200 ms)
        Rationale: short commands / questions are unlikely to have a trailing
        clause; a fast commit reduces perceived latency.

    Tier 3 — Default
        → GRACE_WINDOW_SEC (300 ms)

    Args:
        transcript: Raw transcript string as received from STT.

    Returns:
        Delay in seconds for asyncio.sleep inside _grace_commit.
    """
    stripped = transcript.strip()
    if not stripped:
        return GRACE_WINDOW_SEC  # empty — handled upstream; use base

    # Normalise for token analysis (lowercase, strip trailing punctuation)
    normalised = stripped.lower().rstrip(".,!?;:")
    tokens = normalised.split()
    word_count = len(tokens)

    # Tier 0 — syntactic incompleteness takes precedence over everything
    if not is_syntactically_complete(transcript):
        log.info(
            "event=grace_tier_selected tier=0 delay_ms=%d words=%d "
            "has_comma=False has_subordinate=False syntactic_complete=False",
            int(ADAPTIVE_GRACE_LONG_SEC * 1000), word_count,
        )
        return ADAPTIVE_GRACE_LONG_SEC

    # Tier 1 — long or structurally complex turn
    has_comma      = "," in stripped
    has_subclause  = bool(_SUBORDINATE_CLAUSE_WORDS.intersection(tokens))
    is_long        = word_count > 12

    if is_long or has_comma or has_subclause:
        reason = (
            "word_count"  if is_long      else
            "comma"       if has_comma    else
            "subclause"
        )
        log.info(
            "event=grace_tier_selected tier=1 delay_ms=%d words=%d "
            "has_comma=%s has_subordinate=%s syntactic_complete=True reason=%s",
            int(ADAPTIVE_GRACE_LONG_SEC * 1000), word_count,
            has_comma, has_subclause, reason,
        )
        return ADAPTIVE_GRACE_LONG_SEC

    # Tier 2 — very short turn
    if word_count < 5:
        log.info(
            "event=grace_tier_selected tier=2 delay_ms=%d words=%d "
            "has_comma=%s has_subordinate=%s syntactic_complete=True",
            int(ADAPTIVE_GRACE_SHORT_SEC * 1000), word_count,
            has_comma, has_subclause,
        )
        return ADAPTIVE_GRACE_SHORT_SEC

    # Tier 3 — default base window
    log.info(
        "event=grace_tier_selected tier=3 delay_ms=%d words=%d "
        "has_comma=%s has_subordinate=%s syntactic_complete=True",
        int(GRACE_WINDOW_SEC * 1000), word_count,
        has_comma, has_subclause,
    )
    return GRACE_WINDOW_SEC


# ---------------------------------------------------------------------------
# Low-signal turn filter — intent-aware
# ---------------------------------------------------------------------------

# Question-opener words: a turn starting with any of these is a candidate
# conversational request regardless of length.
_SHORT_QUESTION_STARTERS: frozenset[str] = frozenset({
    "what", "why", "how", "when", "where", "who", "which",
    "are", "is", "was", "were",
    "do", "does", "did",
    "can", "could", "would", "should",
    "will", "have", "has", "had",
})

# Short imperative verbs: a turn starting with these is a command fragment
# that the LLM can act on even without a full sentence.
_SHORT_IMPERATIVES: frozenset[str] = frozenset({
    "tell", "explain", "show", "define", "describe",
    "list", "compare", "give",
})

# Pure filler words: a single-token turn matching one of these is noise.
# Multi-token turns are not rejected by this set alone — see Rule 3 below.
_FILLER_WORDS: frozenset[str] = frozenset({
    "uh", "um", "hmm", "huh", "yeah", "okay", "ok",
    "right", "alright",
})


def is_meaningful_turn(transcript: str, consecutive_repeats: int) -> bool:
    """Return True when the transcript deserves an LLM response.

    Hard word-count floors are removed.  Short turns are accepted or
    rejected based on conversational intent rather than length alone.

    Rules evaluated in strict priority order:

    Rule 1 — Empty transcript
        Nothing to process → reject.

    Rule 2 — Consecutive repeat suppression
        Same normalised string submitted ≥ 3 times in a row → reject.
        Checked before content rules so a repeated filler does not
        leak through on the third occurrence.

    Rule 3 — Single filler word
        Exactly one token AND that token is in _FILLER_WORDS → reject.
        "uh", "hmm", "okay" alone carry no intent.

    Rule 4 — Short-turn intent gate  (word_count <= 2)
        Short turns are accepted only when they carry clear intent:
          4a. First word in _SHORT_QUESTION_STARTERS → accept.
              "What are", "Can you", "Is it" are valid questions.
          4b. First word in _SHORT_IMPERATIVES → accept.
              "Tell me", "Explain this" are valid commands.
          4c. Transcript ends with "?" → accept.
              Handles punctuated fragments the starter sets might miss.
          4d. Otherwise → reject as ambient noise fragment.

    Rule 5 — Three or more words → accept unconditionally.
        Upstream filters (barge-in gate, syntactic checker, analyzer)
        handle quality from here; length alone is no longer a disqualifier.

    Args:
        transcript:          Raw transcript string from Deepgram.
        consecutive_repeats: Times this exact normalised string has been
                             submitted consecutively (VoiceSession owns counter).

    Returns:
        True  → proceed to LLM.
        False → discard; stay in LISTENING / IDLE.
    """
    normalised = transcript.strip().lower().rstrip(".,!?;:")
    words = normalised.split()
    word_count = len(words)

    # Rule 1 — empty
    if word_count == 0:
        log.info("event=turn_filtered rule=empty transcript=%.40r", transcript)
        return False

    # Rule 2 — consecutive repeat
    if consecutive_repeats >= 3:
        log.info(
            "event=turn_filtered rule=consecutive_repeat count=%d transcript=%.40r",
            consecutive_repeats, transcript,
        )
        return False

    # Rule 3 — single filler word
    if word_count == 1 and words[0] in _FILLER_WORDS:
        log.info("event=turn_filtered rule=filler transcript=%.40r", transcript)
        return False

    # Rule 4 — short-turn intent gate
    if word_count <= 2:
        first = words[0]

        if first in _SHORT_QUESTION_STARTERS:
            log.debug(
                "event=short_turn_accept rule=question_start transcript=%.40r",
                transcript,
            )
            return True

        if first in _SHORT_IMPERATIVES:
            log.debug(
                "event=short_turn_accept rule=imperative transcript=%.40r",
                transcript,
            )
            return True

        if transcript.strip().endswith("?"):
            log.debug(
                "event=short_turn_accept rule=question_mark transcript=%.40r",
                transcript,
            )
            return True

        log.info(
            "event=turn_filtered rule=short_noise transcript=%.40r",
            transcript,
        )
        return False

    # Rule 5 — three or more words → accept
    return True


# ---------------------------------------------------------------------------
# Soak test & failure simulation flags
# ---------------------------------------------------------------------------
SOAK_TEST_MODE = os.getenv("SOAK_TEST", "false").lower() == "true"
SIMULATE_WIFI_DROP = os.getenv("SIM_WIFI", "false") == "true"
SIMULATE_LLM_TIMEOUT = os.getenv("SIM_LLM_TIMEOUT", "false") == "true"
SIMULATE_TTS_TIMEOUT = os.getenv("SIM_TTS_TIMEOUT", "false") == "true"
SIMULATE_MIC_DROP = os.getenv("SIM_MIC", "false") == "true"


class TurnState(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"

load_dotenv()


# ---------------------------------------------------------------------------
# Repair phrase detector
# ---------------------------------------------------------------------------

def is_repair_phrase(transcript: str) -> bool:
    """Return True when the transcript *begins* with a repair phrase.

    A repair phrase signals that the user is correcting themselves, asking
    the engine to stop, or restarting their thought.  Examples:
        "wait, that's not right"   → True   (prefix "wait")
        "no actually I meant …"    → True   (prefix "no")
        "hold on let me think"     → True   (prefix "hold on")
        "I hold the opinion that"  → False  ("hold" is not at start)
        "actually" alone           → True   (exact single-word match)

    Design notes
    ────────────
    • Prefix matching only — the phrase must appear at the *start* of the
      normalised transcript.  This prevents false positives from incidental
      usage ("I said no to that yesterday", "she actually left early").

    • Multi-word phrases (e.g. "hold on", "i mean") are checked in a single
      startswith() call against the normalised string.  Because we sort
      longer phrases first and break on the first hit, "hold on" cannot be
      masked by a hypothetical single-word "hold" entry.

    • Normalisation strips leading/trailing whitespace, lowercases, and
      removes leading punctuation artefacts (STT sometimes prepends a comma
      or dash on resumed speech).  No stemming or fuzzy matching — keeping
      this deterministic and zero-latency.

    • The function is pure (no I/O, no side effects) and callable from any
      context — sync or async.

    Args:
        transcript: Raw transcript string as received from STT.

    Returns:
        True  → utterance starts with a repair phrase; caller should reset.
        False → normal utterance; no special handling needed.
    """
    if not transcript:
        return False

    # Normalise: lowercase, strip edges, remove leading punctuation artefacts
    # (STT on resumed speech can emit ", actually" or "- wait").
    normalised = transcript.strip().lower().lstrip(",-– ")

    if not normalised:
        return False

    # Sort by length descending so multi-word phrases are tested before any
    # single-word prefix that shares a prefix (e.g. "hold on" before "hold").
    for phrase in sorted(REPAIR_PHRASES, key=len, reverse=True):
        if normalised == phrase:
            log.debug(
                "event=repair_phrase_detected match=%r transcript=%.60r",
                phrase, transcript,
            )
            return True
        # Prefix match: phrase must be followed by whitespace or punctuation,
        # never by another letter (prevents "no" matching "nothing").
        if normalised.startswith(phrase):
            next_char_idx = len(phrase)
            if next_char_idx >= len(normalised):
                # transcript is exactly the phrase (already caught above,
                # but guard here for safety)
                return True
            if normalised[next_char_idx] in " ,.:;!?-–":
                log.debug(
                    "event=repair_phrase_detected match=%r transcript=%.60r",
                    phrase, transcript,
                )
                return True

    return False


# ---------------------------------------------------------------------------
# Conversational intelligence — turn finalization heuristics
# ---------------------------------------------------------------------------
class SpeechIntentAnalyzer:
    """Decide whether a user turn should finalize based on transcript,
    pause duration, confidence, and adaptive tempo stats.

    All methods are synchronous and non-blocking.
    """

    def __init__(self):
        self._pause_history: list[float] = []   # recent pause_ms values
        self._wps_history: list[float] = []     # recent words-per-second values

    # -- rolling stat helpers ------------------------------------------------

    def record_pause(self, pause_ms: float):
        self._pause_history.append(pause_ms)
        if len(self._pause_history) > ROLLING_WINDOW:
            self._pause_history.pop(0)

    def record_turn(self, word_count: int, duration_sec: float):
        """Call when a turn is finalized so tempo adaptation can update."""
        if duration_sec > 0 and word_count > 0:
            wps = word_count / duration_sec
            self._wps_history.append(wps)
            if len(self._wps_history) > ROLLING_WINDOW:
                self._wps_history.pop(0)

    @property
    def dynamic_pause_threshold(self) -> float:
        """Adaptive pause threshold in ms, clamped to [500, 1200]."""
        if not self._pause_history:
            return float(MIN_PAUSE_THRESHOLD_MS)
        avg = sum(self._pause_history) / len(self._pause_history)
        return max(MIN_PAUSE_THRESHOLD_MS,
                   min(MAX_PAUSE_THRESHOLD_MS, avg * 1.5))

    # -- main decision -------------------------------------------------------

    def should_finalize(
        self,
        transcript: str,
        pause_ms: float,
        confidence: float = 1.0,
    ) -> bool:
        """Return True if the turn should commit, False to keep listening.

        Rules are evaluated in strict priority order.
        """
        text = transcript.strip()
        lower = text.lower()
        words = text.split()
        word_count = len(words)
        threshold = self.dynamic_pause_threshold

        # Rule 1 — continuation word at end → hold
        if words:
            last = lower.split()[-1].rstrip(".,!?;:")
            if last in CONTINUATION_WORDS:
                log.debug(
                    "event=analyzer decision=hold rule=continuation "
                    "word=%s pause=%d threshold=%d words=%d confidence=%.2f",
                    last, pause_ms, threshold, word_count, confidence,
                )
                return False

        # Rule 2 — strong punctuation at end → finalize
        if text and text[-1] in ".!?":
            log.debug(
                "event=analyzer decision=finalize rule=punctuation "
                "pause=%d threshold=%d words=%d confidence=%.2f",
                pause_ms, threshold, word_count, confidence,
            )
            return True

        # Rule 3 — implicit question (auxiliary verb start + pause)
        if words:
            first = lower.split()[0]
            if first in AUXILIARY_VERBS and pause_ms > IMPLICIT_QUESTION_PAUSE_MS:
                log.debug(
                    "event=analyzer decision=finalize rule=implicit_question "
                    "verb=%s pause=%d threshold=%d words=%d confidence=%.2f",
                    first, pause_ms, threshold, word_count, confidence,
                )
                return True

        # Rule 4 — low confidence + short pause → hold
        if confidence < CONFIDENCE_HOLD_THRESHOLD and pause_ms < threshold:
            log.debug(
                "event=analyzer decision=hold rule=low_confidence "
                "pause=%d threshold=%d words=%d confidence=%.2f",
                pause_ms, threshold, word_count, confidence,
            )
            return False

        # Rule 5 — pause exceeds dynamic threshold → finalize
        if pause_ms > threshold:
            log.debug(
                "event=analyzer decision=finalize rule=pause_threshold "
                "pause=%d threshold=%d words=%d confidence=%.2f",
                pause_ms, threshold, word_count, confidence,
            )
            return True

        # Rule 6 — length-based fallback
        if word_count > LENGTH_FALLBACK_WORDS and pause_ms > LENGTH_FALLBACK_PAUSE_MS:
            log.debug(
                "event=analyzer decision=finalize rule=length_fallback "
                "pause=%d threshold=%d words=%d confidence=%.2f",
                pause_ms, threshold, word_count, confidence,
            )
            return True

        # Default — hold
        log.debug(
            "event=analyzer decision=hold rule=default "
            "pause=%d threshold=%d words=%d confidence=%.2f",
            pause_ms, threshold, word_count, confidence,
        )
        return False


audio_queue = queue.Queue()

# Holds the active VoiceSession once main() creates it.
# Written once from the asyncio thread; read from the sounddevice audio thread.
# A plain attribute read on a Python object is GIL-safe — no lock needed.
_active_session: "VoiceSession | None" = None


def audio_callback(indata, frames, time_info, status):
    if status:
        log.warning("mic status=%s", status)
    sess = _active_session
    if sess is not None and sess.state not in (TurnState.IDLE, TurnState.LISTENING):
        log.debug("event=mic_frame_dropped state=%s", sess.state.value)
        return
    audio_queue.put(indata.copy())


FLUX_URL = (
    "wss://api.deepgram.com/v2/listen?"
    "model=flux-general-en"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&eager_eot_threshold=0.7"
    "&eot_threshold=0.7"
    "&eot_timeout_ms=5000"
)

groq_client = Groq(api_key=GROQ_API_KEY)

log.info("event=tts_prewarm status=starting")
try:
    dummy_response = groq_client.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="troy",
        input="Hello.",
        response_format="wav"
    )
    log.info("event=tts_prewarm status=complete")
except Exception as e:
    log.error("event=tts_prewarm status=failed error=%s", e)

class StreamingPlayback:
    """Thread-safe in-memory audio output via sd.OutputStream.

    Accepts numpy float32 chunks from any thread.  The sounddevice
    callback drains them on the audio thread — no file I/O, no GIL
    contention beyond the lock-protected deque.
    """

    SAMPLERATE = 24000  # Orpheus default
    CHANNELS = 1

    def __init__(self):
        self._buf: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._active = False
        self._first_write_time: float | None = None

    # -- called from async context (via run_in_executor) --

    def open(self):
        if self._stream is not None:
            return
        self._first_write_time = None
        self._stream = sd.OutputStream(
            samplerate=self.SAMPLERATE,
            channels=self.CHANNELS,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
            finished_callback=self._on_finished,
        )
        self._active = True
        self._stream.start()

    def write(self, samples: np.ndarray):
        """Enqueue float32 mono samples for playback."""
        if not self._active:
            return
        if self._first_write_time is None:
            self._first_write_time = time.perf_counter()
        with self._lock:
            self._buf.append(samples)

    def stop(self):
        self._active = False
        with self._lock:
            self._buf.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def first_write_time(self) -> float | None:
        return self._first_write_time

    # -- sounddevice audio-thread callback --

    def _callback(self, outdata: np.ndarray, frames: int, _time, status):
        if status:
            log.warning("event=playback_status status=%s", status)
        needed = frames
        pos = 0
        with self._lock:
            while needed > 0 and self._buf:
                chunk = self._buf[0]
                take = min(needed, len(chunk))
                outdata[pos:pos + take, 0] = chunk[:take]
                pos += take
                needed -= take
                if take < len(chunk):
                    self._buf[0] = chunk[take:]
                else:
                    self._buf.pop(0)
        if needed > 0:
            outdata[pos:, 0] = 0.0

    def _on_finished(self):
        self._active = False


class VoiceSession:
    def __init__(self):
        self.history = [
            {"role": "system", "content": "You are a multilingual real-time conversational AI voice assistant. Rules: - Always reply in the SAME language the user speaks. - If the user requests Urdu, respond ONLY in Urdu. - If the user speaks Urdu (Roman or script), respond fully in Urdu. - Do not mix English unless the user mixes. - Keep responses concise and natural."}
        ]
        self.MAX_TURNS = 10
        self.turn_start_time = None
        self.eager_end_time = None
        self.first_token_time = None
        self.llm_end_time = None
        self.tts_start_time = None
        self.tts_end_time = None
        self.playback_start_time = None
        self.playback_guard_until = 0

        # State machine
        self.state = TurnState.IDLE
        self.llm_task: asyncio.Task | None = None
        self.tts_task: asyncio.Task | None = None

        # Streaming TTS pipeline
        self._playback = StreamingPlayback()
        self.sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=3)

        # Conversational intelligence
        self.analyzer = SpeechIntentAnalyzer()
        self._grace_task:         asyncio.Task | None = None
        self._held_task:          asyncio.Task | None = None  # fallback for analyzer-held turns
        self._classifier_task:    asyncio.Task | None = None  # micro-classifier for turn label
        self._analyzer_hold_task: asyncio.Task | None = None  # safety ceiling for analyzer holds
        self._analyzer_task:      asyncio.Task | None = None  # live while analyzer is holding the turn

        # Low-signal repeat tracking (for is_meaningful_turn rule 3)
        self._last_turn_norm: str = ""       # normalised text of last accepted turn
        self._consecutive_repeats: int = 0   # how many times it has repeated

        # Health indicator
        self.last_activity_ts: float = time.monotonic()

        # Per-turn diagnostic timing (Steps 3–14)
        self._turn_id: int = 0
        self._turn_start_ms: float | None = None
        self._last_event_ms: float | None = None

    # -- health & observability ----------------------------------------------

    def touch(self):
        """Mark session as active — called from hot paths."""
        self.last_activity_ts = time.monotonic()

    def is_healthy(self) -> bool:
        return (time.monotonic() - self.last_activity_ts) < HEALTH_STALE_THRESHOLD

    def audit_resources(self):
        """Log a snapshot of every cancellable resource."""
        log.info(
            "event=shutdown_audit llm_alive=%s tts_alive=%s grace_alive=%s "
            "queue_size=%d state=%s",
            self.llm_task is not None and not self.llm_task.done()
                if self.llm_task else False,
            self.tts_task is not None and not self.tts_task.done()
                if self.tts_task else False,
            self._grace_task is not None and not self._grace_task.done()
                if self._grace_task else False,
            self.sentence_queue.qsize(),
            self.state.value,
        )

    # -- token budget trimming -----------------------------------------------

    def trim_history_if_needed(self):
        """Drop oldest user+assistant pair if estimated tokens exceed budget."""
        def _estimate_tokens() -> int:
            return sum(
                int(len(m["content"].split()) * 1.3) for m in self.history
            )
        while _estimate_tokens() > MAX_HISTORY_TOKENS and len(self.history) > 3:
            # history[0] is system; remove earliest user+assistant pair
            del self.history[1:3]
        est = _estimate_tokens()
        if est > MAX_HISTORY_TOKENS:
            return  # nothing more to trim (only system msg left)
        log.info("event=history_trim new_token_estimate=%d", est)

    def print_latency_report(self):
        if not all([self.turn_start_time, self.eager_end_time, self.first_token_time, self.llm_end_time, self.tts_start_time, self.tts_end_time, self.playback_start_time]):
            return
        stt_ms = (self.eager_end_time - self.turn_start_time) * 1000
        first_tok_ms = (self.first_token_time - self.eager_end_time) * 1000
        llm_ms = (self.llm_end_time - self.eager_end_time) * 1000
        tts_ms = (self.tts_end_time - self.tts_start_time) * 1000
        audible_ms = (self.playback_start_time - self.turn_start_time) * 1000
        log.info(
            "event=latency_report stt_ms=%.1f first_token_ms=%.1f "
            "llm_total_ms=%.1f tts_ms=%.1f time_to_audio_ms=%.1f",
            stt_ms, first_tok_ms, llm_ms, tts_ms, audible_ms,
        )

    def reset_timers(self):
        self.turn_start_time = None
        self.eager_end_time = None
        self.first_token_time = None
        self.llm_end_time = None
        self.tts_start_time = None
        self.tts_end_time = None
        self.playback_start_time = None

    def _set_state(self, new_state: TurnState):
        prev = self.state
        self.state = new_state
        self.touch()
        log.info(
            "event=state_change turn_id=%d from=%s to=%s elapsed_since_turn_start_ms=%.1f",
            self._turn_id,
            prev.value,
            new_state.value,
            (now_ms() - self._turn_start_ms) if self._turn_start_ms is not None else 0.0,
        )

    async def cancel_llm(self):
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()
            try:
                await self.llm_task
            except asyncio.CancelledError:
                pass
        self.llm_task = None

    async def cancel_tts(self):
        # Drain any pending sentences so the consumer isn't blocked
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()
            try:
                await self.tts_task
            except asyncio.CancelledError:
                pass
        self.tts_task = None
        self.stop_playback()

    def cancel_grace_timer(self):
        """Cancel any pending soft-commit grace window."""
        if self._grace_task and not self._grace_task.done():
            self._grace_task.cancel()
            log.debug("event=grace_task_cancel")
        self._grace_task = None

    def cancel_held_timer(self):
        """Cancel any pending analyzer-hold fallback timer.

        Called from transition_to_listening() and transition_to_interrupted()
        so that a legitimate speech-resume always wins over the fallback commit.
        Safe to call when no timer is pending.
        """
        if self._held_task and not self._held_task.done():
            self._held_task.cancel()
            log.debug("event=held_timer_cancelled")
        self._held_task = None

    def cancel_classifier(self) -> None:
        """Cancel any in-flight micro-classifier task.

        Called whenever the turn is superseded (new StartOfTurn, repair,
        TurnResumed) so that a stale classifier result can never arm a
        grace timer against the wrong transcript.  Safe to call when no
        classifier is running.
        """
        if self._classifier_task and not self._classifier_task.done():
            self._classifier_task.cancel()
            log.debug("event=classifier_task_cancel")
        self._classifier_task = None

    def cancel_analyzer_hold(self) -> None:
        """Cancel any in-flight analyzer-hold safety timer.

        Called from every path that either commits the turn or abandons it,
        so the 1.2 s ceiling never fires against a turn that is already done.
        Safe to call when no timer is running.
        """
        if self._analyzer_hold_task and not self._analyzer_hold_task.done():
            self._analyzer_hold_task.cancel()
            log.debug("event=analyzer_hold_cancel")
        self._analyzer_hold_task = None

    async def transition_to_listening(self):
        """StartOfTurn: cancel everything, begin listening."""
        self.cancel_held_timer()      # resume arrived — abandon any pending fallback
        self.cancel_grace_timer()
        self.cancel_classifier()      # stale label must not arm a grace timer
        self.cancel_analyzer_hold()   # turn is being superseded; ceiling no longer needed
        await self.cancel_llm()
        await self.cancel_tts()
        self._set_state(TurnState.LISTENING)

    async def transition_to_thinking(self, transcript: str):
        """EndOfTurn: launch LLM (exactly one task).

        Low-signal turns are silently discarded here — before any history
        mutation, LLM spawn, or TTS spawn — so they can never pollute the
        conversation context or trigger repetition loops.
        """
        # --- Low-signal filter (must come first) ---
        norm = transcript.strip().lower().rstrip(".,!?;:")
        if norm == self._last_turn_norm:
            self._consecutive_repeats += 1
        else:
            self._consecutive_repeats = 0
            self._last_turn_norm = norm

        if not is_meaningful_turn(transcript, self._consecutive_repeats):
            # Stay in LISTENING so the next genuine turn is handled normally.
            self._set_state(TurnState.LISTENING)
            return

        # --- Existing LLM guard ---
        if self.llm_task and not self.llm_task.done():
            log.warning("event=state_guard msg=llm_already_running")
            return

        # Feed tempo stats so the analyzer adapts to this speaker
        if self.turn_start_time is not None:
            duration = time.perf_counter() - self.turn_start_time
            word_count = len(transcript.split())
            self.analyzer.record_turn(word_count, duration)

        self._set_state(TurnState.THINKING)
        self.cancel_analyzer_hold()   # turn is committed; ceiling no longer needed
        _history_tokens_est = sum(
            len(m["content"].split()) * 4 // 3 for m in self.history
        )
        log.info(
            "event=turn_commit transcript_len=%d history_tokens_est=%d",
            len(transcript), _history_tokens_est,
        )
        self.add_user_turn(transcript)
        self.trim_history_if_needed()
        # Fresh queue for this turn
        self.sentence_queue = asyncio.Queue(maxsize=3)
        self.llm_task = asyncio.create_task(stream_groq(transcript, self))
        self.tts_task = asyncio.create_task(self._stream_tts_pipeline())

    async def force_commit_thinking(self, transcript: str):
        """Commit a turn directly, bypassing is_meaningful_turn.

        Used exclusively by the held-fallback path.  When the 800 ms
        fallback fires it means the speaker has definitively stopped and
        no TurnResumed arrived — content-quality filtering must not
        override that intent signal.

        The repeat counter is still updated so consecutive-repeat
        suppression stays accurate for subsequent normal turns.
        The LLM guard, tempo tracking, history write, and task lifecycle
        are identical to transition_to_thinking().
        """
        # --- Analyzer-alive guard ---
        # If the 1.2 s ceiling task is still running, the analyzer hold period
        # has not yet expired.  The 800 ms fallback must not race it to a
        # double commit — yield and let the ceiling resolve first.
        if self._analyzer_task and not self._analyzer_task.done():
            log.info("event=fallback_skipped reason=analyzer_alive turn_id=%d", self._turn_id)
            return

        # Keep repeat counter current even on forced commits
        norm = transcript.strip().lower().rstrip(".,!?;:")
        if norm == self._last_turn_norm:
            self._consecutive_repeats += 1
        else:
            self._consecutive_repeats = 0
            self._last_turn_norm = norm

        # LLM guard — identical to transition_to_thinking
        if self.llm_task and not self.llm_task.done():
            log.warning("event=state_guard msg=llm_already_running scope=force_commit")
            return

        if self.turn_start_time is not None:
            duration = time.perf_counter() - self.turn_start_time
            word_count = len(transcript.split())
            self.analyzer.record_turn(word_count, duration)

        self._set_state(TurnState.THINKING)
        self.cancel_analyzer_hold()   # ceiling task must not fire after we commit
        self.add_user_turn(transcript)
        self.trim_history_if_needed()
        self.sentence_queue = asyncio.Queue(maxsize=3)
        self.llm_task = asyncio.create_task(stream_groq(transcript, self))
        self.tts_task = asyncio.create_task(self._stream_tts_pipeline())

    async def _stream_tts_pipeline(self):
        """Consume sentences from the queue, TTS each, stream audio to playback."""
        loop = asyncio.get_running_loop()
        chunk_index = 0

        try:
            self._playback.open()

            while True:
                sentence = await self.sentence_queue.get()
                if sentence is None:
                    break  # LLM finished

                chunk_index += 1
                if chunk_index == 1:
                    self._set_state(TurnState.SPEAKING)
                    self.tts_start_time = time.perf_counter()
                    _tts_start_ms = now_ms()
                    log.info("event=tts_start turn_id=%d", self._turn_id)
                    log.info("event=tts_stream_start")
                    self.touch()

                    # Drain mic chunks that accumulated during THINKING.
                    # These predate TTS playback and would reach Deepgram before
                    # the sender-loop echo gate activates, causing a false
                    # StartOfTurn on the very first frame of speech output.
                    drained = 0
                    while not audio_queue.empty():
                        try:
                            audio_queue.get_nowait()
                            drained += 1
                        except queue.Empty:
                            break
                    if drained:
                        log.debug("event=echo_suppress_drain chunks=%d", drained)

                log.debug("event=tts_chunk chunk=%d text=%.60s", chunk_index, sentence)

                if SIMULATE_TTS_TIMEOUT and chunk_index == 1:
                    log.warning("event=simulation type=tts_timeout triggered")
                    await asyncio.sleep(TTS_SENTENCE_TIMEOUT + 1.0)

                # Blocking Groq call offloaded to thread, guarded by timeout
                try:
                    wav_bytes = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda s=sentence: groq_client.audio.speech.create(
                                model="canopylabs/orpheus-v1-english",
                                voice="troy",
                                input=s[:200],
                                response_format="wav",
                            ).read(),
                        ),
                        timeout=TTS_SENTENCE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    log.warning("event=timeout scope=tts_chunk chunk=%d limit=%.1fs", chunk_index, TTS_SENTENCE_TIMEOUT)
                    continue

                samples = self._decode_wav_bytes(wav_bytes)
                self._playback.write(samples)

                # Record first-audio latency once
                if self.playback_start_time is None and self._playback.first_write_time is not None:
                    self.playback_start_time = self._playback.first_write_time
                    self.playback_guard_until = self.playback_start_time + 0.8
                    log.info("event=playback_start")
                    log.info(
                        "event=tts_playback_start turn_id=%d latency_ms=%.1f",
                        self._turn_id, now_ms() - _tts_start_ms,
                    )
                    self.tts_end_time = time.perf_counter()
                    self.print_latency_report()

            # All sentences processed
            log.info("event=tts_stream_end chunks=%d", chunk_index)
            log.info(
                "event=tts_complete turn_id=%d duration_ms=%.1f",
                self._turn_id, now_ms() - _tts_start_ms,
            )
            self.tts_end_time = time.perf_counter()
            self.reset_timers()
            self._set_state(TurnState.IDLE)

        except asyncio.CancelledError:
            self.stop_playback()
            self._set_state(TurnState.INTERRUPTED)
        except Exception as e:
            log.error("event=tts_pipeline_error error=%s", e, exc_info=True)
            self.stop_playback()
            self._set_state(TurnState.IDLE)

    async def transition_to_interrupted(self):
        """TurnResumed: cancel LLM + TTS, latch into INTERRUPTED.

        Idempotent: if already INTERRUPTED the method returns immediately
        without re-cancelling tasks or emitting a redundant state-change log.
        """
        if self.state == TurnState.INTERRUPTED:
            log.debug("event=transition_skipped reason=already_interrupted")
            return
        self.cancel_held_timer()      # resume arrived — abandon any pending fallback
        self.cancel_grace_timer()
        self.cancel_classifier()      # stale label must not arm a grace timer
        self.cancel_analyzer_hold()   # speech resumed; ceiling no longer needed
        await self.cancel_llm()
        await self.cancel_tts()
        self._set_state(TurnState.INTERRUPTED)

    def add_user_turn(self, transcript_text):
        if not transcript_text:
            return
        self.history.append({
            "role": "user",
            "content": transcript_text
        })
        self._trim_history()

    def add_assistant_turn(self, full_response):
        if not full_response:
            return
        self.history.append({
            "role": "assistant",
            "content": full_response
        })
        self._trim_history()

    def _trim_history(self):
        if len(self.history) > self.MAX_TURNS * 2 + 1:
            self.history = [self.history[0]] + self.history[-self.MAX_TURNS*2:]

    def stop_playback(self):
        """Immediately silence the streaming output."""
        if self._playback.is_active:
            self._playback.stop()
            log.info("event=playback_interrupted")

    def _decode_wav_bytes(self, wav_bytes: bytes) -> np.ndarray:
        """Decode WAV bytes to float32 numpy array in-memory."""
        buf = io.BytesIO(wav_bytes)
        data, _sr = sf.read(buf, dtype="float32")
        if data.ndim == 2:
            data = data[:, 0]  # mono
        return data

async def _enqueue_sentence(q: asyncio.Queue, sentence: str):
    """Push *sentence* to the bounded queue without blocking.

    If the queue is full (maxsize=3), all stale entries are drained and
    only the freshest sentence is kept.  This guarantees the TTS consumer
    always speaks the most recent text when it falls behind the LLM.
    The None sentinel is never sent through this helper — it uses
    ``q.put(None)`` directly so it is never discarded.
    """
    try:
        q.put_nowait(sentence)
    except asyncio.QueueFull:
        # Drain stale sentences
        dropped = 0
        while not q.empty():
            try:
                q.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        log.debug("event=queue_overflow dropped=%d", dropped)
        q.put_nowait(sentence)


def _split_pending(pending: str) -> tuple[str | None, str]:
    """Try to extract a dispatchable sentence from *pending*.

    Returns (sentence, remaining) on success, (None, pending) otherwise.

    Dispatch triggers (checked in priority order):
      1. Regex sentence boundary  [.!?] followed by whitespace/EOL
      2. Long clause:  len > 60  AND  contains a comma-space
      3. (Idle timeout is handled by the caller, not here)
    """
    # --- Rule 1: hard sentence boundary ---
    m = SENTENCE_END_RE.search(pending)
    if m is not None:
        boundary = m.end()
        sentence = pending[:boundary].strip()
        remaining = pending[boundary:]
        if len(sentence) >= MIN_CHUNK_CHARS:
            return sentence, remaining
        # Boundary found but fragment is tiny — keep accumulating
        return None, pending

    # --- Rule 2: long clause with comma ---
    if len(pending) > LONG_CLAUSE_CHARS:
        # Split at the *last* comma-space so the chunk is as large as possible
        # while still leaving a clean start for the next chunk.
        parts = list(COMMA_SPLIT_RE.finditer(pending))
        if parts:
            split_at = parts[-1].end()
            sentence = pending[:split_at].strip()
            remaining = pending[split_at:]
            if len(sentence) >= MIN_CHUNK_CHARS:
                return sentence, remaining

    return None, pending


# ---------------------------------------------------------------------------
# Micro turn classifier
# ---------------------------------------------------------------------------

# Valid label set returned by the classifier prompt.
_CLASSIFIER_LABELS = frozenset({"COMPLETE", "CONTINUATION", "REPAIR", "HESITATION"})

_CLASSIFIER_PROMPT = """\
Classify the following voice utterance into exactly one of these labels:

  COMPLETE     – the speaker has finished their thought
  CONTINUATION – the speaker is mid-sentence and will continue
  REPAIR       – the speaker is correcting or retracting (e.g. "wait", "no", "actually")
  HESITATION   – the speaker paused with a filler but has not finished

Utterance: {transcript}

Reply with exactly one word from the list above. No punctuation, no explanation."""


async def classify_turn_async(
    transcript: str,
    fallback_delay: float,
    session: "VoiceSession",
) -> None:
    """Classify *transcript* and arm the grace timer based on the result.

    This coroutine IS the grace-timer orchestrator for a passing EndOfTurn.
    It replaces the direct _grace_commit spawn so that the classifier label
    can influence the commit delay before the timer is armed.

    Flow
    ────
    1. Attempt a single non-streaming Groq call (run_in_executor → non-blocking).
    2. Race it against CLASSIFIER_TIMEOUT_SEC via asyncio.wait_for.
    3. Normalise the response to one of: COMPLETE / CONTINUATION / REPAIR /
       HESITATION.  Any unexpected value falls through to COMPLETE.
    4. Apply label logic:
         COMPLETE     → arm _grace_commit at fallback_delay (heuristic default)
         CONTINUATION → arm _grace_commit at CLASSIFIER_GRACE_CONTINUATION
         REPAIR       → call transition_to_listening() — no grace timer
         HESITATION   → arm _grace_commit at CLASSIFIER_GRACE_HESITATION
       Timeout / exception → arm _grace_commit at fallback_delay (silent fallback)

    Concurrency
    ───────────
    • Stored in session._classifier_task so callers can cancel it on
      StartOfTurn / TurnResumed before the label resolves and arms a timer.
    • Cancellation at any await point is safe: CancelledError propagates out
      and nothing is armed.
    • The session state guard inside _grace_commit ensures that even if the
      classifier resolves after a state change, the commit is a no-op.

    Args:
        transcript:     Transcript to classify (snapshot at EndOfTurn time).
        fallback_delay: Grace delay from compute_adaptive_grace() — used when
                        the classifier times out or returns COMPLETE.
        session:        Active VoiceSession; used for state checks and
                        transition calls.
    """
    loop = asyncio.get_running_loop()
    label = "COMPLETE"  # safe default before any I/O

    # Entry log — fired once per EndOfTurn that passes the analyzer gate
    _word_count = len(transcript.split())
    start_ts = time.monotonic()
    log.debug(
        "event=classifier_start words=%d transcript_len=%d",
        _word_count, len(transcript),
    )

    # --- Attempt classifier call ---
    def _call_classifier() -> str:
        """Blocking Groq call — runs in thread pool."""
        resp = groq_client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _CLASSIFIER_PROMPT.format(
                        transcript=transcript.strip()
                    ),
                }
            ],
            max_tokens=4,      # label is at most one word
            temperature=0.0,   # deterministic
            stream=False,
        )
        raw = resp.choices[0].message.content or ""
        return raw.strip().upper()

    try:
        raw_label = await asyncio.wait_for(
            loop.run_in_executor(None, _call_classifier),
            timeout=CLASSIFIER_TIMEOUT_SEC,
        )
        elapsed_ms = (time.monotonic() - start_ts) * 1000

        if raw_label in _CLASSIFIER_LABELS:
            label = raw_label
            log.info(
                "event=classifier_result label=%s resolve_ms=%.1f fallback_delay_ms=%d",
                label, elapsed_ms, int(fallback_delay * 1000),
            )
        else:
            # Unexpected token — treat as COMPLETE, log for visibility
            log.warning(
                "event=classifier_unexpected raw=%r resolve_ms=%.1f fallback_delay_ms=%d",
                raw_label, elapsed_ms, int(fallback_delay * 1000),
            )
            label = "COMPLETE"

    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic() - start_ts) * 1000
        log.info(
            "event=classifier_timeout resolve_ms=%.1f fallback_delay_ms=%d",
            elapsed_ms, int(fallback_delay * 1000),
        )
        label = "COMPLETE"  # fall through to heuristic delay

    except asyncio.CancelledError:
        # Caller cancelled us (new StartOfTurn / TurnResumed) — exit cleanly.
        log.debug("event=classifier_cancelled")
        return

    except Exception as exc:
        log.warning("event=classifier_error error=%s fallback_used=True", exc)
        label = "COMPLETE"

    # --- REPAIR: no grace timer, immediate reset ---
    if label == "REPAIR":
        log.info(
            "event=classifier_repair_reset transcript=%.60r", transcript
        )
        if session.state == TurnState.LISTENING:
            await session.transition_to_listening()
        return

    # --- Determine grace delay from label ---
    if label == "CONTINUATION":
        grace_delay = CLASSIFIER_GRACE_CONTINUATION
    elif label == "HESITATION":
        grace_delay = CLASSIFIER_GRACE_HESITATION
    else:
        # COMPLETE — use heuristic (already computed by caller)
        grace_delay = fallback_delay

    # --- Arm the grace commit ---
    # cancel_grace_timer() already called by the EndOfTurn block before we
    # were spawned, but call it again here in case a concurrent path raced us.
    session.cancel_grace_timer()

    log.info(
        "event=grace_armed source=classifier label=%s delay_ms=%d",
        label, int(grace_delay * 1000),
    )
    log.info(
        "event=grace_timer_start turn_id=%d delay_ms=%d",
        session._turn_id, int(grace_delay * 1000),
    )

    _ts = transcript  # closed over by _grace_commit below

    async def _grace_commit(ts=_ts, delay=grace_delay):
        try:
            await asyncio.sleep(delay)
            if session.state != TurnState.LISTENING:
                return

            log.info(
                "event=grace_timer_fired turn_id=%d",
                session._turn_id,
            )

            # --- Semantic gate ---
            # Invoke only when the turn is ambiguous: short, low word-count,
            # or syntactically incomplete.  Long / clearly-complete turns skip
            # the validator entirely to avoid adding latency.
            _words = len(ts.split())
            _synth_complete = is_syntactically_complete(ts)
            _needs_gate = (
                len(ts) < SEMANTIC_GATE_MAX_CHARS
                or not _synth_complete
                or _words <= SEMANTIC_GATE_MAX_WORDS
            )

            if _needs_gate:
                log.info(
                    "event=semantic_gate_invoked turn_id=%d words=%d "
                    "chars=%d syntactic_complete=%s",
                    session._turn_id, _words, len(ts), _synth_complete,
                )
                sem_label = await validate_turn_semantics(ts)
                log.info(
                    "event=semantic_gate_label label=%s turn_id=%d",
                    sem_label, session._turn_id,
                )

                if sem_label == "INCOMPLETE":
                    log.info(
                        "event=semantic_gate_blocked_commit reason=INCOMPLETE "
                        "turn_id=%d extend_ms=%d",
                        session._turn_id,
                        int(SEMANTIC_GATE_EXTEND_INCOMPLETE_SEC * 1000),
                    )
                    # Re-arm listening: cancel the current grace slot and
                    # schedule a fresh hold window before retrying commit.
                    session.cancel_grace_timer()
                    if session.state == TurnState.LISTENING:
                        await asyncio.sleep(SEMANTIC_GATE_EXTEND_INCOMPLETE_SEC)
                        if session.state == TurnState.LISTENING:
                            await session.transition_to_thinking(ts)
                    return

                if sem_label == "CONTINUE_LIKELY":
                    log.info(
                        "event=semantic_gate_extended_hold reason=CONTINUE_LIKELY "
                        "turn_id=%d extend_ms=%d",
                        session._turn_id,
                        int(SEMANTIC_GATE_EXTEND_CONTINUE_SEC * 1000),
                    )
                    if session.state == TurnState.LISTENING:
                        await asyncio.sleep(SEMANTIC_GATE_EXTEND_CONTINUE_SEC)
                        if session.state == TurnState.LISTENING:
                            await session.transition_to_thinking(ts)
                    return

                # sem_label == "COMPLETE" — fall through to normal commit

            log.info(
                "event=grace_execute delay_ms=%d state=%s",
                int(delay * 1000), session.state.name,
            )
            await session.transition_to_thinking(ts)

        except asyncio.CancelledError:
            log.debug("event=grace_cancelled")

    session._grace_task = asyncio.create_task(_grace_commit())


_SENTINEL = object()  # unique marker for "iterator exhausted"


async def stream_groq(text, session):
    """Stream LLM tokens, split on sentence/clause/idle boundaries, push to sentence_queue.

    The blocking Groq iterator runs in a thread and drops tokens into
    *token_q*.  The async loop reads with a TOKEN_IDLE_TIMEOUT so it can
    flush long pauses without waiting for a punctuation boundary.
    """
    log.info("event=groq_stream_start")
    messages = session.history + [{"role": "user", "content": text}]
    tokens: list[str] = []   # accumulated without O(n) string concat
    full_response = ""
    pending = ""

    loop = asyncio.get_running_loop()
    token_q: asyncio.Queue[str | object] = asyncio.Queue()

    # --- background thread: pull chunks, push tokens ---
    def _iter_tokens():
        try:
            stream = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    loop.call_soon_threadsafe(
                        token_q.put_nowait,
                        chunk.choices[0].delta.content,
                    )
        except Exception as exc:
            loop.call_soon_threadsafe(token_q.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(token_q.put_nowait, _SENTINEL)

    try:
        loop.run_in_executor(None, _iter_tokens)
        log.info("event=llm_start model=llama-3.1-8b-instant")

        if SIMULATE_LLM_TIMEOUT:
            log.warning("event=simulation type=llm_timeout triggered")
            await asyncio.sleep(LLM_TOTAL_TIMEOUT + 1.0)

        got_first_token = False
        llm_start = time.perf_counter()
        llm_start_ms = now_ms()

        while True:
            # --- Determine effective timeout for this iteration ---
            elapsed = time.perf_counter() - llm_start

            if not got_first_token:
                # Hard ceiling: first token must arrive within 3 s
                remaining_first = LLM_FIRST_TOKEN_TIMEOUT - elapsed
                if remaining_first <= 0:
                    log.warning("event=timeout scope=llm_first_token limit=%.1fs", LLM_FIRST_TOKEN_TIMEOUT)
                    await session.sentence_queue.put(None)
                    session._set_state(TurnState.IDLE)
                    return ""
                tick_timeout = min(TOKEN_IDLE_TIMEOUT, remaining_first)
            else:
                # Hard ceiling: entire LLM stream must finish within 10 s
                remaining_total = LLM_TOTAL_TIMEOUT - elapsed
                if remaining_total <= 0:
                    log.warning("event=timeout scope=llm_total limit=%.1fs", LLM_TOTAL_TIMEOUT)
                    break  # fall through to flush + sentinel
                tick_timeout = min(TOKEN_IDLE_TIMEOUT, remaining_total)

            # --- Wait for next token OR idle/deadline timeout ---
            try:
                item = await asyncio.wait_for(
                    token_q.get(), timeout=tick_timeout
                )
            except asyncio.TimeoutError:
                # Check if a hard deadline expired
                now = time.perf_counter()
                if not got_first_token and (now - llm_start) >= LLM_FIRST_TOKEN_TIMEOUT:
                    log.warning("event=timeout scope=llm_first_token limit=%.1fs", LLM_FIRST_TOKEN_TIMEOUT)
                    await session.sentence_queue.put(None)
                    session._set_state(TurnState.IDLE)
                    return ""
                if got_first_token and (now - llm_start) >= LLM_TOTAL_TIMEOUT:
                    log.warning("event=timeout scope=llm_total limit=%.1fs", LLM_TOTAL_TIMEOUT)
                    break  # fall through to flush + sentinel

                # Otherwise it's just the idle-flush rule
                flush = pending.strip()
                if len(flush) >= MIN_CHUNK_CHARS:
                    log.debug("event=idle_flush chars=%d", len(flush))
                    await _enqueue_sentence(session.sentence_queue, flush)
                    pending = ""
                continue

            # --- Handle sentinel / errors ---
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                log.error("event=llm_error error=%s", item)
                break

            # --- Normal token ---
            token: str = item
            if not got_first_token:
                got_first_token = True
                session.first_token_time = time.perf_counter()
                session.touch()
                log.info(
                    "event=llm_first_token turn_id=%d latency_ms=%.1f",
                    session._turn_id, now_ms() - llm_start_ms,
                )
            print(token, end="", flush=True)
            tokens.append(token)   # O(1) — no string copy per token
            pending += token

            # --- Try to dispatch a chunk (rules 1 & 2) ---
            while True:
                sentence, pending = _split_pending(pending)
                if sentence is None:
                    break
                await _enqueue_sentence(session.sentence_queue, sentence)

            # Yield control after every token so the event loop can service
            # timers, WebSocket sends, and other coroutines without stalling.
            await asyncio.sleep(0)

        # --- LLM finished (or timed out): flush remainder ---
        leftover = pending.strip()
        if leftover:
            await _enqueue_sentence(session.sentence_queue, leftover)

        full_response = "".join(tokens)  # single O(n) join after all tokens received
        session.llm_end_time = time.perf_counter()
        log.info(
            "event=llm_complete turn_id=%d duration_ms=%.1f",
            session._turn_id, now_ms() - llm_start_ms,
        )
        log.info("event=groq_stream_end chars=%d", len(full_response))

        await session.sentence_queue.put(None)
        session.add_assistant_turn(full_response)
        return full_response

    except asyncio.CancelledError:
        await session.sentence_queue.put(None)
        raise
    except Exception as e:
        log.error("event=llm_error error=%s", e, exc_info=True)
        await session.sentence_queue.put(None)
        return ""

async def main():
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}"
    }

    session = VoiceSession()
    global _active_session
    _active_session = session  # expose to audio_callback for state-gated enqueue

    # Microphone — started once, survives reconnects
    mic_stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype="int16",
        blocksize=1280,
        callback=audio_callback,
    )
    mic_stream.start()
    log.info("event=mic_started")

    loop = asyncio.get_running_loop()
    background_tasks: list[asyncio.Task] = []

    # --- Periodic health check (always on) ---
    async def _health_check_loop():
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            log.debug("event=health_check healthy=%s", session.is_healthy())

    background_tasks.append(asyncio.create_task(_health_check_loop()))

    # --- Soak heartbeat (opt-in) ---
    if SOAK_TEST_MODE:
        async def _soak_heartbeat():
            while True:
                log.debug(
                    "event=soak_heartbeat state=%s queue=%d",
                    session.state.name,
                    session.sentence_queue.qsize(),
                )
                await asyncio.sleep(10)
        background_tasks.append(asyncio.create_task(_soak_heartbeat()))
        log.info("event=soak_mode_enabled")

    # --- Event loop stall monitor ---
    async def monitor_event_loop():
        last = now_ms()
        while True:
            await asyncio.sleep(0.1)
            current = now_ms()
            drift = current - last - 100.0
            if drift > 150.0:
                log.warning("event=event_loop_stall stall_ms=%.1f", drift)
            last = current

    background_tasks.append(asyncio.create_task(monitor_event_loop()))

    # --- Audio queue size monitor (every 500 ms) ---
    async def _audio_queue_monitor():
        while True:
            await asyncio.sleep(0.5)
            log.info("event=audio_queue_status size=%d", audio_queue.qsize())

    background_tasks.append(asyncio.create_task(_audio_queue_monitor()))

    # --- Mic drop simulation (opt-in) ---
    if SIMULATE_MIC_DROP:
        async def _sim_mic_drop():
            while True:
                await asyncio.sleep(random.uniform(30, 90))
                log.warning("event=simulation type=mic_drop triggered")
                try:
                    mic_stream.stop()
                    await asyncio.sleep(2.0)
                    mic_stream.start()
                    log.info("event=simulation type=mic_drop recovered")
                except Exception as exc:
                    log.error("event=simulation type=mic_drop error=%s", exc)
        background_tasks.append(asyncio.create_task(_sim_mic_drop()))

    async def _run_session(ws):
        """Run receiver + sender on a live websocket. Returns on disconnect."""
        connected = False
        current_transcript = ""

        async def receiver():
            nonlocal connected, current_transcript

            async for message in ws:
                msg = json.loads(message)
                log.info("event=ws_receive turn_id=%d", session._turn_id)

                if msg.get("speech_final") is True:
                    transcript = msg.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "").strip()
                    log.info(
                        "event=stt_finalized transcript_len=%d speech_final=%s",
                        len(transcript),
                        msg.get("speech_final")
                    )
                    if session.state == TurnState.LISTENING:
                        # Deepgram already decided — bypass analyzer/classifier
                        # Atomic cancel to prevent double-THINKING if EndOfTurn just fired
                        session.cancel_grace_timer()
                        session.cancel_classifier()
                        await session.force_commit_thinking(transcript)
                    continue  # Yield to next message loop

                if msg.get("type") == "Connected":
                    log.info("event=ws_connected")
                    connected = True
                elif msg.get("type") == "TurnInfo":
                    event = msg.get("event")
                    log.info("event=turn_info turn_event=%s transcript_len=%d",
                             event, len(msg.get("transcript", "")))
                    transcript = msg.get("transcript", "").strip()

                    if transcript:
                        current_transcript = transcript

                    # --- Zero-length Update guard ---
                    # Empty Update events carry no new transcript content and
                    # would cause downstream handlers to operate on stale text.
                    # Drop them unless the session is INTERRUPTED, where a
                    # zero-length Update can carry a meaningful resume signal.
                    if event == "Update" and not transcript:
                        if session.state != TurnState.INTERRUPTED:
                            log.debug(
                                "event=update_suppressed reason=empty_transcript state=%s",
                                session.state.value,
                            )
                            continue

                    if event == "StartOfTurn":
                        now = time.perf_counter()
                        if now < session.playback_guard_until:
                            log.debug("event=start_of_turn_suppressed reason=playback_guard")
                            continue

                        # --- Repair phrase fast-path ---
                        # Checked before the barge-in gate so that single-word
                        # repairs ("wait", "no") are never blocked by the
                        # word-count / duration thresholds.
                        #
                        # Active during SPEAKING and LISTENING:
                        #   SPEAKING  → user is correcting mid-TTS; stop
                        #               everything and reset to LISTENING.
                        #   LISTENING → grace timer may be armed; user is
                        #               retracting the utterance before it
                        #               commits; cancel timers and reset.
                        #
                        # No LLM call is made.  History is NOT mutated — the
                        # repair itself is invisible to the conversation context.
                        # The user will speak again from a clean state.
                        #
                        # Race-safety: transition_to_listening() cancels grace
                        # timer, held timer, LLM task, and TTS task atomically
                        # before setting state — identical to a normal barge-in.
                        if session.state in (TurnState.SPEAKING, TurnState.LISTENING):
                            if is_repair_phrase(current_transcript):
                                log.info(
                                    "event=repair_detected state=%s "
                                    "transcript=%.60r",
                                    session.state.value, current_transcript,
                                )
                                session.turn_start_time = now
                                current_transcript = ""
                                await session.transition_to_listening()
                                continue

                        # --- Barge-in confidence gate (SPEAKING only) ---
                        # During SPEAKING, honour the StartOfTurn only when the
                        # incoming audio is strong enough to be intentional speech.
                        # In every other state the transition proceeds unconditionally.
                        if session.state == TurnState.SPEAKING:
                            speech_duration_ms = (
                                (now - session.turn_start_time) * 1000
                                if session.turn_start_time is not None
                                else 0.0
                            )
                            if not interrupt_is_strong(current_transcript, speech_duration_ms):
                                log.info(
                                    "event=start_of_turn_suppressed "
                                    "reason=barge_in_gate state=SPEAKING "
                                    "transcript_len=%d duration_ms=%.0f",
                                    len(current_transcript), speech_duration_ms,
                                )
                                continue

                        session.turn_start_time = now
                        current_transcript = ""
                        session._turn_id += 1
                        session._turn_start_ms = now_ms()
                        session._last_event_ms = session._turn_start_ms
                        log.info(
                            "event=turn_begin turn_id=%d state=%s",
                            session._turn_id, session.state.name,
                        )
                        await session.transition_to_listening()

                    elif event == "EagerEndOfTurn":
                        _eot_now_ms = now_ms()
                        if session._last_event_ms is not None:
                            log.info(
                                "event=stt_event_timing turn_id=%d stt_event=EagerEndOfTurn delta_ms=%.1f",
                                session._turn_id, _eot_now_ms - session._last_event_ms,
                            )
                        session._last_event_ms = _eot_now_ms
                        session.eager_end_time = time.perf_counter()
                        log.info("event=eager_end_of_turn")

                    elif event == "EndOfTurn":
                        _eot_now_ms = now_ms()
                        if session._last_event_ms is not None:
                            log.info(
                                "event=stt_event_timing turn_id=%d stt_event=EndOfTurn delta_ms=%.1f",
                                session._turn_id, _eot_now_ms - session._last_event_ms,
                            )
                        session._last_event_ms = _eot_now_ms
                        if session.state not in (TurnState.LISTENING, TurnState.INTERRUPTED):
                            log.debug("event=end_of_turn_suppressed state=%s", session.state.value)
                            continue
                        log.info("event=end_of_turn")

                        # --- Conversational intelligence gate ---
                        eot_now = time.perf_counter()
                        pause_ms = ((eot_now - session.eager_end_time) * 1000
                                    if session.eager_end_time else 0.0)
                        confidence = msg.get("confidence", 1.0)
                        session.analyzer.record_pause(pause_ms)
                        session.touch()

                        if not session.analyzer.should_finalize(
                            current_transcript, pause_ms, confidence
                        ):
                            log.info("event=end_of_turn_held reason=analyzer")

                            # --- Analyzer-hold safety ceiling ---
                            # Arms a 1.2 s one-shot that force-commits the turn
                            # if the analyzer keeps holding and no speech resumes.
                            # Cancelled by any path that resolves the turn first.
                            session.cancel_analyzer_hold()
                            _hold_ceiling_snapshot = current_transcript

                            async def _analyzer_hold_timeout(ts=_hold_ceiling_snapshot):
                                try:
                                    await asyncio.sleep(ANALYZER_HOLD_TIMEOUT_SEC)
                                    session._analyzer_task = None  # Step 4b: ceiling fired — hold over
                                    if session.state in (TurnState.LISTENING,
                                                         TurnState.INTERRUPTED):
                                        log.info(
                                            "event=analyzer_hold_timer_fired turn_id=%d",
                                            session._turn_id,
                                        )
                                        log.info(
                                            "event=analyzer_hold_timeout_force_commit "
                                            "delay_ms=%d transcript_len=%d",
                                            int(ANALYZER_HOLD_TIMEOUT_SEC * 1000),
                                            len(ts),
                                        )
                                        await session.force_commit_thinking(ts)
                                except asyncio.CancelledError:
                                    session._analyzer_task = None  # Step 4c: ceiling cancelled — hold over
                                    log.info(
                                        "event=analyzer_hold_timer_cancelled turn_id=%d",
                                        session._turn_id,
                                    )
                                    log.debug("event=analyzer_hold_timeout_cancelled")

                            session._analyzer_hold_task = asyncio.create_task(
                                _analyzer_hold_timeout()
                            )
                            session._analyzer_task = session._analyzer_hold_task  # Step 2: track hold as active analyzer work
                            log.info(
                                "event=analyzer_hold_timer_start turn_id=%d timeout_ms=%d",
                                session._turn_id, int(ANALYZER_HOLD_TIMEOUT_SEC * 1000),
                            )

                            # --- Fallback timer: force-commit if no resume arrives ---
                            # The analyzer held this turn (e.g. trailing conjunction,
                            # low confidence).  If Deepgram never fires TurnResumed
                            # (rare VAD edge-case), the session would deadlock in
                            # LISTENING forever.  We arm an 800 ms one-shot task that
                            # force-finalises the turn, exactly like _grace_commit but
                            # with a longer window and a distinct log tag.
                            #
                            # Race conditions handled:
                            #   • Speech resumes via StartOfTurn   → transition_to_listening()
                            #     calls cancel_held_timer() before the sleep expires.
                            #   • TurnResumed arrives               → transition_to_interrupted()
                            #     calls cancel_held_timer() before the sleep expires.
                            #   • Multiple consecutive held EndOfTurns → cancel_held_timer()
                            #     at the top of this block resets the clock each time,
                            #     so the 800 ms is always measured from the *latest* hold.
                            #   • _grace_commit fires for a *later* passing EndOfTurn while
                            #     the held timer is still armed → _grace_commit calls
                            #     transition_to_thinking() which sets state to THINKING,
                            #     so the held timer's state-guard (`state == LISTENING`)
                            #     fails and it exits cleanly — no double commit.
                            session.cancel_held_timer()   # reset clock on each held event
                            _held_snapshot = current_transcript

                            async def _held_fallback(ts=_held_snapshot):
                                try:
                                    await asyncio.sleep(ANALYZER_HOLD_FALLBACK_SEC)
                                    # State guard: another path may have already
                                    # committed or cancelled us.
                                    if session.state == TurnState.LISTENING:
                                        # Fallback fires only when the speaker has
                                        # genuinely stopped and no TurnResumed arrived.
                                        # Skip is_meaningful_turn — the user stopped
                                        # speaking; we must honour that intent.
                                        log.info(
                                            "event=fallback_force_commit "
                                            "turn_id=%d transcript_len=%d "
                                            "time_since_turn_start_ms=%.1f",
                                            session._turn_id,
                                            len(ts),
                                            (now_ms() - session._turn_start_ms)
                                            if session._turn_start_ms is not None else 0.0,
                                        )
                                        await session.force_commit_thinking(ts)
                                except asyncio.CancelledError:
                                    log.debug("event=held_fallback_cancelled")

                            session._held_task = asyncio.create_task(_held_fallback())
                            continue

                        # Analyzer approved this turn — the hold period is over.
                        session._analyzer_task = None  # Step 4a: hold resolved via pass

                        # --- Heuristic grace delay (classifier fallback) ---
                        # compute_adaptive_grace() runs synchronously — zero
                        # latency, no I/O.  Its result is passed to the
                        # classifier as the fallback delay so that a timeout
                        # never degrades below the heuristic quality we already
                        # had before the classifier was introduced.
                        _grace_delay = compute_adaptive_grace(current_transcript)
                        if _grace_delay != GRACE_WINDOW_SEC:
                            log.debug(
                                "event=grace_heuristic "
                                "delay_ms=%.0f transcript=%.60r",
                                _grace_delay * 1000,
                                current_transcript,
                            )

                        # --- Micro-classifier orchestrator ---
                        # classify_turn_async() races a Groq label call against
                        # CLASSIFIER_TIMEOUT_SEC (100 ms), then arms _grace_task
                        # at the label-specific delay.  On timeout or error it
                        # silently falls back to _grace_delay (heuristic above).
                        #
                        # Concurrency invariants:
                        #   • cancel_classifier() called first: ensures any
                        #     previous in-flight classifier for a stale turn
                        #     cannot arm a grace timer for the wrong transcript.
                        #   • cancel_grace_timer() called inside
                        #     classify_turn_async() just before arming, making
                        #     the arm atomic relative to any residual grace task.
                        #   • StartOfTurn / TurnResumed cancel both
                        #     _classifier_task and _grace_task via
                        #     transition_to_listening() / transition_to_interrupted(),
                        #     so no stale commit can fire after a state change.
                        session.cancel_classifier()
                        session.cancel_grace_timer()
                        _classifier_snapshot = current_transcript
                        session._classifier_task = asyncio.create_task(
                            classify_turn_async(
                                _classifier_snapshot,
                                _grace_delay,
                                session,
                            )
                        )

                    elif event == "TurnResumed":
                        _tr_now_ms = now_ms()
                        if session._last_event_ms is not None:
                            log.info(
                                "event=stt_event_timing turn_id=%d stt_event=TurnResumed delta_ms=%.1f",
                                session._turn_id, _tr_now_ms - session._last_event_ms,
                            )
                        session._last_event_ms = _tr_now_ms
                        log.info("event=turn_resumed")
                        await session.transition_to_interrupted()

        receiver_task = asyncio.create_task(receiver())

        while not connected:
            await asyncio.sleep(0.01)

        # Drain any stale mic chunks that accumulated during reconnect
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        try:
            while True:
                chunk = await loop.run_in_executor(None, audio_queue.get)

                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

                if session.state == TurnState.SPEAKING:
                    # --- Echo suppression gate ---
                    # TTS is playing back through the speaker.  Replace mic audio
                    # with silence so Deepgram never sees TTS bleed-through.
                    #
                    # Exception: a chunk whose RMS clears BARGE_IN_RMS_THRESHOLD
                    # is forwarded as real audio.  This preserves the ability for
                    # a user to interrupt loudly while the engine is speaking.
                    # The barge-in confidence gate (interrupt_is_strong) in the
                    # receiver still applies on top, so two independent filters
                    # must both pass before an interrupt is honoured.
                    if rms >= BARGE_IN_RMS_THRESHOLD:
                        log.debug(
                            "event=echo_suppress_bypass rms=%.0f threshold=%d",
                            rms, BARGE_IN_RMS_THRESHOLD,
                        )
                        _send_start = now_ms()
                        await ws.send(chunk.tobytes())
                        log.info(
                            "event=ws_send duration_ms=%.1f queue_size=%d",
                            now_ms() - _send_start, audio_queue.qsize(),
                        )
                    else:
                        silent_chunk = np.zeros_like(chunk)
                        _send_start = now_ms()
                        await ws.send(silent_chunk.tobytes())
                        log.info(
                            "event=ws_send duration_ms=%.1f queue_size=%d",
                            now_ms() - _send_start, audio_queue.qsize(),
                        )
                else:
                    # --- Normal gate (all non-SPEAKING states) ---
                    # Existing behaviour: pass loud chunks, silence quiet ones.
                    if rms > 1200:
                        _send_start = now_ms()
                        await ws.send(chunk.tobytes())
                        log.info(
                            "event=ws_send duration_ms=%.1f queue_size=%d",
                            now_ms() - _send_start, audio_queue.qsize(),
                        )
                    else:
                        silent_chunk = np.zeros_like(chunk)
                        _send_start = now_ms()
                        await ws.send(silent_chunk.tobytes())
                        log.info(
                            "event=ws_send duration_ms=%.1f queue_size=%d",
                            now_ms() - _send_start, audio_queue.qsize(),
                        )

                await asyncio.sleep(0)
        finally:
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

    # --- Connection loop with automatic reconnect ---
    attempt = 0
    try:
        while True:
            try:
                async with websockets.connect(FLUX_URL, extra_headers=headers) as ws:
                    attempt = 0  # reset on successful connection

                    if SIMULATE_WIFI_DROP:
                        async def _sim_wifi_drop():
                            delay = random.uniform(60, 120)
                            await asyncio.sleep(delay)
                            log.warning("event=simulation type=wifi_drop triggered")
                            await ws.close()

                        wifi_task = asyncio.create_task(_sim_wifi_drop())
                        try:
                            await _run_session(ws)
                        finally:
                            wifi_task.cancel()
                            try:
                                await wifi_task
                            except asyncio.CancelledError:
                                pass
                    else:
                        await _run_session(ws)
                    # Clean exit from _run_session means ws closed normally
                    log.info("event=ws_closed reason=server")

            except (websockets.ConnectionClosed, websockets.InvalidStatusCode, OSError) as e:
                attempt += 1
                # Reset session state so we don't carry stale LLM/TTS tasks
                session.cancel_held_timer()
                session.cancel_grace_timer()
                session.cancel_classifier()
                session.cancel_analyzer_hold()
                await session.cancel_llm()
                await session.cancel_tts()
                session._set_state(TurnState.IDLE)

                if attempt > WS_RECONNECT_ATTEMPTS:
                    log.error("event=ws_reconnect_failed attempts=%d", WS_RECONNECT_ATTEMPTS)
                    break

                log.warning("event=ws_disconnected error=%s attempt=%d/%d delay=%.1fs",
                            e, attempt, WS_RECONNECT_ATTEMPTS, WS_RECONNECT_DELAY)
                await asyncio.sleep(WS_RECONNECT_DELAY)

    except KeyboardInterrupt:
        log.info("event=shutdown reason=keyboard_interrupt")
    finally:
        # --- Graceful resource teardown ---
        for bt in background_tasks:
            bt.cancel()
        for bt in background_tasks:
            try:
                await bt
            except asyncio.CancelledError:
                pass

        session.cancel_held_timer()
        session.cancel_grace_timer()
        session.cancel_classifier()
        session.cancel_analyzer_hold()
        await session.cancel_llm()
        await session.cancel_tts()

        # Drain sentence queue
        while not session.sentence_queue.empty():
            try:
                session.sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        session.stop_playback()
        session.audit_resources()

        try:
            mic_stream.stop()
            mic_stream.close()
        except Exception:
            pass

asyncio.run(main())
