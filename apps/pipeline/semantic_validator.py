import asyncio
import logging
import os
from typing import Literal

from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("voice_engine")

_VALID_LABELS: frozenset[str] = frozenset({"COMPLETE", "INCOMPLETE", "CONTINUE_LIKELY"})

_TERMINAL_PUNCTUATION: frozenset[str] = frozenset(".!?")

_SYSTEM_PROMPT = (
    "You are a multilingual conversational turn boundary validator.\n"
    "You will receive a user utterance in ANY language (English, Urdu, Arabic, or mixed).\n"
    "Determine whether the utterance represents a complete thought that should be answered now.\n"
    "Return ONLY ONE of these labels:\n"
    "COMPLETE\n"
    "INCOMPLETE\n"
    "CONTINUE_LIKELY\n"
    "Do NOT explain.\n"
    "Do NOT add punctuation.\n"
    "Do NOT add quotes."
)

_ROBUST_SYSTEM_PROMPT = (
    "You are an advanced multilingual conversation analyzer.\n"
    "Determine whether responding NOW would feel natural in a real human conversation.\n"
    "Consider unfinished clauses, trailing conjunctions, hesitation, and semantic incompleteness.\n"
    "Return ONLY ONE of these labels:\n"
    "COMPLETE\n"
    "INCOMPLETE\n"
    "CONTINUE_LIKELY\n"
    "Do NOT explain.\n"
    "Do NOT add punctuation.\n"
    "Do NOT add quotes."
)

_ROBUST_TIMEOUT_MS: int = 400

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    return _client


async def _call_model(transcript: str, robust: bool) -> str:
    response = await _get_client().chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": _ROBUST_SYSTEM_PROMPT if robust else _SYSTEM_PROMPT},
            {"role": "user", "content": transcript.strip()},
        ],
        temperature=0.0,
        max_tokens=4 if robust else 2,
        stream=False,
    )
    raw = (response.choices[0].message.content or "").strip().upper()
    return raw


async def _run_validation(
    transcript: str,
    timeout_ms: int,
    robust: bool,
) -> Literal["COMPLETE", "INCOMPLETE", "CONTINUE_LIKELY"]:
    try:
        raw = await asyncio.wait_for(
            _call_model(transcript, robust=robust),
            timeout=timeout_ms / 1000.0,
        )
    except asyncio.TimeoutError:
        log.warning(
            "event=semantic_validation_timeout transcript_len=%d timeout_ms=%d robust=%s",
            len(transcript), timeout_ms, robust,
        )
        return "COMPLETE"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.warning(
            "event=semantic_validation_error error=%s fallback=COMPLETE robust=%s",
            exc, robust,
        )
        return "COMPLETE"

    if raw not in _VALID_LABELS:
        log.warning(
            "event=semantic_validation_invalid_label raw=%r fallback=COMPLETE robust=%s",
            raw, robust,
        )
        return "COMPLETE"

    return raw  # type: ignore[return-value]


def _should_escalate(transcript: str, fast_label: str) -> bool:
    stripped = transcript.strip()
    return (
        fast_label == "COMPLETE"
        and len(stripped) < 20
        and (not stripped or stripped[-1] not in _TERMINAL_PUNCTUATION)
    )


async def validate_turn_semantics(
    transcript: str,
    timeout_ms: int = 250,
    robust: bool = False,
) -> Literal["COMPLETE", "INCOMPLETE", "CONTINUE_LIKELY"]:
    log.info(
        "event=semantic_validation_start transcript_len=%d robust=%s",
        len(transcript), robust,
    )

    label = await _run_validation(transcript, timeout_ms=timeout_ms, robust=robust)

    log.info(
        "event=semantic_validation_result label=%s transcript_len=%d robust=%s",
        label, len(transcript), robust,
    )

    if not robust and _should_escalate(transcript, label):
        log.info(
            "event=semantic_gate_escalation_triggered transcript_len=%d fast_label=%s",
            len(transcript), label,
        )
        robust_label = await _run_validation(
            transcript,
            timeout_ms=_ROBUST_TIMEOUT_MS,
            robust=True,
        )
        log.info(
            "event=semantic_gate_robust_result label=%s transcript_len=%d",
            robust_label, len(transcript),
        )
        return robust_label

    return label
