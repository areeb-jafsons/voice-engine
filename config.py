"""
config.py — Abdion Voice Engine · Runtime Configuration
========================================================
Pydantic models for every tunable parameter across all services.
Serialises to / deserialises from JSON.  Used by:
  • server.py  — GET/PUT /config endpoints, passes config to bot via stdin
  • bot.py     — reads config from stdin, applies to each service constructor
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

log = logging.getLogger("voice_engine.config")

# ---------------------------------------------------------------------------
# Default system prompt (kept here so config.py is the single source of truth)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
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
# Per-service config sections
# ---------------------------------------------------------------------------

class GroqConfig(BaseModel):
    """Groq LLM parameters (passed to GroqLLMService + InputParams)."""
    model: str = Field(default="llama-3.3-70b-versatile", description="Groq model ID")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Randomness (0.0–2.0)")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Nucleus sampling")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Max response tokens")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Penalize repeated tokens")
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Penalize topic repetition")
    seed: Optional[int] = Field(default=None, description="Deterministic sampling seed")


class ElevenLabsConfig(BaseModel):
    """ElevenLabs TTS parameters (passed to ElevenLabsTTSService + InputParams)."""
    voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID")
    model: str = Field(default="eleven_turbo_v2_5", description="TTS model")
    language: Optional[str] = Field(default=None, description="Force language code (e.g. 'en', 'ur')")
    stability: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Voice stability")
    similarity_boost: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Clarity + similarity")
    style: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Style exaggeration")
    use_speaker_boost: Optional[bool] = Field(default=None, description="Speaker clarity boost")
    speed: Optional[float] = Field(default=None, ge=0.25, le=4.0, description="Speaking speed (0.25–4.0)")


class DeepgramConfig(BaseModel):
    """Deepgram STT parameters (passed to LiveOptions)."""
    model: str = Field(default="nova-3", description="Deepgram model")
    language: str = Field(default="multi", description="Language / code-switching mode")
    endpointing: int = Field(default=800, ge=0, le=5000, description="Silence endpointing (ms)")
    smart_format: bool = Field(default=True, description="Auto-formatting")
    punctuate: bool = Field(default=True, description="Add punctuation")
    interim_results: bool = Field(default=True, description="Stream partial results")
    utterance_end_ms: Optional[int] = Field(default=None, ge=0, le=10000, description="Force utterance end timeout (ms)")
    filler_words: Optional[bool] = Field(default=None, description="Include filler words in transcript")
    keywords: Optional[list[str]] = Field(default=None, description="Keyword boosting list")
    diarize: Optional[bool] = Field(default=None, description="Speaker diarization")
    numerals: Optional[bool] = Field(default=None, description="Convert spoken numbers to digits")
    profanity_filter: Optional[bool] = Field(default=None, description="Filter profanity")


class TurnGateConfig(BaseModel):
    """HybridTurnGate tuning parameters."""
    semantic_extension_sec: float = Field(default=1.2, ge=0.1, le=10.0, description="Semantic wait window (seconds)")
    max_semantic_extensions: int = Field(default=2, ge=0, le=10, description="Max extensions before force-commit")


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class VoiceEngineConfig(BaseModel):
    """Complete runtime configuration for the voice engine."""
    groq: GroqConfig = Field(default_factory=GroqConfig)
    elevenlabs: ElevenLabsConfig = Field(default_factory=ElevenLabsConfig)
    deepgram: DeepgramConfig = Field(default_factory=DeepgramConfig)
    turn_gate: TurnGateConfig = Field(default_factory=TurnGateConfig)
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, description="System prompt for the LLM")

    # -- Persistence -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "VoiceEngineConfig":
        """Load config from a JSON file.  Returns defaults if file doesn't exist."""
        p = Path(path)
        if not p.exists():
            log.info("event=config_load_defaults path=%s", p)
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            config = cls.model_validate(data)
            log.info("event=config_loaded path=%s", p)
            return config
        except Exception as exc:
            log.warning("event=config_load_error path=%s error=%s — using defaults", p, exc)
            return cls()

    def save(self, path: str | Path) -> None:
        """Persist config to a JSON file (pretty-printed)."""
        p = Path(path)
        p.write_text(
            self.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
        log.info("event=config_saved path=%s", p)

    def merge_patch(self, patch: dict) -> "VoiceEngineConfig":
        """Return a new config with `patch` merged over `self`.

        Supports nested partial updates, e.g.:
            {"groq": {"temperature": 0.7}}
        only changes groq.temperature, leaving everything else intact.
        """
        base = self.model_dump()
        _deep_merge(base, patch)
        return VoiceEngineConfig.model_validate(base)


def _deep_merge(base: dict, patch: dict) -> None:
    """Recursively merge `patch` into `base` in-place."""
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
