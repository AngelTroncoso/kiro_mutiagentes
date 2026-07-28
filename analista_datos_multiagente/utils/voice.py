"""Capa 7 - Infraestructura de voz: Speech-to-Text y Text-to-Speech.

Mismo principio de degradacion segura que el resto de la app:

    STT: Groq Whisper API (usa la misma GROQ_API_KEY ya configurada).
         Si no hay clave o falla la red, se devuelve None y la UI ofrece
         escribir la pregunta en texto en vez de hablarla.

    TTS: ElevenLabs API (ELEVENLABS_API_KEY) -> fallback gTTS (sin clave,
         requiere internet) -> fallback final: sin audio, solo texto.

Todo el pipeline de voz esta fijado a ESPANOL (idioma del resto de la app):
Whisper recibe `language="es"` y ElevenLabs/gTTS usan una voz/locale en
espanol. Ningun componente de este modulo genera contenido: solo transcribe
o sintetiza texto que ya viene de otro lado (nunca inventa palabras).
"""

from __future__ import annotations

import io
import os
from typing import Any

# Idioma fijo de todo el modulo de voz (coherente con el resto de la app).
LANGUAGE = "es"
_ELEVEN_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # "Rachel", multi-idioma
_ELEVEN_MODEL = "eleven_multilingual_v2"


# ----------------------------------------------------------------------
# Speech-to-Text (transcripcion)
# ----------------------------------------------------------------------
def is_stt_available() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str | None:
    """Transcribe audio a texto en espanol usando Groq Whisper.

    Devuelve None si no hay clave, no hay libreria o falla la llamada (la UI
    debe degradar a un campo de texto manual en ese caso).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not audio_bytes:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        buffer = io.BytesIO(audio_bytes)
        buffer.name = filename
        transcription = client.audio.transcriptions.create(
            file=buffer,
            model="whisper-large-v3",
            language=LANGUAGE,
            response_format="text",
        )
        text = transcription if isinstance(transcription, str) else getattr(
            transcription, "text", None)
        return text.strip() if text else None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Text-to-Speech (sintesis de voz)
# ----------------------------------------------------------------------
def is_tts_available() -> dict[str, bool]:
    return {
        "elevenlabs": bool(os.environ.get("ELEVENLABS_API_KEY")),
        "gtts": True,  # no requiere clave, solo internet
    }


def synthesize_speech(text: str) -> tuple[bytes | None, str]:
    """Convierte texto (ya generado por un agente) en audio hablado en espanol.

    Devuelve (audio_bytes_mp3_o_None, motor_usado). `motor_usado` es
    "elevenlabs", "gtts" o "none" (sin audio disponible; la UI muestra solo
    texto, degradacion segura).
    """
    if not text or not text.strip():
        return None, "none"

    audio = _try_elevenlabs(text)
    if audio is not None:
        return audio, "elevenlabs"

    audio = _try_gtts(text)
    if audio is not None:
        return audio, "gtts"

    return None, "none"


def _try_elevenlabs(text: str) -> bytes | None:
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    try:
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=api_key)
        audio_stream = client.text_to_speech.convert(
            voice_id=_ELEVEN_VOICE_ID,
            model_id=_ELEVEN_MODEL,
            text=text,
            output_format="mp3_44100_128",
        )
        chunks = [chunk for chunk in audio_stream if isinstance(chunk, bytes)]
        audio_bytes = b"".join(chunks)
        return audio_bytes or None
    except Exception:
        return None


def _try_gtts(text: str) -> bytes | None:
    try:
        from gtts import gTTS

        buffer = io.BytesIO()
        gTTS(text=text, lang=LANGUAGE).write_to_fp(buffer)
        buffer.seek(0)
        data = buffer.read()
        return data or None
    except Exception:
        return None


def voice_status() -> dict[str, Any]:
    """Resumen de que capacidades de voz estan activas, para mostrar en la UI."""
    tts = is_tts_available()
    return {
        "stt": is_stt_available(),
        "tts_elevenlabs": tts["elevenlabs"],
        "tts_gtts": tts["gtts"],
    }
