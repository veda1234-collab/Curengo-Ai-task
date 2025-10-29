#!/usr/bin/env python3
import os
import sys
import time
import json
import tempfile
from typing import List

import numpy as np
import sounddevice as sd
import soundfile as sf

# Do not force provider; honor environment or app defaults (OpenAI with optional local base URL)

# Reuse core logic from the API
from app import (
    transcribe_audio_file,
    llm_extract,
    coerce_llm_final_json,
    refine_with_transcript,
    find_best_active_match,
    FORMS,
    FREQUENCIES,
    TIMINGS,
    ROUTES,
    LLM_SCHEMA_INSTRUCTION,
    GROQ_MODEL,
    OPENAI_MODEL,
    OPENAI_BASE_URL,
    LLM_PROVIDER,
)


SAMPLE_RATE = 16000
CHANNELS = 1
THRESHOLD = 0.006
SILENCE_SECS = 0.8
MAX_RECORD_SECS = 20.0


def record_until_silence() -> np.ndarray:
    """Record mono audio until brief silence is detected or max duration reached."""
    print("\nSpeak now... (stop talking to finish)")
    buf: List[np.ndarray] = []
    silence_secs = 0.0
    total_secs = 0.0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
            while True:
                data, _ = stream.read(1024)
                buf.append(data.copy())

                rms = float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))
                if rms < THRESHOLD:
                    silence_secs += len(data) / SAMPLE_RATE
                else:
                    silence_secs = 0.0

                total_secs += len(data) / SAMPLE_RATE
                if silence_secs >= SILENCE_SECS:
                    break
                if total_secs >= MAX_RECORD_SECS:
                    break
    except Exception as e:
        print(f"[audio] Failed to record: {e}", file=sys.stderr)
        raise

    return np.concatenate(buf, axis=0) if buf else np.zeros((0,), dtype=np.float32)


def write_temp_wav(audio: np.ndarray) -> str:
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32, copy=False)
    # DC removal and peak normalization for stability
    if audio.size:
        audio = audio - float(np.mean(audio))
        peak = float(np.max(np.abs(audio)) + 1e-9)
        audio = 0.95 * audio / peak
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, SAMPLE_RATE)
    return tmp.name


def parse_active_medicines(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    # Show provider/model and enum choices (replicates API behavior)
    provider = (LLM_PROVIDER or "openai").lower()
    if provider == "groq":
        model_info = f"GROQ | model: {GROQ_MODEL}"
    else:
        base = OPENAI_BASE_URL or "(OpenAI cloud)"
        model_info = f"OpenAI-compatible | model: {OPENAI_MODEL} | base_url: {base}"
    print(f"\nUsing LLM provider: {model_info}")
    # Show enum choices for the user (replicates API docs)
    print("\nChoices in use:")
    print("  Forms:", ", ".join(FORMS))
    print("  Frequencies:", ", ".join(FREQUENCIES))
    print("  Timings:", ", ".join(TIMINGS))
    print("  Routes:", ", ".join(ROUTES))

    # Optional: print LLM instruction for transparency
    if os.getenv("CLI_SHOW_INSTRUCTION", "0") in {"1", "true", "True"}:
        print("\nAI Instruction (schema & rules):\n" + LLM_SCHEMA_INSTRUCTION)

    # Check keys only when required
    if provider == "groq" and not os.getenv("GROQ_API_KEY"):
        print("[error] GROQ_API_KEY is not set. export GROQ_API_KEY=...", file=sys.stderr)
        return 2

    # 1) Record speech
    audio = record_until_silence()
    if audio.size == 0:
        print("[warn] no audio captured")
        return 0

    # 2) Ask for active medicines (comma separated)
    try:
        active_raw = input("Enter active medicines (comma-separated), or leave empty: ").strip()
    except EOFError:
        active_raw = ""
    active_medicines = parse_active_medicines(active_raw)

    # 3) Save to wav and transcribe
    wav_path = write_temp_wav(audio)
    try:
        transcript = transcribe_audio_file(wav_path)
    finally:
        try:
            os.unlink(wav_path)
        except Exception:
            pass
    print("\n[Transcript]\n" + transcript)

    # 4) LLM single-step JSON
    extracted = llm_extract(transcript)
    extracted = coerce_llm_final_json(extracted)
    extracted = refine_with_transcript(transcript, extracted)

    # 5) Map removed medicines against active list; keep both add fields and removals if present
    active_lower = {m.lower(): m for m in active_medicines}
    removed = []
    seen = set()
    for name in (extracted.get("removed_medicines") or []):
        if not isinstance(name, str):
            continue
        match = find_best_active_match(name, active_lower, cutoff=0.8)
        if match and match not in seen:
            seen.add(match)
            removed.append(match)
    extracted["removed_medicines"] = removed

    # Ensure all required keys exist
    response = {
        "name": (extracted.get("name") or "").strip(),
        "dose": (extracted.get("dose") or "").strip(),
        "form": (extracted.get("form") or "").strip(),
        "route": (extracted.get("route") or "").strip(),
        "frequency": (extracted.get("frequency") or "").strip(),
        "dosage_timing": (extracted.get("dosage_timing") or "").strip(),
        "duration": (extracted.get("duration") or "").strip(),
        "comments": (extracted.get("comments") or "").strip(),
        "removed_medicines": removed,
    }

    # 6) Print JSON
    print("\n[Result]\n" + json.dumps(response, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

