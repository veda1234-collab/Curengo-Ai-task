#!/usr/bin/env python3
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile, json, sys, re, argparse, time
import whisperx
from groq import Groq
import os

# ---------------- Config (defaults) ----------------
DEVICE = "cpu"   # or "cuda"
MODEL_NAME = "small"
SAMPLE_RATE = 16000
CHANNELS = 1
THRESHOLD = 0.005     # base RMS threshold floor
SILENCE_SECS = 0.7    # fast stop like once.py
MAX_RECORD_SECS = 20  # safety cap (let user speak up to 20 seconds)
# Dynamic silence detection parameters (hysteresis)
BASELINE_ALPHA = 0.10     # EMA update rate for noise floor
SPEECH_ENTER_MULT = 6.0   # enter speech when rms > baseline * this
SPEECH_EXIT_MULT = 2.5    # exit speech when rms < baseline * this

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Record speech until silence, transcribe with WhisperX, then structure with Groq LLM")
parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
parser.add_argument("--input-device", type=str, default=None, help="Input device index or name")
parser.add_argument("--model", type=str, default=MODEL_NAME, help="Whisper/WhisperX model name (e.g., tiny, small, medium, large-v2)")
parser.add_argument("--cpu", action="store_true", help="Force CPU for ASR")
parser.add_argument("--cuda", action="store_true", help="Force CUDA GPU for ASR")
parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="Microphone sample rate")
parser.add_argument("--channels", type=int, default=CHANNELS, help="Microphone channels (1=mono, 2=stereo)")
parser.add_argument("--threshold", type=float, default=THRESHOLD, help="RMS threshold to detect silence")
parser.add_argument("--silence-secs", type=float, default=SILENCE_SECS, help="Silence duration to stop recording")
parser.add_argument("--max-record-secs", type=float, default=MAX_RECORD_SECS, help="Safety cap on max recording length")
parser.add_argument("--groq-model", type=str, default=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"), help="Groq chat model id (e.g., llama-3.1-8b-instant)")
parser.add_argument("--baseline-alpha", type=float, default=BASELINE_ALPHA, help="EMA alpha for noise floor")
parser.add_argument("--enter-mult", type=float, default=SPEECH_ENTER_MULT, help="Multiplier over baseline to enter speech state")
parser.add_argument("--exit-mult", type=float, default=SPEECH_EXIT_MULT, help="Multiplier over baseline to exit speech state")
parser.add_argument("--robust-vad", action="store_true", help="Use adaptive hysteresis VAD instead of fast RMS stop")
args = parser.parse_args()

if args.list_devices:
    print("[devices] Available audio devices:", file=sys.stderr)
    try:
        default_in = None
        try:
            default_in = sd.default.device[0]
        except Exception:
            default_in = None
        for idx, dev in enumerate(sd.query_devices()):
            mark = " (default input)" if default_in is not None and idx == default_in else ""
            print(f"  {idx}: {dev['name']} | max_input_channels={dev['max_input_channels']} | default_sr={dev.get('default_samplerate', 'n/a')}{mark}", file=sys.stderr)
    except Exception as e:
        print(f"[devices] Failed to query devices: {e}", file=sys.stderr)
    sys.exit(0)

# Apply CLI overrides
MODEL_NAME = args.model
SAMPLE_RATE = args.sample_rate
CHANNELS = args.channels
THRESHOLD = args.threshold
SILENCE_SECS = args.silence_secs
MAX_RECORD_SECS = args.max_record_secs
BASELINE_ALPHA = args.baseline_alpha
SPEECH_ENTER_MULT = args.enter_mult
SPEECH_EXIT_MULT = args.exit_mult
if args.cuda:
    DEVICE = "cuda"
elif args.cpu:
    DEVICE = "cpu"

if DEVICE == "cpu" and MODEL_NAME.lower() not in {"tiny", "base", "small"}:
    print(f"[warn] Running heavy model '{MODEL_NAME}' on CPU. Consider --model small for responsiveness.", file=sys.stderr)
print(f"[init] loading WhisperX model={MODEL_NAME} on {DEVICE}")
model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type="int8" if DEVICE=="cpu" else "float16")

# Groq client (required)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("[error] GROQ_API_KEY is not set. Please export it before running.", file=sys.stderr)
    sys.exit(2)
client = Groq(api_key=GROQ_API_KEY)

# ---------------- Recording ----------------
def record_until_silence():
    """Record until silence is detected quickly after speech"""
    print("\n🎙️ Speak now... (stop talking to finish)")
    buf = []
    silence_count = 0
    total_secs = 0
    device_arg = None
    if args.input_device is not None:
        device_arg = int(args.input_device) if args.input_device.isdigit() else args.input_device
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", device=device_arg)
        stream.start()
    except Exception as e:
        print(f"[audio] Failed to open input stream: {e}. Try --list-devices and --input-device.", file=sys.stderr)
        raise
    last_log = 0.0
    baseline_rms = 0.002  # start with a small floor
    in_speech = False
    try:
        while True:
            data, _ = stream.read(1024)
            buf.append(data.copy())
            rms = np.sqrt(np.mean(data**2))
            now = time.time()
            if now - last_log >= 0.8:
                last_log = now
                if args.robust_vad:
                    enter_thr = max(THRESHOLD, baseline_rms * SPEECH_ENTER_MULT)
                    exit_thr = max(THRESHOLD, baseline_rms * SPEECH_EXIT_MULT)
                    state = "speech" if in_speech else "idle"
                    print(f"[audio] rms={float(rms):.3f} state={state} enter>{float(enter_thr):.3f} exit<{float(exit_thr):.3f}", file=sys.stderr)
                else:
                    print(f"[audio] rms={float(rms):.3f} (fast)", file=sys.stderr)

            if args.robust_vad:
                # Update baseline using EMA when below a conservative cap to avoid loud speech pulling it up
                if rms < baseline_rms * 4.0:
                    baseline_rms = (1.0 - BASELINE_ALPHA) * baseline_rms + BASELINE_ALPHA * float(rms)

                enter_thr = max(THRESHOLD, baseline_rms * SPEECH_ENTER_MULT)
                exit_thr = max(THRESHOLD, baseline_rms * SPEECH_EXIT_MULT)

                if not in_speech:
                    if rms > enter_thr:
                        in_speech = True
                        silence_count = 0
                else:
                    if rms < exit_thr:
                        silence_count += len(data) / SAMPLE_RATE
                    else:
                        silence_count = 0
            else:
                # Fast RMS-based silence like once.py
                if rms < THRESHOLD:
                    silence_count += len(data) / SAMPLE_RATE
                else:
                    silence_count = 0

            total_secs += len(data) / SAMPLE_RATE

            if silence_count >= SILENCE_SECS:
                break
            if total_secs >= MAX_RECORD_SECS:
                break
    finally:
        stream.stop(); stream.close()

    return np.concatenate(buf, axis=0)

# ---------------- WhisperX ----------------
def normalize_numbers(text: str) -> str:
    """Collapse spaced out digits into normal numbers"""
    text = re.sub(r'(\d+)(?:,\s*|\s+)(\d+)', r'\1\2', text)
    return text

def transcribe(audio):
    # Preprocess: convert to mono, DC remove, pre-emphasis, peak normalize
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32, copy=False)
    if audio.size:
        audio = audio - float(np.mean(audio))
        pre = 0.97
        # pre-emphasis filter
        emphasized = np.empty_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - pre * audio[:-1]
        peak = float(np.max(np.abs(emphasized)) + 1e-9)
        audio = 0.95 * emphasized / peak

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        audio_path = f.name

    raw_audio = whisperx.load_audio(audio_path)
    result = model.transcribe(raw_audio, batch_size=8, task="translate")
    texts = []
    for seg in result["segments"]:
        clean_text = normalize_numbers(seg["text"].strip())
        texts.append(clean_text)
    return " ".join(texts)

# ---------------- LLM JSON conversion ----------------
def convert_to_json(transcript: str):
    schema = """
    You are a medical transcription assistant.
    Convert the doctor's speech into structured JSON.

    Always follow this schema:
    {
      "patient_id": "<string or null>",
      "name": "<string or null>",
      "medications": [
        {
          "drug": "<string>",
          "dosage": "<string>",
          "frequency": "<string>"
        }
      ]
    }
    If any field is missing in the speech, set it to null.
    Only output valid JSON.
    """

    try:
        resp = client.chat.completions.create(
            model=args.groq_model,
            messages=[
                {"role": "system", "content": schema},
                {"role": "user", "content": transcript}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[llm-error] {e}", file=sys.stderr)
        raise

# ---------------- Main Loop ----------------
try:
    while True:
        audio = record_until_silence()
        if audio.size == 0:
            continue

        print("✅ Transcribing...")
        transcript = transcribe(audio)
        print(f"[raw transcript] {transcript}")

        print("🤖 Sending to LLM for JSON structuring...")
        structured = convert_to_json(transcript)
        print(json.dumps(structured, indent=2, ensure_ascii=False))

        # Ask whether to continue recording another turn
        try:
            resp = input("\nWould you like to speak again? (yes/no): ").strip().lower()
        except EOFError:
            resp = "no"
        if resp not in ("y", "yes"):
            print("\n👋 Exiting.")
            break

except KeyboardInterrupt:
    print("\n👋 Exiting.")
    sys.exit(0)
