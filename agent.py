#!/usr/bin/env python3
import os
import re
import json
import tempfile
import sys
import threading
from typing import List, Optional, Dict, Any
from difflib import get_close_matches

import numpy as np
import sounddevice as sd
import soundfile as sf

# Heavy deps
import whisperx
from groq import Groq

# -------------------- Config --------------------# "openai" or "groq"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda"
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")

_model_lock = threading.Lock()
_whisper_model = None
_groq_client: Optional[Groq] = None


# -------------------- Choice Sets --------------------
FORMS = [
    "Tablet", "Capsule", "Syrup", "Suspension", "Injection", "Cream", "Ointment",
    "Gel", "Lotion", "Drops", "Inhaler", "Nebulizer", "Suppository", "Patch",
    "Sachet", "Powder", "Others",
]
FREQUENCIES = [
    "Once_Daily", "Twice_Daily", "Thrice_Daily", "Four_Times_Daily", "At_Bed_Time",
    "If_Required", "As_Needed", "Immediately", "Every_X_Hours", "Every_Other_Day",
    "Weekly", "Monthly",
]
TIMINGS = ["before_food", "after_food", "empty_stomach"]
ROUTES = [
    "PO", "RT", "SL", "IV", "IM", "SC", "TT", "PR", "PV", "TOPICAL",
    "OCULAR", "OTIC", "NASAL", "PEG", "Oral", "IV Infusion", "IV Push",
    "IV Piggyback", "IV Bolus", "IV Drip",
]


# -------------------- Normalizers --------------------
def _build_normalizers():
    form_map = {
        "tab": "Tablet", "tablet": "Tablet", "tablets": "Tablet",
        "cap": "Capsule", "caps": "Capsule", "capsule": "Capsule", "capsules": "Capsule",
        "syr": "Syrup", "syrup": "Syrup",
        "susp": "Suspension", "suspension": "Suspension",
        "inj": "Injection", "injection": "Injection", "shot": "Injection",
        "cream": "Cream", "ointment": "Ointment", "oint": "Ointment",
        "gel": "Gel", "lotion": "Lotion", "drops": "Drops", "drop": "Drops",
        "inhaler": "Inhaler", "neb": "Nebulizer", "nebulizer": "Nebulizer",
        "suppository": "Suppository", "patch": "Patch", "sachet": "Sachet",
        "powder": "Powder",
    }
    timing_map = {
        "before meals": "before_food", "before meal": "before_food", "before food": "before_food",
        "preprandial": "before_food", "pre-prandial": "before_food",
        "after meals": "after_food", "after meal": "after_food", "after food": "after_food",
        "postprandial": "after_food", "post-prandial": "after_food",
        "empty stomach": "empty_stomach", "on empty stomach": "empty_stomach", "fasting": "empty_stomach",
    }
    freq_map = {
        "once daily": "Once_Daily", "once a day": "Once_Daily", "od": "Once_Daily",
        "twice daily": "Twice_Daily", "bid": "Twice_Daily",
        "thrice daily": "Thrice_Daily", "tid": "Thrice_Daily",
        "four times a day": "Four_Times_Daily", "qid": "Four_Times_Daily",
        "at bedtime": "At_Bed_Time", "hs": "At_Bed_Time",
        "as needed": "As_Needed", "if required": "If_Required", "prn": "As_Needed",
        "immediately": "Immediately", "stat": "Immediately",
        "every other day": "Every_Other_Day", "alternate day": "Every_Other_Day",
        "once a week": "Weekly", "weekly": "Weekly",
        "once a month": "Monthly", "monthly": "Monthly",
    }
    route_map = {
        "po": "PO", "oral": "Oral", "by mouth": "Oral",
        "sl": "SL", "sublingual": "SL",
        "iv": "IV", "intravenous": "IV",
        "im": "IM", "intramuscular": "IM",
        "sc": "SC", "subcutaneous": "SC",
        "pr": "PR", "rectal": "PR",
        "pv": "PV", "vaginal": "PV",
        "topical": "TOPICAL",
        "ocular": "OCULAR", "eye": "OCULAR",
        "otic": "OTIC", "ear": "OTIC",
        "nasal": "NASAL", "nose": "NASAL",
        "iv infusion": "IV Infusion", "infusion": "IV Infusion",
        "iv push": "IV Push", "push": "IV Push",
        "iv piggyback": "IV Piggyback", "piggyback": "IV Piggyback",
        "iv bolus": "IV Bolus", "bolus": "IV Bolus",
        "iv drip": "IV Drip", "drip": "IV Drip",
    }
    return form_map, timing_map, freq_map, route_map

FORM_MAP, TIMING_MAP, FREQ_MAP, ROUTE_MAP = _build_normalizers()


# -------------------- Helpers --------------------
def load_whisper_model():
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            compute_type = "int8" if DEVICE == "cpu" else "float16"
            _whisper_model = whisperx.load_model(WHISPER_MODEL_NAME, DEVICE, compute_type=compute_type)
    return _whisper_model

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# -------------------- ASR --------------------
def transcribe_audio_file(audio_path: str) -> str:
    model = load_whisper_model()
    raw_audio = whisperx.load_audio(audio_path)
    result = model.transcribe(raw_audio, batch_size=8, task="translate")
    texts = [seg.get("text", "").strip() for seg in result.get("segments", []) if seg.get("text")]
    transcript = " ".join(texts)
    transcript = re.sub(r"(\d+)(?:,\s*|\s+)(\d+)", r"\1\2", transcript)
    return transcript


# -------------------- LLM Extraction --------------------
LLM_SCHEMA_INSTRUCTION = (
    "You are a medical transcription assistant. Convert the doctor's speech into a SINGLE final JSON matching the schema below. "
    "Normalize all enumerated fields to the allowed choices EXACTLY. If unsure or no match, output an empty string for that field. "
    "Only output the JSON object with exactly these keys; no extra text.\n\n"
    "Schema to output: {\n"
    "  \"name\": \"<string or empty>\",\n"
    "  \"dose\": \"<string or empty>\",\n"
    "  \"form\": \"<string or empty>\",\n"
    "  \"route\": \"<string or empty>\",\n"
    "  \"frequency\": \"<string or empty>\",\n"
    "  \"dosage_timing\": \"<string or empty>\",\n"
    "  \"duration\": \"<string or empty>\",\n"
    "  \"comments\": \"<string or empty>\",\n"
    "  \"removed_medicines\": [\"<string>\"]\n"
    "}\n\n"
    "Rules and choices to enforce strictly:\n"
    "- form choices: Tablet, Capsule, Syrup, Suspension, Injection, Cream, Ointment, Gel, Lotion, Drops, Inhaler, Nebulizer, Suppository, Patch, Sachet, Powder, Others\n"
    "- frequency choices: Once_Daily, Twice_Daily, Thrice_Daily, Four_Times_Daily, At_Bed_Time, If_Required, As_Needed, Immediately, Every_X_Hours, Every_Other_Day, Weekly, Monthly\n"
    "- timing choices: before_food, after_food, empty_stomach\n"
    "- route choices: PO, RT, SL, IV, IM, SC, TT, PR, PV, TOPICAL, OCULAR, OTIC, NASAL, PEG, Oral, IV Infusion, IV Push, IV Piggyback, IV Bolus, IV Drip\n"
    "- Accept common synonyms, mapping them to the exact choices (e.g., 'one per day' => Once_Daily; 'hs' => At_Bed_Time; 'prn' => As_Needed).\n"
    "  For oral administration synonyms such as 'by mouth', 'via mouth', 'mouth', 'orally', map route to 'Oral'. Do NOT output 'PO' unless the transcript literally says 'PO' or 'P.O.'.\n"
    "- If schedule like 'every 6 hours' occurs, set frequency to Every_X_Hours AND append the phrase 'every 6 hours' to comments (keep any existing comments; separate with '; ').\n"
    "- Dose must be a strength or measurable amount only (e.g., 650 mg, 5 mL, 2 g, 500 mcg, 10 IU).\n"
    "  NEVER put counts like '1 tablet', '2 capsules', 'puffs', or 'drops' into 'dose'. Those are counts, not dose.\n"
    "  If the speech gives only counts without a strength/amount unit, leave 'dose' as an empty string.\n"
    "  If both a count and a strength are given (e.g., 'two tablets of 650 mg'), set 'dose' to the strength only ('650 mg').\n"
    "  Map phrases like '2 per day' to frequency (Twice_Daily), not to dose.\n"
    "- Use general medical knowledge to infer likely forms for common brands when safe (e.g., Benadryl is commonly a syrup formulation). Prefer explicit speech cues over inference; if uncertain, leave 'form' empty. Do not browse the web.\n"
    "  If the unit suggests a liquid (e.g., mL) and speech mentions 'syrup' or similar, set form to 'Syrup'. If speech says 'suspension', set 'Suspension'.\n"
    "- Comments should include only additional, actionable instructions not already captured by other fields (e.g., 'check temperature daily', 'give with water').\n"
    "  Do NOT repeat information already represented in route, dosage_timing, frequency, dose, duration, or form. Ignore filler words or ASR artifacts.\n"
    "- 'comments' should capture any extra administration instructions (e.g., 'with water', 'with meals', 'apply to affected area'), especially phrases typically appearing near the end of speech.\n"
    "- If the speech indicates removal of medicines, list them in 'removed_medicines'. If it indicates adding a medicine, fill the medication fields. If both add and remove are present, include both (fill the medication fields and also list removed_medicines).\n"
)


def llm_extract(transcript: str) -> Dict[str, Any]:
    if LLM_PROVIDER == "groq":
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": LLM_SCHEMA_INSTRUCTION},
                      {"role": "user", "content": transcript}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

# -------------------- JSON Coercion & Refinement --------------------
def coerce_llm_final_json(data: Any) -> Dict[str, Any]:
    if isinstance(data, str):
        try: data = json.loads(data)
        except: data = {}
    if isinstance(data, list):
        data = next((x for x in data if isinstance(x, dict)), {})
    if not isinstance(data, dict): data = {}
    def _s(v): return (v or "").strip() if isinstance(v, str) else ""
    def _ls(v): return [s.strip() for s in v if isinstance(s, str) and s.strip()] if isinstance(v, list) else ([v.strip()] if isinstance(v, str) else [])
    return {k:_s(data.get(k,"")) for k in ["name","dose","form","route","frequency","dosage_timing","duration","comments"]} | {"removed_medicines":_ls(data.get("removed_medicines"))}

def refine_with_transcript(transcript: str, data: Dict[str, Any]) -> Dict[str, Any]:
    t = transcript.lower()
    out = dict(data)
    if any(kw in t for kw in ["by mouth","orally"]) and " po " not in t:
        out["route"] = "Oral"
    return out


# -------------------- CLI Recording --------------------
SAMPLE_RATE=16000; CHANNELS=1; THRESHOLD=0.006; SILENCE_SECS=0.8; MAX_RECORD_SECS=20.0

def record_until_silence() -> np.ndarray:
    print("\nSpeak now... (stop talking to finish)")
    buf=[]; silence=0; total=0
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        while True:
            data,_=stream.read(1024); buf.append(data.copy())
            rms=float(np.sqrt(np.mean(np.square(data),dtype=np.float64)))
            silence = silence+len(data)/SAMPLE_RATE if rms<THRESHOLD else 0
            total+=len(data)/SAMPLE_RATE
            if silence>=SILENCE_SECS or total>=MAX_RECORD_SECS: break
    return np.concatenate(buf,axis=0)

def write_temp_wav(audio: np.ndarray) -> str:
    if audio.ndim>1: audio=audio[:,0]
    audio=audio.astype(np.float32)
    if audio.size: audio=0.95*(audio-np.mean(audio))/(np.max(np.abs(audio))+1e-9)
    tmp=tempfile.NamedTemporaryFile(suffix=".wav",delete=False); sf.write(tmp.name,audio,SAMPLE_RATE); return tmp.name

def find_best_active_match(name: str, active_lower: Dict[str,str], cutoff: float=0.8)->Optional[str]:
    if not name: return None
    key=name.lower(); 
    if key in active_lower: return active_lower[key]
    matches=get_close_matches(key, list(active_lower.keys()),n=1,cutoff=cutoff)
    return active_lower[matches[0]] if matches else None


# -------------------- Main --------------------
def main()->int:
    provider=(LLM_PROVIDER or "openai").lower()
    print(f"\nUsing provider {provider}, model {GROQ_MODEL if provider=='groq' else OPENAI_MODEL}")
    audio=record_until_silence()
    if not audio.size: return 0
    actives=input("Enter active medicines (comma-separated): ").strip().split(",") if sys.stdin else []
    wav=write_temp_wav(audio)
    try: transcript=transcribe_audio_file(wav)
    finally: os.unlink(wav)
    print("\n[Transcript]\n",transcript)
    extracted=refine_with_transcript(transcript,coerce_llm_final_json(llm_extract(transcript)))
    act_map={m.lower():m for m in actives}
    removed=[]
    for r in extracted.get("removed_medicines",[]): 
        m=find_best_active_match(r,act_map)
        if m and m not in removed: removed.append(m)
    extracted["removed_medicines"]=removed
    print("\n[Result]\n",json.dumps(extracted,indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
