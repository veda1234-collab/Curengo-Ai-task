#!/usr/bin/env python3
import os
import re
import threading
import json
from typing import List, Optional, Dict, Any
from difflib import get_close_matches

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Heavy deps
import whisperx
import ollama


# -------------------- Config --------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" or "cuda"
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")


# -------------------- Globals --------------------
app = FastAPI(
    title="Curengo Medication Processor",
    version="1.0.0",
    description=(
        "API to transcribe doctor audio, detect add/remove intent, and return structured medication JSON.\n\n"
        "Form choices: Tablet, Capsule, Syrup, Suspension, Injection, Cream, Ointment, Gel, Lotion, Drops, Inhaler, Nebulizer, Suppository, Patch, Sachet, Powder, Others.\n\n"
        "Frequency choices: Once_Daily, Twice_Daily, Thrice_Daily, Four_Times_Daily, At_Bed_Time, If_Required, As_Needed, Immediately, Every_X_Hours, Every_Other_Day, Weekly, Monthly.\n\n"
        "Timing choices: before_food, after_food, empty_stomach.\n\n"
        "Route choices: PO, RT, SL, IV, IM, SC, TT, PR, PV, TOPICAL, OCULAR, OTIC, NASAL, PEG, Oral, IV Infusion, IV Push, IV Piggyback, IV Bolus, IV Drip."
    ),
    contact={"name": "Curengo", "url": "https://example.com"},
)

_model_lock = threading.Lock()
_whisper_model = None


# -------------------- Pydantic Models --------------------
class ProcessRequest(BaseModel):
    audio_path: str = Field(..., description="Absolute path to audio file")
    active_medicines: List[str] = Field(default_factory=list)

    class Config:
        schema_extra = {
            "example": {
                "audio_path": "/abs/path/to/audio.wav",
                "active_medicines": ["dolo", "paracetamol"]
            }
        }


class ProcessResponse(BaseModel):
    name: str = ""
    dose: str = ""
    form: str = ""
    route: str = ""
    frequency: str = ""
    dosage_timing: str = ""
    duration: str = ""
    comments: str = ""
    removed_medicines: List[str] = Field(default_factory=list)

    class Config:
        schema_extra = {
            "examples": {
                "add": {
                    "summary": "Add intent response",
                    "value": {
                        "name": "amoxicillin",
                        "dose": "500 mg",
                        "form": "Tablet",
                        "route": "Oral",
                        "frequency": "Once_Daily",
                        "dosage_timing": "after_food",
                        "duration": "7 days",
                        "comments": "",
                        "removed_medicines": []
                    }
                },
                "remove": {
                    "summary": "Remove intent response",
                    "value": {
                        "name": "",
                        "dose": "",
                        "form": "",
                        "route": "",
                        "frequency": "",
                        "dosage_timing": "",
                        "duration": "",
                        "comments": "",
                        "removed_medicines": ["dolo", "paracetamol"]
                    }
                }
            }
        }


# -------------------- Choice Sets and Normalizers --------------------
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

    # Frequency phrases to canonical
    freq_map = {
        # once daily
        "once daily": "Once_Daily", "once a day": "Once_Daily", "one per day": "Once_Daily",
        "one a day": "Once_Daily", "od": "Once_Daily", "q.d": "Once_Daily", "qd": "Once_Daily",
        # twice daily
        "twice daily": "Twice_Daily", "two times a day": "Twice_Daily", "bid": "Twice_Daily", "b.i.d": "Twice_Daily",
        # thrice daily
        "thrice daily": "Thrice_Daily", "three times a day": "Thrice_Daily", "tid": "Thrice_Daily", "t.i.d": "Thrice_Daily",
        # four times daily
        "four times a day": "Four_Times_Daily", "qid": "Four_Times_Daily", "q.i.d": "Four_Times_Daily",
        # bedtime
        "at bedtime": "At_Bed_Time", "bedtime": "At_Bed_Time", "hs": "At_Bed_Time",
        # prn
        "as needed": "As_Needed", "if required": "If_Required", "prn": "As_Needed",
        # immed
        "immediately": "Immediately", "stat": "Immediately",
        # every other day
        "every other day": "Every_Other_Day", "alternate day": "Every_Other_Day",
        # weekly / monthly
        "once a week": "Weekly", "weekly": "Weekly",
        "once a month": "Monthly", "monthly": "Monthly",
    }

    route_map = {
        "po": "PO", "oral": "Oral", "by mouth": "Oral",
        "rt": "RT",  # rectal tube, if needed
        "sl": "SL", "sublingual": "SL",
        "iv": "IV", "intravenous": "IV",
        "im": "IM", "intramuscular": "IM",
        "sc": "SC", "subcutaneous": "SC", "sub-cutaneous": "SC",
        "tt": "TT",
        "pr": "PR", "rectal": "PR",
        "pv": "PV", "vaginal": "PV",
        "topical": "TOPICAL", "apply to skin": "TOPICAL",
        "ocular": "OCULAR", "eye": "OCULAR",
        "otic": "OTIC", "ear": "OTIC",
        "nasal": "NASAL", "nose": "NASAL",
        "peg": "PEG",
        # IV methods
        "iv infusion": "IV Infusion", "infusion": "IV Infusion",
        "iv push": "IV Push", "push": "IV Push",
        "iv piggyback": "IV Piggyback", "piggyback": "IV Piggyback",
        "iv bolus": "IV Bolus", "bolus": "IV Bolus",
        "iv drip": "IV Drip", "drip": "IV Drip",
    }

    return form_map, timing_map, freq_map, route_map


FORM_MAP, TIMING_MAP, FREQ_MAP, ROUTE_MAP = _build_normalizers()


def _normalize_from_map(value: str, mapping: Dict[str, str], allowed: List[str]) -> str:
    if not value:
        return ""
    val = value.strip().lower()
    # exact mapping
    if val in mapping:
        return mapping[val]
    # try normalization by tokens
    for key, canon in mapping.items():
        if key in val:
            return canon
    # final pass: direct allowed exact
    for a in allowed:
        if val == a.lower():
            return a
    # strict: only allowed choices; otherwise empty
    return ""


def normalize_medicine_fields(med: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(med)
    normalized["form"] = _normalize_from_map(med.get("form", ""), FORM_MAP, FORMS)
    # frequency requires special handling for "every X hours"
    freq_raw = med.get("frequency", "") or ""
    freq_norm = _normalize_from_map(freq_raw, FREQ_MAP, FREQUENCIES)
    every_x = re.search(r"every\s+(\d{1,2})\s*(hours|hour|hrs|hr|h)\b", freq_raw.lower())
    if every_x:
        freq_norm = "Every_X_Hours"
        # append detail into comments
        detail = every_x.group(1)
        comments = (med.get("comments") or "").strip()
        detail_text = f"every {detail} hours"
        normalized["comments"] = detail_text if not comments else f"{comments}; {detail_text}"
    normalized["frequency"] = freq_norm

    normalized["dosage_timing"] = _normalize_from_map(med.get("dosage_timing", ""), TIMING_MAP, TIMINGS)
    normalized["route"] = _normalize_from_map(med.get("route", ""), ROUTE_MAP, ROUTES)

    # duration: normalize number words like "seven days" -> "7 days"
    normalized["duration"] = normalize_duration_string(med.get("duration", ""))

    return normalized


_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
}


def normalize_duration_string(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    # every simple patterns
    m = re.search(r"(\d{1,3})\s*(day|days|week|weeks|month|months)\b", t)
    if m:
        num = m.group(1)
        unit = m.group(2)
        return f"{num} {unit}"
    # word numbers
    for w, n in _NUM_WORDS.items():
        if re.search(rf"\b{w}\b", t):
            unit = "days"
            if "week" in t:
                unit = "weeks"
            elif "month" in t:
                unit = "months"
            return f"{n} {unit}"
    return text


# -------------------- WhisperX Init --------------------
def load_whisper_model():
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            compute_type = "int8" if DEVICE == "cpu" else "float16"
            _whisper_model = whisperx.load_model(WHISPER_MODEL_NAME, DEVICE, compute_type=compute_type)
    return _whisper_model
@app.on_event("startup")
def _startup():
    load_whisper_model()


# -------------------- ASR --------------------
def transcribe_audio_file(audio_path: str) -> str:
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model = load_whisper_model()
    raw_audio = whisperx.load_audio(audio_path)
    result = model.transcribe(raw_audio, batch_size=8, task="translate")
    texts: List[str] = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if text:
            texts.append(text)
    transcript = " ".join(texts)
    # minor normalization: collapse spaced digits like "1 0 0 mg" -> "100 mg"
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
    # Local Ollama chat; ensure model is pulled in the host
    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": LLM_SCHEMA_INSTRUCTION},
            {"role": "user", "content": transcript},
        ],
        options={"num_ctx": 2048},
        format="json",
    )
    content = resp.get("message", {}).get("content", "{}")
    try:
        return json.loads(content)
    except Exception:
        return {}


# -------------------- LLM Output Coercion --------------------
def coerce_llm_final_json(data: Any) -> Dict[str, Any]:
    """Coerce model output into the expected final JSON object shape.

    Handles cases where the model returns a list, a stringified JSON, or missing keys.
    """
    # Stringified JSON
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = {}

    # If list, try to find a dict entry
    if isinstance(data, list):
        chosen: Dict[str, Any] = {}
        for item in data:
            if isinstance(item, dict):
                chosen = item
                break
        data = chosen

    # If not dict, fallback to empty
    if not isinstance(data, dict):
        data = {}

    def _s(val: Any) -> str:
        return (val or "").strip() if isinstance(val, str) else ""

    def _list_str(val: Any) -> List[str]:
        if isinstance(val, list):
            out: List[str] = []
            for v in val:
                if isinstance(v, str):
                    s = v.strip()
                    if s:
                        out.append(s)
            return out
        if isinstance(val, str):
            s = val.strip()
            return [s] if s else []
        return []

    coerced = {
        "name": _s(data.get("name")),
        "dose": _s(data.get("dose")),
        "form": _s(data.get("form")),
        "route": _s(data.get("route")),
        "frequency": _s(data.get("frequency")),
        "dosage_timing": _s(data.get("dosage_timing")),
        "duration": _s(data.get("duration")),
        "comments": _s(data.get("comments")),
        "removed_medicines": _list_str(data.get("removed_medicines")),
    }

    return coerced

def refine_with_transcript(transcript: str, data: Dict[str, Any]) -> Dict[str, Any]:
    t = (transcript or "").lower()
    out = dict(data)

    mouth_present = any(kw in t for kw in ["by mouth", "via mouth", " mouth", "orally", "oral "])
    po_literal = any(tok in t for tok in [" po ", " p.o ", " p.o."])
    if mouth_present and not po_literal:
        out["route"] = "Oral"

    if "empty stomach" in t:
        out["dosage_timing"] = "empty_stomach"
    elif any(kw in t for kw in ["after food", "after foods", "after meal", "after meals"]):
        out["dosage_timing"] = "after_food"

    if not out.get("frequency"):
        m = re.search(r"(\d+)\s*(?:x\s*)?(?:per|/)?\s*day", t)
        if m:
            n = m.group(1)
            if n == "1":
                out["frequency"] = "Once_Daily"
            elif n == "2":
                out["frequency"] = "Twice_Daily"
            elif n == "3":
                out["frequency"] = "Thrice_Daily"
            elif n == "4":
                out["frequency"] = "Four_Times_Daily"

    dur = (out.get("duration") or "").strip()
    if dur and re.fullmatch(r"\d+", dur):
        m = re.search(r"(\d{1,3})\s*(day|days|week|weeks|month|months)\b", t)
        if m:
            out["duration"] = f"{m.group(1)} {m.group(2)}"
        else:
            out["duration"] = ""

    comments = (out.get("comments") or "").strip()
    c = comments.lower()
    for phrase in ["empty stomach", "after food", "after meals", "by mouth", "via mouth", "orally", " mouth"]:
        c = c.replace(phrase, "").strip()
    c = re.sub(r"\s+", " ", c).strip(",; .")
    out["comments"] = c

    return out

# -------------------- Matching Helpers --------------------
def find_best_active_match(name: str, active_lower: Dict[str, str], cutoff: float = 0.8) -> Optional[str]:
    if not name:
        return None
    key = name.strip().lower()
    if key in active_lower:
        return active_lower[key]
    keys = list(active_lower.keys())
    matches = get_close_matches(key, keys, n=1, cutoff=cutoff)
    if matches:
        return active_lower[matches[0]]
    return None


# -------------------- Core Handler --------------------
@app.post(
    "/process",
    response_model=ProcessResponse,
    tags=["medications"],
    summary="Process doctor audio into structured medication JSON",
    description=(
        "Accepts an audio file path and current active medicines. Transcribes audio, detects intent (add/remove),\n"
        "normalizes to defined choices, and returns the JSON."
    ),
    responses={
        200: {
            "description": "Successful processing",
            "content": {
                "application/json": {
                    "examples": {
                        "add": {
                            "summary": "Add intent",
                            "value": {
                                "name": "amoxicillin",
                                "dose": "500 mg",
                                "form": "Tablet",
                                "route": "Oral",
                                "frequency": "Once_Daily",
                                "dosage_timing": "after_food",
                                "duration": "7 days",
                                "comments": "",
                                "removed_medicines": []
                            }
                        },
                        "remove": {
                            "summary": "Remove intent",
                            "value": {
                                "name": "",
                                "dose": "",
                                "form": "",
                                "route": "",
                                "frequency": "",
                                "dosage_timing": "",
                                "duration": "",
                                "comments": "",
                                "removed_medicines": ["dolo", "paracetamol"]
                            }
                        }
                    }
                }
            }
        }
    }
)
def process_audio(req: ProcessRequest) -> ProcessResponse:
    try:
        transcript = transcribe_audio_file(req.audio_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR error: {e}")

    try:
        extracted = llm_extract(transcript)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM extraction error: {e}")

    # Coerce to expected object shape, then refine with transcript cues
    extracted = coerce_llm_final_json(extracted)
    extracted = refine_with_transcript(transcript, extracted)

    # Initialize output from model's single-step JSON
    output = ProcessResponse()

    output.name = (extracted.get("name") or "").strip()
    output.dose = (extracted.get("dose") or "").strip()
    output.form = (extracted.get("form") or "").strip()
    output.route = (extracted.get("route") or "").strip()
    output.frequency = (extracted.get("frequency") or "").strip()
    output.dosage_timing = (extracted.get("dosage_timing") or "").strip()
    output.duration = (extracted.get("duration") or "").strip()
    output.comments = (extracted.get("comments") or "").strip()

    # Map removed medicines against active list for canonicalization
    to_remove_raw = extracted.get("removed_medicines") or []
    if isinstance(to_remove_raw, list):
        active_lower = {m.lower(): m for m in req.active_medicines}
        removed: List[str] = []
        seen = set()
        for name in to_remove_raw:
            if not isinstance(name, str):
                continue
            match = find_best_active_match(name, active_lower, cutoff=0.8)
            if match and match not in seen:
                seen.add(match)
                removed.append(match)
        output.removed_medicines = removed

    return output


# -------------------- Health --------------------
@app.get("/health", tags=["health"], summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# -------------------- Dev Entrypoint --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


