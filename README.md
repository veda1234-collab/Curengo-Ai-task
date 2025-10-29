# Curengo-Ai-task

## FastAPI Service

Set environment variables (OpenAI by default):

```bash
export OPENAI_API_KEY=YOUR_OPENAI_KEY
# optional (defaults shown)
export LLM_PROVIDER=openai
export OPENAI_MODEL=gpt-4o-mini
export WHISPER_DEVICE=cpu  # or cuda
export WHISPER_MODEL=small
```

To use Groq instead (optional):

```bash
export LLM_PROVIDER=groq
export GROQ_API_KEY=YOUR_GROQ_KEY
export GROQ_MODEL=llama-3.1-8b-instant
```

Use a free local model (no cloud):

- LM Studio (OpenAI-compatible):
  ```bash
  # In LM Studio: start a local server (defaults to http://localhost:1234/v1) and select a model
  export LLM_PROVIDER=openai
  export OPENAI_BASE_URL=http://localhost:1234/v1
  export OPENAI_MODEL=TheModelNameShownInLMStudio
  # OPENAI_API_KEY optional; app will use a dummy key if base_url is set
  # export OPENAI_API_KEY=lm-studio
  ```

- Ollama (OpenAI-compatible):
  ```bash
  # Install ollama and pull a model, e.g. llama3.1:8b-instruct
  ollama run llama3.1:8b-instruct
  export LLM_PROVIDER=openai
  export OPENAI_BASE_URL=http://localhost:11434/v1
  export OPENAI_MODEL=llama3.1:8b-instruct
  # OPENAI_API_KEY optional; app will use a dummy key if base_url is set
  # export OPENAI_API_KEY=ollama
  ```

Install dependencies (inside your venv):

```bash
pip install -r requirements.txt
```

Run the API:

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
```

POST /process with JSON body:

```json
{
  "audio_path": "/absolute/path/to/audio.wav",
  "active_medicines": ["dolo", "paracetamol"]
}
```

Response (add intent):

```json
{
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
```

Response (remove intent):

```json
{
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
```

## Console Local Test (no FastAPI)

Record via microphone, then input active medicines in console, and get the same JSON result:

```bash
# Make sure environment is set as above (OpenAI or local server)
python local_cli.py
```

Flow:
- Speak and pause to auto-stop recording
- When prompted, enter active medicines as comma-separated values (e.g., `dolo, paracetamol`)
- The result JSON will be printed