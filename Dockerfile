# --- BASE PYTHON IMAGE ---
FROM python:3.12-slim

# --- SYSTEM DEPENDENCIES (for sound + torch + ffmpeg etc.) ---
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- INSTALL PYTHON DEPENDENCIES ---
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- COPY APP CODE ---
COPY . /app
WORKDIR /app

# --- ENVIRONMENT VARIABLES (Optional: recommended to use .env in prod)---
ENV OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
ENV WHISPER_DEVICE=cpu
ENV WHISPER_MODEL=small
ENV PORT=8000
ENV OLLAMA_NO_GPU=1


EXPOSE 8000


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
