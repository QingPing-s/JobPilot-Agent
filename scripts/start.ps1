$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.10 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
Push-Location frontend
npm ci
npm run build
Pop-Location
& .\.venv\Scripts\python.exe scripts\init_data.py
& .\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
