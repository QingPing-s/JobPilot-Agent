$ErrorActionPreference = "Stop"

$pythonVersion = py -3.10 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pythonVersion.Trim() -ne "3.10") {
    throw "JobPilot requires Python 3.10.x. Detected: $pythonVersion"
}

$nodeVersion = (node --version).TrimStart("v")
$nodeMajor = [int]$nodeVersion.Split(".")[0]
if ($nodeMajor -lt 20 -or $nodeMajor -ge 25) {
    throw "JobPilot requires Node.js >=20 and <25. Detected: $nodeVersion"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.10 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install -r requirements.lock
& .\.venv\Scripts\python.exe -m pip check
Push-Location frontend
npm ci
npm run build
Pop-Location
& .\.venv\Scripts\python.exe scripts\init_data.py
& .\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
