# JobPilot RAG-Agent: Internship Job Matching and Resume Optimization Agent

JobPilot RAG-Agent is an internship job matching and resume optimization agent built with LangGraph, DeepSeek API, ChromaDB, FastAPI, and React. It reads a candidate profile and local job descriptions, records reusable job postings, retrieves relevant jobs, reranks candidates, scores job fit, analyzes gaps, generates resume suggestions, records execution traces, and supports offline evaluation.

This project is designed as an engineering-oriented AI Agent system rather than a simple LLM demo. It combines structured Pydantic schemas, LangGraph orchestration, hybrid retrieval, deterministic scoring, trace logging, a Web workbench, and offline metrics.

## Core Features

- Candidate Profile Extraction
- JD Parsing
- Weighted RRF Hybrid Retrieval
- Reranking
- Match Scoring
- Gap Analysis
- Resume Suggestions
- Job Posting Recorder
- Trace Logging
- Offline Evaluation
- FastAPI Backend
- React Agent Workbench
- Async Runs, SSE Progress, Cancellation, Timeout, and Cache
- JWT Authentication, RBAC, Rate Limiting, and Audit Logging

## Architecture

```text
User Profile + Job Descriptions
        ↓
React Agent Workbench
        ↓
FastAPI Backend
        ↓
Profile Agent / JD Parser Agent
        ↓
Keyword + BGE Vector Retriever
        ↓
Weighted Reciprocal Rank Fusion
        ↓
Rule Reranker / Optional LLM Reranker
        ↓
Match Scoring Agent
        ↓
Gap Analysis Agent
        ↓
Resume Suggestion Agent
        ↓
Reports + Traces + Evaluation
```

The graph uses conditional routing. Fatal profile/JD failures halt the workflow,
high JD parse-failure rates can interrupt for administrator review, and low
match scores skip expensive deep analysis. SQLite checkpoints allow interrupted
runs to resume with the same `thread_id`.

## Tech Stack

- Python 3.10+
- LangGraph
- DeepSeek API
- OpenAI SDK
- FastAPI
- React
- Vite
- Pydantic
- ChromaDB
- scikit-learn
- pytest

## Installation

On Windows:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm ci
```

One-command local build and startup:

```powershell
.\scripts\start.ps1
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your DeepSeek API key.

```env
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
LLM_TIMEOUT_SECONDS=60
LLM_SDK_MAX_RETRIES=1
LLM_INPUT_COST_PER_MILLION=0
LLM_OUTPUT_COST_PER_MILLION=0
```

The API key is only read by the Python backend. It is never sent to the React frontend. If the API key is missing or still set to the placeholder value, JobPilot runs with local rule-based fallback for retrieval, scoring, gap analysis, and resume suggestions.

For public deployment, enable JWT authentication and configure separate user
and administrator accounts:

```env
JOBPILOT_AUTH_ENABLED=true
JOBPILOT_JWT_SECRET=<long-random-secret>
JOBPILOT_USER_USERNAME=jobpilot-user
JOBPILOT_USER_PASSWORD=<password>
JOBPILOT_ADMIN_USERNAME=jobpilot-admin
JOBPILOT_ADMIN_PASSWORD=<password>
JOBPILOT_RUNS_PER_MINUTE=10
```

## CLI Run

```powershell
py -3.10 -m src.main
```

Optional arguments:

```powershell
py -3.10 -m src.main --profile data/user_profile.json --jd-folder data/sample_jds --target-role "AI Agent Intern"
```

## Web App Run

Start the FastAPI backend:

```powershell
py -3.10 -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Start the React frontend in a second terminal:

```powershell
cd frontend
npm run dev
```

If `5173` is already occupied, start Vite on another allowed port:

```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5176
```

Open:

```text
http://127.0.0.1:5173
```

or, when using the fallback frontend port:

```text
http://127.0.0.1:5176
```

Run a real-browser UI smoke test after both services are running:

```powershell
cd frontend
npm run smoke:ui
```

The smoke test opens the React app with Playwright, saves sample JDs into the local job library, clicks `Run Agent`, waits for matched jobs, checks the Gaps, Resume, Trace, and Report tabs, verifies Markdown export/copy actions, then saves a screenshot to `outputs/jobpilot_ui_smoke.png`.

## Azure App Service Deployment

The root `Dockerfile` packages the React frontend and FastAPI backend into one
production container. FastAPI serves both `/api/*` and the compiled React app.
On an empty deployment, `data/job_seed.json` initializes the SQLite job library.

Recommended Azure App Service settings:

```text
Region=East Asia
Operating System=Linux
Publish=Container
Container Port=8000
JOBPILOT_DATA_DIR=/home/jobpilot-data
WEBSITES_ENABLE_APP_SERVICE_STORAGE=true
OPENAI_API_KEY=<DeepSeek API Key>
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

Keep `.env` local. Configure API keys only through Azure App Service environment
variables.

The same production container can be tested locally:

```powershell
docker compose up --build
```

## API Endpoints

- `GET /api/health`: Backend health and API-key availability.
- `POST /api/auth/login`: Issue a role-bearing JWT when authentication is enabled.
- `POST /api/runs`: Create an asynchronous run and return `202 + run_id`.
- `GET /api/runs/{run_id}`: Read owner-isolated run status and result.
- `GET /api/runs/{run_id}/events`: Stream node/status events with SSE.
- `DELETE /api/runs/{run_id}`: Cooperatively cancel a queued or running task.
- `POST /api/runs/{run_id}/review`: Administrator review and checkpoint resume.
- `POST /api/run-jobpilot`: Backward-compatible synchronous pipeline endpoint.
- `POST /api/record-jobs`: Administrator-only JD import with incremental index refresh.
- `GET /api/jobs`: List active jobs from SQLite.
- `DELETE /api/jobs/{job_id}`: Administrator-only soft deletion.
- `POST /api/jobs/{job_id}/restore`: Administrator-only restoration of a disabled job.
- `GET /api/latest-trace`: Read the latest saved trace.
- `GET /api/latest-report`: Read the latest Markdown report.

## Evaluation

Run offline evaluation:

```powershell
py -3.10 eval/run_eval.py
```

The 50-case evaluation compares five deterministic baselines: keyword,
vector, simple hybrid union, weighted RRF hybrid, and RRF hybrid plus rule rerank. It reports Recall@5/10,
Precision@5, Hit@5/10, MRR, NDCG@10, Top-1 accuracy, average/P95 latency,
fallback rate, JSON validity, tool success, Token usage, and estimated cost.

## Outputs

- `outputs/matched_jobs.json`: Ranked job matching results.
- `outputs/resume_suggestions.json`: Resume optimization suggestions for top jobs.
- `outputs/final_report.md`: Human-readable final report.
- `traces/latest_trace.json`: Structured execution trace for debugging and presentation.
- `eval/metrics_report.md`: Offline evaluation report with Recall@K, Precision@K, Hit@K, JSON Valid Rate, Tool Success Rate, and average match score.
- `data/jobs_csv/job_records.jsonl`: Append-only job posting records.
- `data/sample_jds/*.txt`: Local JD files generated for matching.

## Project Highlights

- Not a simple LLM Demo: combines retrieval, reranking, scoring, generation, tracing, evaluation, and a real Web interface.
- Uses LangGraph to orchestrate a multi-node Agent workflow.
- Uses Pydantic to validate structured LLM outputs and internal data contracts.
- Uses a pinned `BAAI/bge-small-zh-v1.5` revision and weighted RRF to fuse vector and exact-keyword retrieval.
- Updates only new, changed, or deleted jobs in the persistent Chroma index.
- Uses deterministic rule-based scoring as fallback to reduce LLM cost and improve explainability.
- Uses conditional LangGraph routes, retry/timeout controls, human-review interrupts, and SQLite checkpoints.
- Uses asynchronous run persistence, SSE progress, cooperative cancellation, owner-scoped caching, and per-run output isolation.
- Uses optional JWT authentication, user/admin RBAC, rate limiting, soft deletion, and audit logging.
- Uses FastAPI to expose the Agent pipeline as a backend service.
- Uses React to provide an interactive Agent workbench with results, gaps, resume suggestions, job recording, and trace timeline.
- Uses Playwright smoke testing to verify the real browser interaction path.
- Uses Trace Logger to support debugging, observability, and engineering demonstration.
- Uses Offline Eval to compare five retrieval/fusion baselines across 50 manually reviewed cases.
- Uses GitHub Actions to run backend tests, frontend build, browser smoke tests, evaluation, and Docker build.

## Resume Description

### 中文版

JobPilot RAG-Agent 是一个面向 AI Agent / RAG / LLM 实习岗位的智能匹配与简历优化系统。使用 LangGraph 构建支持条件路由、人工审核中断与 SQLite checkpoint 恢复的 Agent 工作流；基于固定版本 BGE 向量模型、TF-IDF 关键词召回和加权 RRF 实现混合检索，并通过增量 Chroma 索引降低重复 embedding 成本。使用 FastAPI 提供异步任务、SSE 进度、取消、超时、缓存和用户隔离能力，结合 JWT/RBAC、限流、审计日志和 Pydantic 结构化校验增强部署可靠性。构建 50 条人工标注评测集，对比五组检索与融合基线并统计 Recall、MRR、NDCG、P95 延迟、降级率和成本指标。

### English

Built a production-oriented internship matching Agent with conditional LangGraph orchestration, resumable SQLite checkpoints, DeepSeek structured output, weighted RRF retrieval over pinned BGE embeddings and TF-IDF keywords, incremental Chroma indexing, and deterministic fallback scoring. Exposed asynchronous FastAPI runs with SSE progress, cancellation, timeout, owner-scoped cache, JWT/RBAC, rate limiting, and audit logs. Created a 50-case offline benchmark comparing five retrieval and fusion baselines with Recall, MRR, NDCG, P95 latency, fallback-rate, and cost reporting.
