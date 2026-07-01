<p align="center">
  <a href="./README.md">简体中文</a> | <strong>English</strong>
</p>

# JobPilot RAG-Agent

An intelligent internship job matching and resume optimization system for AI Agent, RAG, and LLM application roles.

JobPilot provides an interactive React + FastAPI workbench and uses LangGraph to orchestrate candidate profiling, JD parsing, hybrid retrieval, reranking, match scoring, gap analysis, and resume suggestions. Deterministic rules keep the system explainable and allow the core workflow to continue when DeepSeek is unavailable.

**Live demo:** [Azure Container Apps](https://jobpilot-agent.gentlefield-019d4ae8.eastasia.azurecontainerapps.io/)

## System Architecture

![JobPilot RAG-Agent System Architecture](docs/images/JobPilot.png)

## Core Features

- Candidate profile ingestion, normalization, and Pydantic validation
- Structured JD parsing and persistent SQLite job library
- BGE vector retrieval plus TF-IDF keyword retrieval
- Weighted Reciprocal Rank Fusion
- Rule-based reranking with optional LLM Top-5 reranking
- Explainable deterministic job-match scoring
- Gap analysis and targeted resume suggestions
- LangGraph conditional routing, retry, fallback, and checkpoints
- FastAPI asynchronous runs, SSE progress, cancellation, and timeout
- Trace, latency, token, and cost monitoring
- Offline retrieval and ranking evaluation

## Architecture

```text
Candidate Profile + Target Role + Job Descriptions
                       |
                       v
              React Agent Workbench
                       |
                REST API + SSE
                       |
                       v
                  FastAPI Backend
                       |
                       v
            LangGraph Conditional Workflow
                       |
        +--------------+--------------+
        |                             |
        v                             v
 ChromaDB Vector Search       TF-IDF Keyword Search
        |                             |
        +--------------+--------------+
                       |
                Weighted RRF Fusion
                       |
                       v
        Rule Reranker / Optional LLM Reranker
                       |
                       v
 Match Scoring -> Gap Analysis -> Resume Suggestions
                       |
                       v
       Recommendations + Traces + Markdown Report
```

Main LangGraph flow:

```text
START
  -> profile_node
  -> jd_parse_node
  -> retrieve_node
  -> rerank_node
  -> match_score_node
  -> gap_analysis_node
  -> resume_suggestion_node
  -> finalize_workflow_node
  -> END
```

Conditional routes handle fatal failures, administrator review, low-score deep-analysis skipping, LLM retries, and rule-based fallback. SQLite checkpoints allow interrupted runs to resume with the same `thread_id`.

## Tech Stack

- Python 3.10+
- LangGraph
- DeepSeek API / OpenAI SDK
- FastAPI
- Pydantic
- React + Vite
- ChromaDB
- BAAI/bge-small-zh-v1.5
- scikit-learn
- SQLite
- pytest + Playwright
- Docker
- Azure Container Registry
- Azure Container Apps
- Azure Files

## Project Structure

```text
JobPilot-Agent/
├── src/                 # Agent, API, retrieval, scoring, storage, and tracing
├── frontend/            # React workbench
├── data/                # Candidate data, job seed, SQLite, and vector index
├── eval/                # Offline evaluation cases and runner
├── tests/               # Backend unit and integration tests
├── traces/              # Agent execution traces
├── outputs/             # Recommendations and Markdown reports
├── scripts/             # Initialization and startup scripts
├── portfolio-blog/      # Independent Astro project showcase
├── Dockerfile
└── docker-compose.yml
```

## Local Setup

Required runtimes:

- Python 3.10.x
- Node.js 20.19.5
- npm 10.x

Install backend dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Use the complete lock file for an environment matching CI and container validation:

```powershell
python -m pip install -r requirements.lock
```

Install frontend dependencies:

```powershell
cd frontend
npm ci
```

Build and start with one command:

```powershell
.\scripts\start.ps1
```

Or start each service separately:

```powershell
py -3.10 -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`.

## Environment Variables

Copy `.env.example` to `.env`:

```env
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
LLM_TIMEOUT_SECONDS=60
LLM_SDK_MAX_RETRIES=1
```

The API key is read only by the Python backend and is never sent to the React frontend. When the key is unavailable, local rules provide retrieval, scoring, gap analysis, and resume suggestions.

Enable authentication for public deployments:

```env
JOBPILOT_AUTH_ENABLED=true
JOBPILOT_JWT_SECRET=<long-random-secret>
JOBPILOT_USER_USERNAME=jobpilot-user
JOBPILOT_USER_PASSWORD=<password>
JOBPILOT_ADMIN_USERNAME=jobpilot-admin
JOBPILOT_ADMIN_PASSWORD=<password>
JOBPILOT_RUNS_PER_MINUTE=10
```

## Retrieval and Indexing

- The vector model is pinned to `BAAI/bge-small-zh-v1.5` and a fixed revision.
- ChromaDB updates only new, changed, or deleted jobs.
- TF-IDF retains exact terms such as LangGraph, FastAPI, RAG, and DeepSeek.
- Weighted RRF combines vector and keyword rankings.
- Keyword retrieval remains available when ChromaDB or the embedding model fails.

Recommended funnel:

```text
Job Library
  -> Hybrid Retrieval Top-20
  -> Rule-based Rerank Top-10
  -> Optional LLM Rerank Top-5
  -> Match Scoring
  -> Top Jobs Deep Analysis
```

## API

- `GET /api/health`: Health and runtime configuration
- `POST /api/auth/login`: Issue a JWT when authentication is enabled
- `POST /api/runs`: Create an asynchronous run
- `GET /api/runs/{run_id}`: Read run status and result
- `GET /api/runs/{run_id}/events`: Stream node events with SSE
- `DELETE /api/runs/{run_id}`: Cancel a run
- `POST /api/runs/{run_id}/review`: Resume a checkpoint after review
- `POST /api/record-jobs`: Import JDs and incrementally refresh the index
- `GET /api/jobs`: List active jobs
- `DELETE /api/jobs/{job_id}`: Administrator soft deletion
- `POST /api/jobs/{job_id}/restore`: Administrator job restoration

## Testing and Evaluation

Backend tests:

```powershell
python -m pytest -q
```

Frontend tests and build:

```powershell
cd frontend
npm test
npm run build
```

Browser smoke test:

```powershell
cd frontend
npm run smoke:ui
```

Offline evaluation:

```powershell
py -3.10 eval/run_eval.py
```

The evaluator compares keyword, vector, hybrid union, weighted RRF, and RRF plus rule reranking. Metrics include Recall@5/10, Precision@5, Hit@5/10, MRR, NDCG@10, Top-1 accuracy, average latency, and P95 latency.

## Docker and Azure

Validate the production container locally:

```powershell
docker compose up --build
```

Cloud architecture:

```text
GitHub
  -> Docker Multi-stage Build
  -> Azure Container Registry
  -> Azure Container Apps (East Asia)
  -> Azure Files
```

The production container serves both the compiled React application and FastAPI `/api/*`. Azure Files persists the job library and retrieval index. DeepSeek configuration is injected through Azure secrets and environment variables.

## Outputs

- `outputs/matched_jobs.json`
- `outputs/resume_suggestions.json`
- `outputs/final_report.md`
- `traces/latest_trace.json`
- `eval/metrics_report.md`

## Engineering Highlights

- Built a conditional LangGraph workflow with human review, retry, fallback, and resumable checkpoints.
- Combined pinned BGE embeddings, TF-IDF, and Weighted RRF for semantic and exact-keyword retrieval.
- Reduced repeated embedding work with incremental Chroma indexing.
- Controlled cost and improved explainability with deterministic scoring and reliable LLM fallback.
- Supported real Web workflows with asynchronous FastAPI runs, SSE, cancellation, timeout, and run isolation.
- Improved reliability with Pydantic, structured traces, offline evaluation, pytest, Playwright, and GitHub Actions.

## Resume Summary

Designed and implemented an AI internship matching system using a conditional LangGraph workflow with retry, human review, fallback, and checkpoint recovery. Built hybrid retrieval with pinned BGE embeddings, TF-IDF, Weighted RRF, and incremental Chroma indexing. Exposed asynchronous FastAPI runs with SSE progress, cancellation, and timeout controls, and improved explainability and validation through Pydantic schemas, deterministic scoring, execution traces, and a 50-case offline benchmark. Deployed the full-stack application with Docker, Azure Container Registry, Azure Container Apps, and Azure Files.
