# JobPilot RAG-Agent: Internship Job Matching and Resume Optimization Agent

JobPilot RAG-Agent is an internship job matching and resume optimization agent built with LangGraph, DeepSeek API, ChromaDB, FastAPI, and React. It reads a candidate profile and local job descriptions, records reusable job postings, retrieves relevant jobs, reranks candidates, scores job fit, analyzes gaps, generates resume suggestions, records execution traces, and supports offline evaluation.

This project is designed as an engineering-oriented AI Agent system rather than a simple LLM demo. It combines structured Pydantic schemas, LangGraph orchestration, hybrid retrieval, deterministic scoring, trace logging, a Web workbench, and offline metrics.

## Core Features

- Candidate Profile Extraction
- JD Parsing
- Hybrid Retrieval
- Reranking
- Match Scoring
- Gap Analysis
- Resume Suggestions
- Job Posting Recorder
- Trace Logging
- Offline Evaluation
- FastAPI Backend
- React Agent Workbench

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
Hybrid Retriever
        ↓
Reranker
        ↓
Match Scoring Agent
        ↓
Gap Analysis Agent
        ↓
Resume Suggestion Agent
        ↓
Reports + Traces + Evaluation
```

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
npm install
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your DeepSeek API key.

```env
OPENAI_API_KEY=your_deepseek_api_key
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

The API key is only read by the Python backend. It is never sent to the React frontend. If the API key is missing or still set to the placeholder value, JobPilot runs with local rule-based fallback for retrieval, scoring, gap analysis, and resume suggestions.

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
py -3.10 -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8021
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

## API Endpoints

- `GET /api/health`: Backend health and API-key availability.
- `POST /api/run-jobpilot`: Run the JobPilot LangGraph pipeline.
- `POST /api/record-jobs`: Save JD text into `data/jobs_csv/job_records.jsonl` and generate local JD `.txt` files.
- `GET /api/latest-trace`: Read the latest saved trace.
- `GET /api/latest-report`: Read the latest Markdown report.

## Evaluation

Run offline evaluation:

```powershell
py -3.10 eval/run_eval.py
```

The evaluation pipeline measures retrieval and matching quality with deterministic fallback logic, so it can still run when the LLM API is unavailable.

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
- Uses Hybrid Retrieval + Rerank to improve job recommendation quality.
- Uses deterministic rule-based scoring as fallback to reduce LLM cost and improve explainability.
- Uses FastAPI to expose the Agent pipeline as a backend service.
- Uses React to provide an interactive Agent workbench with results, gaps, resume suggestions, job recording, and trace timeline.
- Uses Playwright smoke testing to verify the real browser interaction path.
- Uses Trace Logger to support debugging, observability, and engineering demonstration.
- Uses Offline Eval to measure Recall@K, Precision@K, Hit@K, JSON Valid Rate, and Tool Success Rate.

## Resume Description

### 中文版

JobPilot RAG-Agent 是一个面向 AI Agent / RAG / LLM 应用实习岗位申请的智能匹配与简历优化系统。项目使用 LangGraph 编排多节点 Agent 工作流，集成 DeepSeek API、OpenAI SDK、FastAPI、React、Pydantic、ChromaDB 和 scikit-learn，实现候选人画像抽取、岗位 JD 解析、岗位记录入库、Hybrid Retrieval、Rerank、规则化匹配评分、技能差距分析、简历优化建议、执行 Trace 记录、Web 交互工作台和离线评测。系统通过结构化 schema 校验 LLM 输出，并使用离线评测报告统计 Recall@K、Precision@K、Hit@K 等指标，提升推荐质量和工程可解释性。

### English

JobPilot RAG-Agent is an AI Agent system for internship job matching and resume optimization in AI Agent, RAG, and LLM application roles. The project uses LangGraph to orchestrate a multi-node workflow and integrates DeepSeek API, OpenAI SDK, FastAPI, React, Pydantic, ChromaDB, and scikit-learn. It supports candidate profile extraction, JD parsing, job posting recording, hybrid retrieval, reranking, deterministic match scoring, gap analysis, resume suggestions, trace logging, a Web workbench, and offline evaluation. The system validates structured outputs with Pydantic and reports metrics such as Recall@K, Precision@K, and Hit@K to improve recommendation quality and engineering explainability.
