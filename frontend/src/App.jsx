import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Briefcase,
  CheckCircle2,
  ChevronDown,
  FileJson,
  FileText,
  Loader2,
  LockKeyhole,
  LogOut,
  Play,
  Upload,
} from "lucide-react";

import { getAccessToken, getAuthRole, jobPilotApi } from "./api/jobpilot";
import { LoginPanel } from "./components/auth/LoginPanel";
import { JobLibraryPanel, JobSourceSelector } from "./components/jobs/JobLibraryPanel";
import { ProfileEditor } from "./components/profile/ProfileEditor";
import { ResultWorkspace } from "./components/results/ResultWorkspace";
import { useJobPilotRun } from "./hooks/useJobPilotRun";
import {
  asArray,
  parseProfileInput,
  splitJdText,
  validateProfileInput,
} from "./utils/jobpilot";

const SAMPLE_PROFILE = {
  name: "AAA建材",
  education: ["研究生学历", "人工智能专业"],
  skills: ["Python", "RAG", "LLM", "Prompt Engineering", "API integration", "Pytest"],
  soft_skills: ["学习速度快", "主动查阅资料", "问题拆解能力", "自驱力强", "沟通协作能力", "责任心强", "能独立解决问题"],
  projects: [
    {
      name: "Mini RAG Assistant",
      description: "Built a local document QA prototype with retrieval and prompt templates.",
      tech_stack: ["Python", "RAG", "Vector Search"],
      highlights: ["Implemented retrieval flow", "Added basic tests and trace logs"],
    },
  ],
  internships: [],
  target_roles: ["AI Agent Intern", "RAG Intern", "LLM Application Intern"],
  preferences: {
    location: "北京",
    degree: "研究生",
    major: "人工智能",
    days_per_week: 5,
    duration_months: 6,
    work_mode: "线下",
    availability: "每周5天，可持续6个月，接受线下实习",
  },
};

const SAMPLE_JD = `Title: AI Agent 开发实习生
Company: 示例科技
Location: 北京
Salary: 300-500元/天
Education: 本科及以上
Responsibilities:
- 设计 Agent 工作流、工具调用和状态管理
- 建设 RAG 知识库并优化检索质量
- 开发 FastAPI 后端接口并补充自动化测试
Requirements:
- Python
- LangGraph
- RAG
- Prompt Engineering
Preferred:
- ChromaDB
- MCP
- Docker`;

function StatusPill({ health }) {
  if (!health) return <span className="status-pill">检查服务</span>;
  if (health.status === "offline") return <span className="status-pill warning">后端未连接</span>;
  return (
    <span className={`status-pill ${health.api_available ? "online" : "warning"}`}>
      {health.api_available ? "DeepSeek 已启用" : "规则模式"}
    </span>
  );
}

function CollapsibleSection({ children, defaultOpen = true, meta, title }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`input-section ${open ? "open" : ""}`}>
      <button className="section-trigger" onClick={() => setOpen((value) => !value)} type="button">
        <span>{title}</span>
        <div>
          {meta && <em>{meta}</em>}
          <ChevronDown size={15} />
        </div>
      </button>
      {open && <div className="section-body">{children}</div>}
    </section>
  );
}

export default function App() {
  const {
    cancel,
    durationMs,
    error: runError,
    events,
    loading,
    result,
    run,
    runId,
    status,
  } = useJobPilotRun();
  const [health, setHealth] = useState(null);
  const [authRole, setAuthRole] = useState(getAccessToken() ? getAuthRole() || "user" : "user");
  const [adminDialogOpen, setAdminDialogOpen] = useState(false);
  const [targetRole, setTargetRole] = useState("AI Agent Intern");
  const [profileInput, setProfileInput] = useState(JSON.stringify(SAMPLE_PROFILE, null, 2));
  const [profileDocumentLoading, setProfileDocumentLoading] = useState(false);
  const [jdInput, setJdInput] = useState("");
  const [uploadedJds, setUploadedJds] = useState([]);
  const [sourceMode, setSourceMode] = useState("library");
  const [jobLibrary, setJobLibrary] = useState([]);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [jobMutation, setJobMutation] = useState("");
  const [retrievalTopK, setRetrievalTopK] = useState(20);
  const [useLlmRerank, setUseLlmRerank] = useState(false);
  const [useLlmMatch, setUseLlmMatch] = useState(false);
  const [deepAnalysis, setDeepAnalysis] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const profileStatus = useMemo(() => validateProfileInput(profileInput), [profileInput]);
  const manualJds = useMemo(() => splitJdText(jdInput), [jdInput]);
  const jdCount = sourceMode === "library" ? jobLibrary.length : uploadedJds.length + manualJds.length;
  const canRun = profileStatus.status !== "empty" && jdCount > 0 && !libraryLoading;

  useEffect(() => {
    jobPilotApi
      .health()
      .then(setHealth)
      .catch(() => setHealth({ status: "offline", api_available: false, auth_enabled: false }));
  }, []);

  useEffect(() => {
    if (health && (!health.auth_enabled || health.public_access || getAccessToken())) refreshJobLibrary();
  }, [health, authRole]);

  async function refreshJobLibrary() {
    setLibraryLoading(true);
    try {
      const response = await jobPilotApi.listJobs();
      setJobLibrary(asArray(response.jobs));
      setError("");
    } catch (requestError) {
      setError(requestError.message || "读取岗位库失败。");
    } finally {
      setLibraryLoading(false);
    }
  }

  async function uploadJds(event) {
    const files = Array.from(event.target.files || []);
    const records = await Promise.all(
      files.map(async (file) => ({ filename: file.name, text: await file.text() }))
    );
    setUploadedJds(records);
    event.target.value = "";
  }

  function jdPayload(forceManual = false) {
    if (sourceMode === "library" && !forceManual) return { jd_texts: [], jd_filenames: [] };
    return {
      jd_texts: [...uploadedJds.map((item) => item.text), ...manualJds],
      jd_filenames: [
        ...uploadedJds.map((item) => item.filename),
        ...manualJds.map((_, index) => `manual_job_${index + 1}.txt`),
      ],
    };
  }

  async function saveJobs() {
    const payload = jdPayload(true);
    if (!payload.jd_texts.length) {
      setError("请先粘贴或上传至少一条 JD。");
      return;
    }
    setJobMutation("save");
    try {
      const response = await jobPilotApi.saveJobs({ ...payload, source: "frontend" });
      const organizedTexts = asArray(response.jobs)
        .map((job) => job.organized_text)
        .filter(Boolean);
      if (!uploadedJds.length && organizedTexts.length) {
        setJdInput(organizedTexts.join("\n\n---JOB---\n\n"));
      }
      setMessage(`已整理并保存 ${response.saved_count} 条岗位，同时刷新检索索引。`);
      setError("");
      await refreshJobLibrary();
    } catch (requestError) {
      setError(requestError.message || "保存岗位失败。");
    } finally {
      setJobMutation("");
    }
  }

  async function extractProfileDocument(file) {
    setProfileDocumentLoading(true);
    try {
      const response = await jobPilotApi.extractProfileDocument(file, targetRole);
      setProfileInput(JSON.stringify(response.candidate_profile, null, 2));
      const warningText = asArray(response.warnings).join(" ");
      setMessage(
        `已解析 ${response.filename}（${response.extraction_method}，${response.line_count} 行）并生成候选人画像。${warningText}`
      );
      setError("");
    } catch (requestError) {
      setError(requestError.message || "简历文档解析失败。");
      setMessage("");
    } finally {
      setProfileDocumentLoading(false);
    }
  }

  async function deleteJob(job) {
    if (!window.confirm(`确定停用“${job.title || "该岗位"}”吗？停用后将不再参与匹配。`)) return;
    setJobMutation(job.job_id);
    try {
      await jobPilotApi.deleteJob(job.job_id);
      setMessage(`已停用岗位：${job.title || job.job_id}`);
      setError("");
      await refreshJobLibrary();
    } catch (requestError) {
      setError(requestError.message || "停用岗位失败。");
    } finally {
      setJobMutation("");
    }
  }

  function exitAdminMode() {
    jobPilotApi.logout();
    setAuthRole("user");
    setMessage("已退出管理员模式。");
  }

  async function runAgent() {
    if (!canRun) {
      setError("请先准备候选人画像和至少一条岗位。");
      return;
    }
    const jobs = jdPayload();
    setError("");
    setMessage("");
    await run({
      target_role: targetRole,
      ...parseProfileInput(profileInput),
      ...jobs,
      use_job_library: sourceMode === "library",
      retrieval_top_k: Number(retrievalTopK) || 20,
      use_llm_rerank: useLlmRerank,
      use_llm_match_scoring: useLlmMatch,
      deep_analysis: deepAnalysis,
      allow_cache: true,
    });
  }

  return (
    <main className="app-shell">
      {adminDialogOpen && (
        <LoginPanel
          onClose={() => setAdminDialogOpen(false)}
          onLogin={(role) => {
            setAuthRole(role);
            setAdminDialogOpen(false);
            setMessage("已进入管理员模式。");
          }}
        />
      )}
      <header className="topbar">
        <div>
          <h1>JobPilot Agent 工作台</h1>
          <p>基于 FastAPI、React、LangGraph 与 Hybrid RAG 的岗位匹配系统。</p>
        </div>
        <div className="topbar-actions">
          <StatusPill health={health} />
          {authRole === "admin" ? (
            <button className="admin-access-button active" onClick={exitAdminMode} type="button">
              <LogOut size={15} />退出管理员
            </button>
          ) : (
            <button className="admin-access-button" onClick={() => setAdminDialogOpen(true)} type="button">
              <LockKeyhole size={15} />管理员
            </button>
          )}
        </div>
      </header>

      <section className="workspace-grid">
        <aside className="input-panel">
          <div className="field">
            <label>目标岗位</label>
            <input disabled={loading} value={targetRole} onChange={(event) => setTargetRole(event.target.value)} />
          </div>

          <ProfileEditor
            disabled={loading}
            documentLoading={profileDocumentLoading}
            onChange={setProfileInput}
            onDocument={extractProfileDocument}
            onMessage={(text, isError = false) => {
              setError(isError ? text : "");
              setMessage(isError ? "" : text);
            }}
            sampleProfile={SAMPLE_PROFILE}
            status={profileStatus}
            targetRole={targetRole}
            value={profileInput}
          />

          <CollapsibleSection defaultOpen meta={`${jdCount} 条`} title="岗位来源">
            <JobSourceSelector disabled={loading} onChange={setSourceMode} value={sourceMode} />
            {sourceMode === "library" ? (
              <JobLibraryPanel
                isAdmin={authRole === "admin"}
                jobs={jobLibrary}
                loading={loading || libraryLoading || Boolean(jobMutation)}
                onDelete={deleteJob}
                onRefresh={refreshJobLibrary}
              />
            ) : (
              <>
                <div className="input-actions">
                  <button disabled={loading} onClick={() => { setJdInput(SAMPLE_JD); setUploadedJds([]); }} type="button">
                    <FileText size={14} />加载示例 JD
                  </button>
                  {(!health?.auth_enabled || authRole === "admin") && (
                    <button disabled={loading || Boolean(jobMutation)} onClick={saveJobs} type="button">
                      {jobMutation === "save" ? <Loader2 className="spin" size={14} /> : <Briefcase size={14} />}
                      {jobMutation === "save" ? "保存并更新索引中" : "保存当前 JD"}
                    </button>
                  )}
                </div>
                <div className="field">
                  <div className="field-head">
                    <label>岗位 JD</label>
                    <label className="icon-button" title="上传 JD 文件">
                      <Upload size={16} />
                      <input disabled={loading} type="file" accept=".txt,.md" multiple onChange={uploadJds} />
                    </label>
                  </div>
                  <textarea
                    className="jd-textarea"
                    disabled={loading}
                    placeholder="粘贴岗位 JD，多个岗位使用 ---JOB--- 分隔"
                    value={jdInput}
                    onChange={(event) => setJdInput(event.target.value)}
                  />
                  <div className="upload-list">
                    {uploadedJds.map((item) => <span key={item.filename}><FileJson size={13} />{item.filename}</span>)}
                  </div>
                </div>
              </>
            )}
          </CollapsibleSection>

          <CollapsibleSection defaultOpen={false} meta={`Top-K ${retrievalTopK}`} title="运行设置">
            <div className="settings-grid">
              <label>
                <span>召回 Top-K</span>
                <input
                  disabled={loading}
                  max="50"
                  min="1"
                  type="number"
                  value={retrievalTopK}
                  onChange={(event) => setRetrievalTopK(event.target.value)}
                />
              </label>
              <label className="toggle"><input checked={useLlmRerank} disabled={loading} type="checkbox" onChange={(event) => setUseLlmRerank(event.target.checked)} /><span>LLM 重排</span></label>
              <label className="toggle"><input checked={useLlmMatch} disabled={loading} type="checkbox" onChange={(event) => setUseLlmMatch(event.target.checked)} /><span>LLM 解释</span></label>
              <label className="toggle"><input checked={deepAnalysis} disabled={loading} type="checkbox" onChange={(event) => setDeepAnalysis(event.target.checked)} /><span>深度分析 Top3</span></label>
            </div>
            <div className="settings-help">
              <div><strong>默认低成本路径</strong><span>Hybrid Top20 → 规则重排 Top10 → 规则评分；LLM 功能按需开启。</span></div>
              <div><strong>LLM 重排</strong><span>仅处理规则重排后的 Top5，避免把全部岗位发送给模型。</span></div>
            </div>
          </CollapsibleSection>

          <div className="input-messages">
            {(error || runError) && <div className="error-box"><AlertTriangle size={16} /><span>{error || runError}</span></div>}
            {message && <div className="save-box"><CheckCircle2 size={16} /><span>{message}</span></div>}
          </div>

          <div className="run-dock">
            <div>
              <span>{loading ? `任务状态：${status}` : sourceMode === "library" ? "岗位库" : "手动 JD"}</span>
              <strong title={runId}>{loading ? runId.slice(0, 12) : `${jdCount} 条岗位`}</strong>
            </div>
            {loading && <button className="cancel-button" onClick={cancel} type="button">取消</button>}
            <button className="run-button" disabled={loading || !canRun} onClick={runAgent} type="button">
              {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              {loading ? "运行中" : "运行 Agent"}
            </button>
          </div>
        </aside>

        <ResultWorkspace durationMs={durationMs} liveTrace={events} loading={loading} result={result} />
      </section>
    </main>
  );
}
