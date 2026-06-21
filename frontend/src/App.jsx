import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Briefcase,
  ChevronDown,
  CheckCircle2,
  ClipboardCopy,
  Database,
  Download,
  FileJson,
  FileText,
  Loader2,
  Play,
  RefreshCcw,
  Route,
  Target,
  Trash2,
  Upload,
  Wand2,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const SAMPLE_PROFILE = {
  name: "Alex Chen",
  education: ["Computer Science undergraduate candidate"],
  skills: ["Python", "RAG", "LLM", "Prompt Engineering", "API integration", "Pytest"],
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
  preferences: { location: "Remote or Shanghai" },
};

const SAMPLE_JD_TEXT = `Title: 大模型agent实习生
Company: 小红书
Location: 北京
Salary: 300-450元/天
Duration: 4天/周，3个月
Education: 本科
Responsibilities:
- 提升模型搜索意图判断能力，将复杂需求拆解为可执行的搜索子目标，生成 Query，选择检索源，控制搜索预算。
- 设计可增长、可替换、可分层的 context 架构，例如对话摘要层、搜索证据层、推理草稿层、最终引用层。
- 做检索后处理与引用可信度提升，包括 Snippet 选优、去重、冲突检测与引用自检。
- 降低模型幻觉，提高 evidence-answer 一致性。
Requirements:
- LLM 检索
- RAG
- 多轮对话
- 搜索策略
- Context Engineering
Preferred:
- Query Generation
- Evidence Answer
- Snippet Ranking
- Hallucination Reduction`;

const PIPELINE_NODES = [
  ["profile_node", "候选人画像"],
  ["jd_parse_node", "JD 解析"],
  ["retrieve_node", "混合召回"],
  ["rerank_node", "岗位重排"],
  ["match_score_node", "匹配评分"],
  ["gap_analysis_node", "差距分析"],
  ["resume_suggestion_node", "简历建议"],
];

const SCORE_DIMENSIONS = [
  { key: "requirements", label: "任职要求匹配", max: 70 },
  { key: "bonus", label: "加分项匹配", max: 20 },
  { key: "responsibilities", label: "岗位职责相关性", max: 10 },
];

const GAP_TYPE_LABELS = {
  missing_skill: "缺失技能",
  weak_project_evidence: "项目证据偏弱",
  no_quantification: "缺少量化结果",
  low_keyword_match: "关键词覆盖不足",
  missing_experience: "经历缺口",
};

const SEVERITY_LABELS = {
  high: "高",
  medium: "中",
  low: "低",
};

const TRACE_STATUS_LABELS = {
  success: "成功",
  error: "失败",
  running: "运行中",
  start: "开始",
  end: "完成",
  pending: "等待中",
};

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function safeScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(100, number)) : 0;
}

function scoreTone(score) {
  const value = safeScore(score);
  if (value >= 80) return "excellent";
  if (value >= 60) return "good";
  if (value >= 40) return "fair";
  return "poor";
}

function matchLevel(score) {
  const value = safeScore(score);
  if (value >= 80) return "强匹配";
  if (value >= 60) return "较匹配";
  if (value >= 40) return "可投但需优化";
  return "不建议优先投递";
}

function parseProfileInput(text) {
  const trimmed = text.trim();
  if (!trimmed) return {};
  try {
    return { user_profile_json: JSON.parse(trimmed) };
  } catch {
    return { user_profile_text: trimmed };
  }
}

function uniqueList(items) {
  return [...new Set(items.map((item) => String(item).trim()).filter(Boolean))];
}

function extractByLabel(text, labels) {
  const escapedLabels = labels.map((label) => label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|");
  const pattern = new RegExp(`(?:^|\\n)\\s*(?:${escapedLabels})\\s*[:：]\\s*([^\\n]+)`, "i");
  const match = text.match(pattern);
  return match?.[1]?.trim() || null;
}

function extractSkillsFromText(text) {
  const skillPatterns = [
    ["Python", /\bpython\b|熟练掌握\s*Python/i],
    ["JavaScript", /\bjavascript\b|\bjs\b/i],
    ["TypeScript", /\btypescript\b|\bts\b/i],
    ["React", /\breact\b/i],
    ["FastAPI", /\bfastapi\b/i],
    ["LangGraph", /\blanggraph\b/i],
    ["LangChain", /\blangchain\b/i],
    ["LlamaIndex", /\bllamaindex\b/i],
    ["RAG", /\brag\b|检索增强生成/i],
    ["LLM", /\bllm\b|大模型|大语言模型/i],
    ["AI Agent", /\bai agent\b|\bagent\b|智能体/i],
    ["DeepSeek API", /\bdeepseek\b/i],
    ["OpenAI SDK", /\bopenai\b/i],
    ["ChromaDB", /\bchromadb\b|向量库|Vector DB|Vector Database/i],
    ["Pydantic", /\bpydantic\b/i],
    ["Prompt Engineering", /prompt engineering|提示词|Prompt/i],
    ["Tool Calling", /tool calling|工具调用/i],
    ["PyTorch", /\bpytorch\b/i],
    ["TensorFlow", /\btensorflow\b/i],
    ["PaddlePaddle", /\bpaddlepaddle\b/i],
    ["pytest", /\bpytest\b/i],
    ["scikit-learn", /scikit-learn|sklearn/i],
    ["Docker", /\bdocker\b/i],
    ["Git", /\bgit\b|GitHub/i],
    ["SQL", /\bsql\b/i],
  ];

  return uniqueList(skillPatterns.filter(([, pattern]) => pattern.test(text)).map(([skill]) => skill));
}

function extractListSection(text, labels) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const labelPattern = new RegExp(`^(${labels.join("|")})\\s*[:：]?`, "i");
  const items = [];
  let collecting = false;

  for (const line of lines) {
    if (labelPattern.test(line)) {
      collecting = true;
      const inlineValue = line.replace(labelPattern, "").replace(/^[-*]\s*/, "").trim();
      if (inlineValue) items.push(inlineValue);
      continue;
    }
    if (collecting && /^[\u4e00-\u9fa5A-Za-z ]{2,10}[:：]/.test(line) && !/^[-*]/.test(line)) {
      collecting = false;
    }
    if (collecting) {
      items.push(line.replace(/^[-*]\s*/, "").trim());
    }
  }

  return uniqueList(items);
}

function extractProjectsFromText(text, skills) {
  const projectLines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /项目|project/i.test(line));

  return projectLines.slice(0, 5).map((line, index) => {
    const cleaned = line.replace(/^[-*]\s*/, "").replace(/^项目经历?\s*[:：]?/i, "").trim();
    const name = cleaned.split(/[，,。.;；]/)[0]?.trim() || `项目经历 ${index + 1}`;
    const projectSkills = skills.filter((skill) => new RegExp(skill.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i").test(line));
    return {
      name,
      description: cleaned || line,
      tech_stack: projectSkills,
      highlights: [],
    };
  });
}

function buildProfileJsonFromText(text, targetRole) {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error("请先输入候选人文本或 JSON。");
  }

  try {
    const parsed = JSON.parse(trimmed);
    return {
      name: parsed.name ?? null,
      education: asArray(parsed.education),
      skills: asArray(parsed.skills),
      projects: asArray(parsed.projects),
      internships: asArray(parsed.internships),
      target_roles: asArray(parsed.target_roles),
      preferences: parsed.preferences && typeof parsed.preferences === "object" && !Array.isArray(parsed.preferences) ? parsed.preferences : {},
    };
  } catch {
    const skills = extractSkillsFromText(trimmed);
    const education = uniqueList([
      ...extractListSection(trimmed, ["教育经历", "教育背景", "Education"]),
      ...trimmed
        .split(/[。\n]/)
        .map((item) => item.trim())
        .filter((item) => /本科|硕士|博士|计算机|软件工程|人工智能|Computer Science|Software Engineering/i.test(item))
        .slice(0, 3),
    ]);
    const internships = uniqueList(extractListSection(trimmed, ["实习经历", "工作经历", "Internships", "Experience"]));

    return {
      name: extractByLabel(trimmed, ["姓名", "Name"]) || null,
      education,
      skills,
      projects: extractProjectsFromText(trimmed, skills),
      internships,
      target_roles: targetRole ? [targetRole] : [],
      preferences: {
        source: "frontend_text_to_profile_json",
        raw_text: trimmed,
      },
    };
  }
}

function validateProfileJson(text) {
  if (!text.trim()) return { status: "empty", label: "未填写" };
  try {
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { status: "invalid", label: "JSON 需要是对象" };
    }
    return { status: "valid", label: "JSON 有效" };
  } catch {
    return { status: "text", label: "文本模式" };
  }
}

function splitJdText(text) {
  return text
    .split(/\n-{3,}\s*JOB\s*-{3,}\n/i)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseScoreBreakdown(reason = "") {
  const current = reason.match(/任职要求=([\d.]+)\/70[，,]\s*加分项=([\d.]+)\/20[，,]\s*岗位职责=([\d.]+)\/10/);
  if (current) {
    return {
      requirements: Number(current[1]),
      bonus: Number(current[2]),
      responsibilities: Number(current[3]),
    };
  }

  const legacy = [
    /技能=([\d.]+)\/40[，,]\s*项目=([\d.]+)\/25[，,]\s*经历=([\d.]+)\/15[，,]\s*关键词=([\d.]+)\/10[，,]\s*偏好=([\d.]+)\/10/,
    /skills=([\d.]+)\/40.*projects=([\d.]+)\/25.*experience=([\d.]+)\/15.*keywords=([\d.]+)\/10.*preferences=([\d.]+)\/10/i,
  ]
    .map((pattern) => reason.match(pattern))
    .find(Boolean);
  if (!legacy) return null;

  return {
    requirements: Number(legacy[1]) + Number(legacy[3]) + Number(legacy[5]),
    bonus: Number(legacy[2]),
    responsibilities: Number(legacy[4]),
  };
}

function formatReasonText(reason = "") {
  if (!reason) return "暂无推荐理由。";
  if (reason.includes("任职要求=") && reason.includes("加分项=") && reason.includes("岗位职责=")) {
    return reason;
  }

  const legacy = reason.match(
    /技能=([\d.]+)\/40[，,]\s*项目=([\d.]+)\/25[，,]\s*经历=([\d.]+)\/15[，,]\s*关键词=([\d.]+)\/10[，,]\s*偏好=([\d.]+)\/10/
  );
  if (!legacy) return reason;

  const requirements = Number(legacy[1]) + Number(legacy[3]) + Number(legacy[5]);
  const bonus = Number(legacy[2]);
  const responsibilities = Number(legacy[4]);
  return [
    "旧版评分结果已按新版结构换算展示：",
    `任职要求=${requirements.toFixed(2)}/70，`,
    `加分项=${bonus.toFixed(2)}/20，`,
    `岗位职责=${responsibilities.toFixed(2)}/10。`,
    "请重新运行 Agent 获取基于新版评分逻辑生成的完整推荐理由。",
  ].join("");
}

function formatDuration(ms) {
  if (!ms) return "尚未运行";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTime(timestamp) {
  if (!timestamp) return "无时间戳";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return String(timestamp);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function firstResumeSuggestion(result) {
  const firstGroup = asArray(result?.resume_suggestions)[0];
  return asArray(firstGroup?.suggestions)[0];
}

function summarizeProfileInput(text) {
  if (!text.trim()) {
    return { mode: "未填写", skillCount: 0, projectCount: 0, educationCount: 0 };
  }
  try {
    const parsed = JSON.parse(text);
    return {
      mode: "JSON 画像",
      skillCount: asArray(parsed.skills).length,
      projectCount: asArray(parsed.projects).length,
      educationCount: asArray(parsed.education).length,
    };
  } catch {
    return { mode: "文本画像", skillCount: 0, projectCount: 0, educationCount: 0 };
  }
}

function statusLabel(status) {
  return TRACE_STATUS_LABELS[status] || status || "未知";
}

function getPipelineStepStatus(node, trace, loading) {
  const records = asArray(trace).filter((record) => record.node === node);
  if (records.some((record) => record.status === "error" || record.event_type === "error")) return "error";
  if (records.some((record) => ["success", "end"].includes(record.status || record.event_type))) return "success";

  if (loading) {
    const firstPendingNode = PIPELINE_NODES.find(([candidate]) => !getPipelineStepStatus(candidate, trace, false).match(/success|error/))?.[0];
    return node === firstPendingNode ? "running" : "pending";
  }
  return "pending";
}

function retrieveSourceLabel(source) {
  if (source === "both") return "向量 + 关键词";
  if (source === "vector") return "向量召回";
  if (source === "keyword") return "关键词召回";
  return source || "未标注";
}

function StatusPill({ health }) {
  const apiAvailable = Boolean(health?.api_available);
  return (
    <span className={`status-pill ${apiAvailable ? "ok" : "fallback"}`}>
      {apiAvailable ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
      {apiAvailable ? "DeepSeek 已启用" : "规则兜底"}
    </span>
  );
}

function EmptyState({ icon: Icon, title, description }) {
  return (
    <div className="empty-state">
      <Icon size={26} />
      <strong>{title}</strong>
      {description && <span>{description}</span>}
    </div>
  );
}

function LoadingState() {
  return (
    <div className="loading-state">
      <Loader2 className="spin" size={26} />
      <div>
        <strong>Agent 正在执行</strong>
        <span>正在按 LangGraph 节点顺序完成画像、解析、召回、重排、评分和生成。</span>
      </div>
      <div className="pipeline-list">
        {PIPELINE_NODES.map(([node, label], index) => (
          <div className={`pipeline-step ${index === 0 ? "running" : ""}`} key={node}>
            <span />
            <strong>{label}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function CollapsibleSection({ children, defaultOpen = true, meta, title }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`input-section ${open ? "open" : "closed"}`}>
      <button className="section-trigger" onClick={() => setOpen((value) => !value)} type="button">
        <span>{title}</span>
        {meta && <em>{meta}</em>}
        <ChevronDown className={open ? "open" : ""} size={17} />
      </button>
      {open && <div className="section-body">{children}</div>}
    </section>
  );
}

function InputReadiness({ jdCount, profileStatus, profileSummary, targetRole, uploadedCount }) {
  const ready = profileStatus.status !== "empty" && jdCount > 0;
  const items = [
    { icon: Target, label: "目标岗位", value: targetRole || "未填写" },
    { icon: FileText, label: "画像模式", value: profileSummary.mode },
    { icon: Briefcase, label: "JD 数量", value: `${jdCount} 条` },
    { icon: Upload, label: "上传文件", value: `${uploadedCount} 个` },
  ];

  return (
    <section className={`readiness-card ${ready ? "ready" : "waiting"}`}>
      <div className="readiness-head">
        <strong>{ready ? "输入已就绪" : "等待补充输入"}</strong>
        <span>{ready ? "可以运行 Agent" : "至少需要候选人画像和 1 条 JD"}</span>
      </div>
      <div className="readiness-grid">
        {items.map(({ icon: Icon, label, value }) => (
          <div key={label}>
            <Icon size={15} />
            <span>{label}</span>
            <strong title={value}>{value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function JobSourceSelector({ disabled, onChange, value }) {
  return (
    <div className="source-segment" role="group" aria-label="岗位来源">
      <button className={value === "library" ? "active" : ""} disabled={disabled} onClick={() => onChange("library")} type="button">
        <Database size={15} />
        使用岗位库
      </button>
      <button className={value === "manual" ? "active" : ""} disabled={disabled} onClick={() => onChange("manual")} type="button">
        <FileText size={15} />
        手动输入 JD
      </button>
    </div>
  );
}

function JobLibraryPanel({ jobs, loading, onDelete, onRefresh }) {
  const activeJobs = asArray(jobs);
  return (
    <section className="job-library-panel">
      <div className="library-head">
        <div>
          <strong>岗位库</strong>
          <span>SQLite 永久保存，删除后不再参与岗位库匹配</span>
        </div>
        <button disabled={loading} onClick={onRefresh} type="button">
          <RefreshCcw size={14} />
          刷新
        </button>
      </div>
      <div className="library-list">
        {activeJobs.length ? (
          activeJobs.slice(0, 6).map((job) => (
            <div className="library-item" key={job.job_id}>
              <div>
                <strong>{job.title || "未命名岗位"}</strong>
                <span>
                  {job.company || "未知公司"}
                  {job.location ? ` · ${job.location}` : ""}
                </span>
                <small>
                  {job.salary || "薪资未标注"}
                  {job.education ? ` · ${job.education}` : ""}
                </small>
              </div>
              <button disabled={loading} onClick={() => onDelete(job.job_id)} title="从岗位库删除" type="button">
                <Trash2 size={14} />
              </button>
            </div>
          ))
        ) : (
          <div className="library-empty">暂无已保存岗位。粘贴 JD 后点击“保存 JD 到岗位库”。</div>
        )}
      </div>
    </section>
  );
}

function PipelineOverview({ jobs, loading, trace }) {
  const records = asArray(trace);
  const matchedJobs = asArray(jobs);
  const completedCount = PIPELINE_NODES.filter(([node]) => getPipelineStepStatus(node, records, loading) === "success").length;
  const errorCount = records.filter((record) => record.status === "error" || record.event_type === "error").length;
  const topScore = matchedJobs.length ? Math.max(...matchedJobs.map((job) => safeScore(job.match_score))).toFixed(1) : "0.0";

  return (
    <section className="workflow-overview">
      <div className="overview-summary">
        <div>
          <Route size={17} />
          <span>Agent 流程</span>
          <strong>
            {completedCount}/{PIPELINE_NODES.length} 节点完成
          </strong>
        </div>
        <div>
          <Database size={17} />
          <span>推荐结果</span>
          <strong>{matchedJobs.length} 个岗位</strong>
        </div>
        <div>
          <BarChart3 size={17} />
          <span>最高分</span>
          <strong>{topScore}/100</strong>
        </div>
        <div>
          <AlertTriangle size={17} />
          <span>异常事件</span>
          <strong>{errorCount} 个</strong>
        </div>
      </div>
      <div className="workflow-steps">
        {PIPELINE_NODES.map(([node, label], index) => {
          const status = getPipelineStepStatus(node, records, loading);
          return (
            <div className={`workflow-step ${status}`} key={node}>
              <span className="step-index">{index + 1}</span>
              <div>
                <strong>{label}</strong>
                <small>{statusLabel(status)}</small>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function SkillList({ items, tone = "neutral" }) {
  const list = asArray(items);
  if (!list.length) return <span className="muted">暂无</span>;
  return (
    <div className="chip-row">
      {list.map((item) => (
        <span className={`chip ${tone}`} key={item}>
          {item}
        </span>
      ))}
    </div>
  );
}

function ScoreBar({ score }) {
  const value = safeScore(score);
  return (
    <div className="score-bar" aria-label={`匹配分 ${value}`}>
      <div className={`score-fill ${scoreTone(value)}`} style={{ width: `${value}%` }} />
    </div>
  );
}

function ScoreBreakdown({ job }) {
  const parsed = parseScoreBreakdown(job?.reason);
  const fallbackTotal = safeScore(job?.match_score);
  return (
    <div className="score-breakdown">
      {SCORE_DIMENSIONS.map((dimension) => {
        const raw = parsed ? Number(parsed[dimension.key]) : (fallbackTotal / 100) * dimension.max;
        const value = Number.isFinite(raw) ? Math.max(0, Math.min(dimension.max, raw)) : 0;
        const percent = dimension.max ? (value / dimension.max) * 100 : 0;
        return (
          <div className="score-row" key={dimension.key}>
            <div className="score-row-head">
              <span>{dimension.label}</span>
              <strong>
                {value.toFixed(value % 1 === 0 ? 0 : 1)}/{dimension.max}
              </strong>
            </div>
            <div className="progress-track">
              <div className={`progress-fill ${scoreTone(percent)}`} style={{ width: `${percent}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function JobMatchCard({ job, index }) {
  const score = safeScore(job.match_score);
  const matchedProjects = asArray(job.matched_projects);
  return (
    <article className="job-card">
      <div className="job-card-head">
        <div className="job-title-block">
          <div className="rank">#{index + 1}</div>
          <h3>{job.title || "未命名岗位"}</h3>
          <p>{job.company || "未知公司"}</p>
          <div className="job-badges">
            {job.job_id && <span>{job.job_id}</span>}
            {job.retrieve_source && <span>{retrieveSourceLabel(job.retrieve_source)}</span>}
            {job.rerank_score !== undefined && <span>重排分 {safeScore(job.rerank_score).toFixed(1)}</span>}
          </div>
        </div>
        <div className={`score-box ${scoreTone(score)}`}>
          <strong>{score.toFixed(1)}</strong>
          <span>/100</span>
          <em>{matchLevel(score)}</em>
        </div>
      </div>
      <ScoreBar score={score} />
      <div className="job-meta-grid">
        <div>
          <label>已匹配依据</label>
          <SkillList items={job.skill_overlap} tone="positive" />
        </div>
        <div>
          <label>待补强要求</label>
          <SkillList items={job.missing_skills} tone="warning" />
        </div>
      </div>
      <ScoreBreakdown job={job} />
      {matchedProjects.length > 0 && (
        <div className="matched-projects">
          <label>匹配项目证据</label>
          <SkillList items={matchedProjects} tone="evidence" />
        </div>
      )}
      <div className="reason-block">
        <label>推荐理由</label>
        <p>{formatReasonText(job.reason)}</p>
      </div>
      <div className="recommendation">
        <strong>下一步建议</strong>
        <span>{job.recommendation || "暂无建议。"}</span>
      </div>
    </article>
  );
}

function JobsView({ jobs, loading }) {
  const list = asArray(jobs);
  if (loading) return <LoadingState />;
  if (!list.length) {
    return (
      <div className="empty-state start-empty">
        <Briefcase size={26} />
        <strong>尚未运行 Agent</strong>
        <span>开始前确认候选人画像和岗位来源，然后点击左侧底部的运行按钮。</span>
        <div className="start-steps">
          <span>1. 整理候选人画像</span>
          <span>2. 选择岗位库或手动 JD</span>
        </div>
      </div>
    );
  }
  return (
    <div className="job-list">
      {list.map((job, index) => (
        <JobMatchCard job={job} index={index} key={job.job_id || index} />
      ))}
    </div>
  );
}

function GapsView({ gaps, loading }) {
  const list = asArray(gaps);
  if (loading) return <LoadingState />;
  if (!list.length) return <EmptyState icon={AlertTriangle} title="暂无差距分析" description="运行后会展示 Top 岗位对应的技能和项目证据差距。" />;
  return (
    <div className="stack">
      {list.map((group) => (
        <section className="plain-section" key={group.job_id}>
          <h3>{group.job_id}</h3>
          {asArray(group.gaps).map((gap, index) => (
            <div className="line-item" key={`${group.job_id}-${index}`}>
              <div>
                <strong>{GAP_TYPE_LABELS[gap.type] || gap.type}</strong>
                <span className={`severity ${gap.severity}`}>{SEVERITY_LABELS[gap.severity] || gap.severity}</span>
              </div>
              <p>{gap.description}</p>
              <small>{gap.suggestion}</small>
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}

function ResumeView({ suggestions, loading }) {
  const list = asArray(suggestions);
  if (loading) return <LoadingState />;
  if (!list.length) return <EmptyState icon={FileText} title="暂无简历建议" description="运行后会根据差距分析生成可落地的简历优化建议。" />;
  return (
    <div className="stack">
      {list.map((group) => (
        <section className="plain-section" key={group.job_id}>
          <h3>{group.job_id}</h3>
          {asArray(group.suggestions).map((item, index) => (
            <div className="line-item" key={`${group.job_id}-${index}`}>
              <div>
                <strong>{item.section}</strong>
              </div>
              <p>{item.original_problem}</p>
              <small>{item.suggestion}</small>
              <pre>{item.improved_example}</pre>
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}

function TraceTimeline({ trace, loading }) {
  const records = asArray(trace);
  if (loading && !records.length) return <LoadingState />;
  if (!records.length) return <EmptyState icon={Activity} title="暂无执行轨迹" description="运行后会按节点展示 LangGraph 执行时间线。" />;

  return (
    <div className="trace-timeline">
      {records.map((record, index) => {
        const status = record.status || record.event_type || "success";
        return (
          <article className={`trace-event ${status}`} key={`${record.node}-${index}`}>
            <div className="trace-dot" />
            <div className="trace-content">
              <div className="trace-head">
                <div>
                  <strong>{PIPELINE_NODES.find(([node]) => node === record.node)?.[1] || record.node}</strong>
                  <code>{record.node}</code>
                </div>
                <span className={`trace-status ${status}`}>{status}</span>
              </div>
              <div className="trace-meta">
                <span>时间：{formatTime(record.timestamp)}</span>
                <span>输入：{record.input_count ?? 0}</span>
                <span>输出：{record.output_count ?? 0}</span>
              </div>
              <p>{record.message || "节点已完成。"}</p>
              {record.error_message && <pre className="trace-error">{record.error_message}</pre>}
              {(record.query || record.merged_count !== undefined || record.final_retrieved_count !== undefined) && (
                <div className="trace-detail">
                  {record.query && <span>查询：{record.query}</span>}
                  {record.merged_count !== undefined && <span>合并数：{record.merged_count}</span>}
                  {record.final_retrieved_count !== undefined && <span>最终召回：{record.final_retrieved_count}</span>}
                </div>
              )}
            </div>
          </article>
        );
      })}
    </div>
  );
}

function ReportActions({ result, onCopied }) {
  const markdown = result?.final_report_markdown || "";

  function downloadMarkdown() {
    if (!markdown) return;
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `jobpilot-report-${result?.run_id || "latest"}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function copyText(text, label) {
    if (!text) return;
    await navigator.clipboard.writeText(text);
    onCopied(`${label}已复制`);
  }

  const resume = firstResumeSuggestion(result);
  const resumeText = resume
    ? [`模块：${resume.section}`, `问题：${resume.original_problem}`, `建议：${resume.suggestion}`, `示例：${resume.improved_example}`].join("\n")
    : "";

  return (
    <div className="report-actions">
      <button disabled={!markdown} onClick={downloadMarkdown} type="button">
        <Download size={15} />
        导出 Markdown 报告
      </button>
      <button disabled={!resumeText} onClick={() => copyText(resumeText, "Top1 简历建议")} type="button">
        <ClipboardCopy size={15} />
        复制 Top1 简历建议
      </button>
    </div>
  );
}

function ReportView({ markdown, loading }) {
  if (loading) return <LoadingState />;
  if (!markdown) return <EmptyState icon={FileText} title="暂无报告" description="运行后可在这里预览并导出 Markdown 报告。" />;
  return <pre className="report-panel">{markdown}</pre>;
}

function StatsBar({ result, runDurationMs }) {
  const jobs = asArray(result?.matched_jobs);
  const trace = asArray(result?.trace);
  const scores = jobs.map((job) => safeScore(job.match_score));
  const average = scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : 0;
  const missingSkills = new Set(jobs.flatMap((job) => asArray(job.missing_skills)));
  const stats = [
    ["匹配岗位数量", jobs.length || "0"],
    ["平均匹配分", scores.length ? average.toFixed(1) : "0.0"],
    ["Top1 匹配分", scores.length ? Math.max(...scores).toFixed(1) : "0.0"],
    ["缺失技能总数", missingSkills.size || "0"],
    ["执行事件数", trace.length || "0"],
    ["运行耗时", formatDuration(runDurationMs)],
    ["运行 ID", result?.run_id || "尚未运行"],
  ];
  return (
    <div className="stats-bar">
      {stats.map(([label, value]) => (
        <div className="stat-card" key={label}>
          <span>{label}</span>
          <strong title={String(value)}>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function TabButton({ active, count, label, onClick }) {
  return (
    <button className={active ? "active" : ""} onClick={onClick} type="button">
      {label}
      {count !== null && <span>{count}</span>}
    </button>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [targetRole, setTargetRole] = useState("AI Agent Intern");
  const [profileInput, setProfileInput] = useState(JSON.stringify(SAMPLE_PROFILE, null, 2));
  const [jdInput, setJdInput] = useState("");
  const [uploadedJds, setUploadedJds] = useState([]);
  const [useJobLibrary, setUseJobLibrary] = useState(false);
  const [jobLibrary, setJobLibrary] = useState([]);
  const [jobLibraryLoading, setJobLibraryLoading] = useState(false);
  const [retrievalTopK, setRetrievalTopK] = useState(10);
  const [useLlmRerank, setUseLlmRerank] = useState(false);
  const [useLlmMatchScoring, setUseLlmMatchScoring] = useState(false);
  const [activeTab, setActiveTab] = useState("jobs");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [runDurationMs, setRunDurationMs] = useState(0);
  const [copyMessage, setCopyMessage] = useState("");
  const [saveMessage, setSaveMessage] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((response) => response.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "offline", api_available: false }));
  }, []);

  useEffect(() => {
    refreshJobLibrary();
  }, []);

  const profileStatus = useMemo(() => validateProfileJson(profileInput), [profileInput]);
  const profileSummary = useMemo(() => summarizeProfileInput(profileInput), [profileInput]);
  const manualJdCount = useMemo(() => uploadedJds.length + splitJdText(jdInput).length, [jdInput, uploadedJds]);
  const jdCount = useJobLibrary ? jobLibrary.length : manualJdCount;

  async function handleProfileFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    setProfileInput(text);
    event.target.value = "";
  }

  async function handleJdFiles(event) {
    const files = Array.from(event.target.files || []);
    const loaded = await Promise.all(
      files.map(async (file) => ({
        filename: file.name,
        text: await file.text(),
      }))
    );
    setUploadedJds(loaded);
    event.target.value = "";
  }

  function loadSampleProfile() {
    setProfileInput(JSON.stringify(SAMPLE_PROFILE, null, 2));
  }

  function loadSampleJds() {
    setJdInput(SAMPLE_JD_TEXT);
    setUploadedJds([]);
  }

  function clearInputs() {
    setProfileInput("");
    setJdInput("");
    setUploadedJds([]);
    setError("");
    setSaveMessage("");
  }

  function organizeProfileInput() {
    const trimmed = profileInput.trim();
    if (!trimmed) {
      setError("请先输入候选人文本或 JSON。");
      setSaveMessage("");
      return;
    }

    try {
      const parsed = JSON.parse(trimmed);
      setProfileInput(JSON.stringify(parsed, null, 2));
      setError("");
      setSaveMessage("已美化候选人 JSON。");
      return;
    } catch {
      // Continue below and treat the input as raw resume/profile text.
    }

    try {
      const profile = buildProfileJsonFromText(trimmed, targetRole);
      setProfileInput(JSON.stringify(profile, null, 2));
      setError("");
      setSaveMessage("已将文本整理为 CandidateProfile JSON。本地转换不会调用 API，复杂信息仍建议运行 Agent 进一步抽取。");
    } catch (err) {
      setError(err.message || "整理候选人画像失败。");
      setSaveMessage("");
    }
  }

  function currentJdPayload(forceManual = false) {
    if (useJobLibrary && !forceManual) {
      return { jd_texts: [], jd_filenames: [] };
    }

    const jdFromText = splitJdText(jdInput);
    return {
      jd_texts: [...uploadedJds.map((item) => item.text), ...jdFromText],
      jd_filenames: [
        ...uploadedJds.map((item) => item.filename),
        ...jdFromText.map((_, index) => `${targetRole || "job"}_${index + 1}.txt`),
      ],
    };
  }

  async function refreshJobLibrary() {
    setJobLibraryLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/jobs`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "读取岗位库失败。");
      setJobLibrary(asArray(data.jobs));
    } catch (err) {
      setError(err.message || "读取岗位库失败，请检查后端服务。");
    } finally {
      setJobLibraryLoading(false);
    }
  }

  async function deleteJobFromLibrary(jobId) {
    if (!jobId) return;
    setError("");
    setSaveMessage("");
    try {
      const response = await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(jobId)}`, {
        method: "DELETE",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "删除岗位失败。");
      setSaveMessage("岗位已从数据库中删除。");
      await refreshJobLibrary();
    } catch (err) {
      setError(err.message || "删除岗位失败，请检查后端服务。");
    }
  }

  async function saveJobsToLibrary() {
    setError("");
    setSaveMessage("");
    const payload = currentJdPayload(true);
    if (!payload.jd_texts.length) {
      setError("请先粘贴或上传至少一条岗位 JD，再保存到岗位库。");
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/api/record-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, source: "frontend" }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "保存岗位失败。");
      setSaveMessage(`已保存 ${data.saved_count} 条岗位到 SQLite 岗位库。`);
      await refreshJobLibrary();
    } catch (err) {
      setError(err.message || "保存岗位失败，请检查后端服务。");
    }
  }

  async function runAgent() {
    setLoading(true);
    setError("");
    setCopyMessage("");
    const start = performance.now();
    try {
      const jdPayload = currentJdPayload();
      const body = {
        target_role: targetRole,
        ...parseProfileInput(profileInput),
        jd_texts: jdPayload.jd_texts,
        jd_filenames: jdPayload.jd_filenames,
        use_job_library: useJobLibrary,
        retrieval_top_k: Number(retrievalTopK) || 10,
        use_llm_rerank: useLlmRerank,
        use_llm_match_scoring: useLlmMatchScoring,
      };

      const response = await fetch(`${API_BASE}/api/run-jobpilot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "JobPilot 运行失败。");

      setResult(data);
      setHealth((previous) => ({ ...(previous || {}), api_available: data.api_available }));
      setRunDurationMs(performance.now() - start);
      setActiveTab("jobs");
    } catch (err) {
      setError(err.message || "请求失败，请检查后端服务。");
      setRunDurationMs(performance.now() - start);
    } finally {
      setLoading(false);
    }
  }

  const tabs = [
    ["jobs", "岗位推荐", asArray(result?.matched_jobs).length],
    ["gaps", "差距分析", asArray(result?.gaps).length],
    ["resume", "简历建议", asArray(result?.resume_suggestions).length],
    ["trace", "执行轨迹", asArray(result?.trace).length],
    ["report", "报告", null],
  ];

  const tabContent = {
    jobs: <JobsView jobs={result?.matched_jobs} loading={loading} />,
    gaps: <GapsView gaps={result?.gaps} loading={loading} />,
    resume: <ResumeView suggestions={result?.resume_suggestions} loading={loading} />,
    trace: <TraceTimeline trace={result?.trace} loading={loading} />,
    report: <ReportView markdown={result?.final_report_markdown} loading={loading} />,
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>JobPilot Agent 工作台</h1>
          <p>基于 FastAPI + React 的 LangGraph 岗位匹配 Agent 交互界面。</p>
        </div>
        <StatusPill health={health} />
      </header>

      <section className="workspace-grid">
        <aside className="input-panel">
          <div className="field">
            <label>目标岗位</label>
            <input disabled={loading} value={targetRole} onChange={(event) => setTargetRole(event.target.value)} />
          </div>

          <InputReadiness
            jdCount={jdCount}
            profileStatus={profileStatus}
            profileSummary={profileSummary}
            targetRole={targetRole}
            uploadedCount={uploadedJds.length}
          />

          <CollapsibleSection defaultOpen meta={profileStatus.label} title="候选人画像">
            <div className="input-actions">
              <button disabled={loading} onClick={loadSampleProfile} type="button">
                <RefreshCcw size={14} />
                加载示例候选人
              </button>
              <button disabled={loading} onClick={organizeProfileInput} type="button">
                <Wand2 size={14} />
                整理画像
              </button>
            </div>

            <div className="field">
              <div className="field-head">
                <label>候选人画像</label>
                <div className="field-actions">
                  <span className={`validation-pill ${profileStatus.status}`}>{profileStatus.label}</span>
                  <label className="icon-button" title="上传候选人 JSON 或文本">
                    <Upload size={16} />
                    <input disabled={loading} type="file" accept=".json,.txt,.md" onChange={handleProfileFile} />
                  </label>
                </div>
              </div>
              <textarea disabled={loading} value={profileInput} onChange={(event) => setProfileInput(event.target.value)} />
            </div>
          </CollapsibleSection>

          <CollapsibleSection defaultOpen meta={useJobLibrary ? `岗位库 ${jobLibrary.length} 条` : `手动 ${manualJdCount} 条`} title="岗位来源">
            <JobSourceSelector
              disabled={loading}
              onChange={(mode) => setUseJobLibrary(mode === "library")}
              value={useJobLibrary ? "library" : "manual"}
            />

            {useJobLibrary ? (
              <JobLibraryPanel jobs={jobLibrary} loading={loading || jobLibraryLoading} onDelete={deleteJobFromLibrary} onRefresh={refreshJobLibrary} />
            ) : (
              <>
                <div className="input-actions">
                  <button disabled={loading} onClick={loadSampleJds} type="button">
                    <FileText size={14} />
                    加载示例 JD
                  </button>
                  <button disabled={loading} onClick={saveJobsToLibrary} type="button">
                    <Briefcase size={14} />
                    保存当前 JD
                  </button>
                </div>

                <div className="field">
                  <div className="field-head">
                    <label>岗位 JD</label>
                    <label className="icon-button" title="上传 JD 文本文件">
                      <Upload size={16} />
                      <input disabled={loading} type="file" accept=".txt,.md" multiple onChange={handleJdFiles} />
                    </label>
                  </div>
                  <textarea
                    className="jd-textarea"
                    disabled={loading}
                    value={jdInput}
                    onChange={(event) => setJdInput(event.target.value)}
                    placeholder="粘贴岗位 JD。多个 JD 可用 ---JOB--- 分隔"
                  />
                  <div className="upload-list">
                    {uploadedJds.map((item) => (
                      <span key={item.filename}>
                        <FileJson size={13} />
                        {item.filename}
                      </span>
                    ))}
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
                  type="number"
                  min="1"
                  max="50"
                  value={retrievalTopK}
                  onChange={(event) => setRetrievalTopK(event.target.value)}
                />
              </label>
              <label className="toggle">
                <input disabled={loading} type="checkbox" checked={useLlmRerank} onChange={(event) => setUseLlmRerank(event.target.checked)} />
                <span>LLM 重排</span>
              </label>
              <label className="toggle">
                <input
                  disabled={loading}
                  type="checkbox"
                  checked={useLlmMatchScoring}
                  onChange={(event) => setUseLlmMatchScoring(event.target.checked)}
                />
                <span>LLM 解释</span>
              </label>
            </div>
          </CollapsibleSection>

          <div className="input-messages">
            {error && (
              <div className="error-box">
                <AlertTriangle size={16} />
                <span>{error}</span>
              </div>
            )}
            {saveMessage && (
              <div className="save-box">
                <CheckCircle2 size={16} />
                <span>{saveMessage}</span>
              </div>
            )}
          </div>

          <div className="run-dock">
            <div>
              <span>{useJobLibrary ? "岗位库" : "手动 JD"}</span>
              <strong>{jdCount} 条岗位</strong>
            </div>
            <button className="run-button" onClick={runAgent} disabled={loading} type="button">
              {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              {loading ? "运行中" : "运行 Agent"}
            </button>
          </div>
        </aside>

        <section className="result-panel">
          <StatsBar result={result} runDurationMs={runDurationMs} />
          <PipelineOverview jobs={result?.matched_jobs} loading={loading} trace={result?.trace} />
          <nav className="tabs">
            {tabs.map(([key, label, count]) => (
              <TabButton active={activeTab === key} count={count} key={key} label={label} onClick={() => setActiveTab(key)} />
            ))}
          </nav>
          <div className="result-actions">
            <ReportActions result={result} onCopied={setCopyMessage} />
            {copyMessage && <span>{copyMessage}</span>}
          </div>
          <div className="tab-body">{tabContent[activeTab]}</div>
        </section>
      </section>
    </main>
  );
}
