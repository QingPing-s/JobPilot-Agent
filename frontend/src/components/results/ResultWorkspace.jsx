import {
  AlertTriangle,
  ClipboardCopy,
  Download,
  FileText,
  Loader2,
} from "lucide-react";
import { useMemo, useState } from "react";

import { PipelineOverview } from "../pipeline/PipelineOverview";
import {
  asArray,
  formatDuration,
  isTraceWarning,
  matchLevel,
  safeScore,
  scoreTone,
} from "../../utils/jobpilot";

const GAP_LABELS = {
  missing_skill: "缺失技能",
  weak_project_evidence: "项目证据偏弱",
  no_quantification: "缺少量化结果",
  low_keyword_match: "关键词覆盖不足",
  missing_experience: "经历不足",
};

function EmptyState({ title, description }) {
  return (
    <div className="empty-state">
      <FileText size={28} />
      <strong>{title}</strong>
      <span>{description}</span>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="loading-state">
      <Loader2 className="spin" size={24} />
      <strong>Agent 正在运行</strong>
      <span>节点事件会实时更新，完成后展示结果。</span>
    </div>
  );
}

function Chips({ items, tone = "neutral" }) {
  const values = asArray(items);
  if (!values.length) return <span className="muted">暂无</span>;
  return (
    <div className="chip-row">
      {values.map((item) => (
        <span className={`chip ${tone}`} key={String(item)}>
          {String(item)}
        </span>
      ))}
    </div>
  );
}

function ScoreBar({ score }) {
  const value = safeScore(score);
  return (
    <div className="score-bar">
      <span className={`score-fill ${scoreTone(value)}`} style={{ width: `${value}%` }} />
    </div>
  );
}

function scoreBreakdown(job) {
  const direct = job?.score_breakdown;
  if (direct && typeof direct === "object") {
    return [
      ["任职要求", Number(direct.requirements ?? direct.requirement_score ?? 0), 70],
      ["加分项", Number(direct.bonus ?? direct.bonus_score ?? 0), 20],
      ["岗位职责", Number(direct.responsibilities ?? direct.responsibility_score ?? 0), 10],
    ];
  }
  const reason = String(job?.reason || "");
  const patterns = [
    ["任职要求", /任职要求[=：]\s*([\d.]+)\/70/i, 70],
    ["加分项", /加分项[=：]\s*([\d.]+)\/20/i, 20],
    ["岗位职责", /岗位职责(?:相关性)?[=：]\s*([\d.]+)\/10/i, 10],
  ];
  return patterns.map(([label, pattern, max]) => {
    const match = reason.match(pattern);
    return [label, match ? Number(match[1]) : 0, max];
  });
}

function JobCard({ job, index }) {
  const score = safeScore(job.match_score);
  return (
    <article className="job-card">
      <div className="job-card-head">
        <div>
          <small>#{index + 1}</small>
          <h2>{job.title || "未命名岗位"}</h2>
          <p>{job.company || "公司未标注"}{job.location ? ` · ${job.location}` : ""}</p>
        </div>
        <div className={`score-number ${scoreTone(score)}`}>
          <strong>{score.toFixed(1)}</strong>
          <span>/100</span>
          <em>{matchLevel(score)}</em>
        </div>
      </div>
      <ScoreBar score={score} />
      <div className="job-meta-grid">
        <section>
          <strong>匹配证据</strong>
          <Chips items={job.skill_overlap} tone="positive" />
        </section>
        <section>
          <strong>待补要求</strong>
          <Chips items={job.missing_skills} tone="warning" />
        </section>
      </div>
      <div className="score-breakdown">
        {scoreBreakdown(job).map(([label, value, max]) => (
          <div key={label}>
            <span>{label}</span>
            <strong>{Number(value).toFixed(1)}/{max}</strong>
            <div className="progress-track">
              <span
                className={`progress-fill ${scoreTone((Number(value) / max) * 100)}`}
                style={{ width: `${Math.min(100, (Number(value) / max) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
      {!!asArray(job.matched_projects).length && (
        <section className="project-evidence">
          <strong>匹配项目证据</strong>
          <Chips items={job.matched_projects} tone="evidence" />
        </section>
      )}
      <div className="job-copy">
        <strong>推荐理由</strong>
        <p>{job.reason || "已根据任职要求、加分项和职责相关性完成规则评分。"}</p>
      </div>
      <div className="recommendation">
        <strong>下一步建议</strong>
        <p>{job.recommendation || "结合缺失要求补充项目证据后再投递。"}</p>
      </div>
    </article>
  );
}

function JobsView({ jobs, loading }) {
  if (loading) return <LoadingState />;
  const values = asArray(jobs);
  if (!values.length) {
    return <EmptyState title="尚未运行 Agent" description="开始前确认候选人画像和岗位来源，然后点击左侧底部的运行按钮。" />;
  }
  return <div className="job-list">{values.map((job, index) => <JobCard job={job} index={index} key={job.job_id || index} />)}</div>;
}

function GapsView({ gaps, loading }) {
  if (loading) return <LoadingState />;
  const values = asArray(gaps);
  if (!values.length) return <EmptyState title="暂无差距分析" description="高匹配岗位完成后会生成差距分析。" />;
  return (
    <div className="plain-list">
      {values.map((group, index) => (
        <section className="plain-section" key={group.job_id || index}>
          <h2>{group.title || group.job_id || `岗位 ${index + 1}`}</h2>
          {asArray(group.gaps).map((gap, gapIndex) => (
            <article className={`gap-item ${gap.severity || "medium"}`} key={`${gap.type}-${gapIndex}`}>
              <div>
                <strong>{GAP_LABELS[gap.type] || gap.type || "差距项"}</strong>
                <span>{gap.severity || "medium"}</span>
              </div>
              <p>{gap.description}</p>
              <small>{gap.suggestion}</small>
            </article>
          ))}
        </section>
      ))}
    </div>
  );
}

function ResumeView({ suggestions, loading }) {
  if (loading) return <LoadingState />;
  const values = asArray(suggestions);
  if (!values.length) return <EmptyState title="暂无简历建议" description="深度分析完成后会在这里给出可执行的改写建议。" />;
  return (
    <div className="plain-list">
      {values.map((group, index) => (
        <section className="plain-section" key={group.job_id || index}>
          <h2>{group.title || group.job_id || `岗位 ${index + 1}`}</h2>
          {asArray(group.suggestions).map((item, itemIndex) => (
            <article className="resume-item" key={`${item.section}-${itemIndex}`}>
              <strong>{item.section || "简历模块"}</strong>
              <p><b>问题：</b>{item.original_problem}</p>
              <p><b>建议：</b>{item.suggestion}</p>
              <pre>{item.improved_example}</pre>
            </article>
          ))}
        </section>
      ))}
    </div>
  );
}

function TraceView({ trace, loading }) {
  const values = asArray(trace);
  if (loading && !values.length) return <LoadingState />;
  if (!values.length) return <EmptyState title="暂无执行轨迹" description="节点开始执行后会在这里显示状态。" />;
  return (
    <div className="trace-list">
      {values.map((record, index) => {
        const warning = isTraceWarning(record);
        return (
          <article className="trace-event" key={`${record.node}-${record.timestamp}-${index}`}>
            <span className={`trace-status ${warning ? "warning" : "success"}`}>
              {warning ? <AlertTriangle size={14} /> : index + 1}
            </span>
            <div>
              <header>
                <strong>{record.node || "unknown_node"}</strong>
                <small>{record.duration_ms !== undefined ? `${record.duration_ms}ms` : ""}</small>
              </header>
              <p>{record.message || record.event_type || "节点事件"}</p>
              <div className="trace-detail">
                <span>输入 {record.input_count ?? 0}</span>
                <span>输出 {record.output_count ?? 0}</span>
                <span>LLM {record.llm_calls ?? 0} 次</span>
                <span>Token {record.total_tokens ?? 0}</span>
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function ReportView({ result, loading, onMessage }) {
  const markdown = result?.final_report_markdown || "";
  if (loading) return <LoadingState />;
  if (!markdown) return <EmptyState title="暂无报告" description="运行完成后可预览和导出 Markdown 报告。" />;

  function download() {
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `jobpilot-report-${result.run_id || "latest"}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function copySuggestion() {
    const first = asArray(result?.resume_suggestions)[0]?.suggestions?.[0];
    if (!first) return;
    await navigator.clipboard.writeText(
      [`模块：${first.section}`, `问题：${first.original_problem}`, `建议：${first.suggestion}`, `示例：${first.improved_example}`].join("\n")
    );
    onMessage("Top1 简历建议已复制");
  }

  return (
    <>
      <div className="report-actions">
        <button onClick={download} type="button"><Download size={15} />导出 Markdown 报告</button>
        <button disabled={!asArray(result?.resume_suggestions).length} onClick={copySuggestion} type="button">
          <ClipboardCopy size={15} />复制 Top1 简历建议
        </button>
      </div>
      <pre className="report-panel">{markdown}</pre>
    </>
  );
}

function Stats({ result, durationMs }) {
  const jobs = asArray(result?.matched_jobs);
  const usage = result?.token_usage || {};
  const scores = jobs.map((job) => safeScore(job.match_score));
  const average = scores.length ? scores.reduce((sum, value) => sum + value, 0) / scores.length : 0;
  const warnings = asArray(result?.trace).filter(isTraceWarning).length;
  const missing = new Set(jobs.flatMap((job) => asArray(job.missing_skills))).size;
  const stats = [
    ["匹配岗位数量", jobs.length],
    ["平均匹配分", average.toFixed(1)],
    ["Top1 匹配分", scores.length ? Math.max(...scores).toFixed(1) : "0.0"],
    ["缺失技能总数", missing],
    ["警告节点", warnings],
    ["LLM 调用", usage.calls || 0],
    ["Token 消耗", Number(usage.total_tokens || 0).toLocaleString()],
    ["估算成本", `$${Number(usage.estimated_cost_usd || 0).toFixed(4)}`],
    ["运行耗时", formatDuration(durationMs)],
  ];
  return <div className="stats-bar">{stats.map(([label, value]) => <div className="stat-card" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>;
}

export function ResultWorkspace({ durationMs, liveTrace, loading, result }) {
  const [tab, setTab] = useState("jobs");
  const [message, setMessage] = useState("");
  const tabs = useMemo(() => [
    ["jobs", "岗位推荐", asArray(result?.matched_jobs).length],
    ["gaps", "差距分析", asArray(result?.gaps).length],
    ["resume", "简历建议", asArray(result?.resume_suggestions).length],
    ["trace", "执行轨迹", asArray(loading ? liveTrace : result?.trace).length],
    ["report", "报告", null],
  ], [liveTrace, loading, result]);
  const trace = loading ? liveTrace : result?.trace;

  return (
    <section className="result-panel">
      <Stats result={result} durationMs={durationMs} />
      <PipelineOverview jobs={result?.matched_jobs} loading={loading} trace={trace} />
      <nav className="tabs">
        {tabs.map(([key, label, count]) => (
          <button className={tab === key ? "active" : ""} key={key} onClick={() => setTab(key)} type="button">
            {label}{count !== null && <span>{count}</span>}
          </button>
        ))}
      </nav>
      {message && <div className="result-actions"><span>{message}</span></div>}
      <div className="tab-body">
        {tab === "jobs" && <JobsView jobs={result?.matched_jobs} loading={loading} />}
        {tab === "gaps" && <GapsView gaps={result?.gaps} loading={loading} />}
        {tab === "resume" && <ResumeView suggestions={result?.resume_suggestions} loading={loading} />}
        {tab === "trace" && <TraceView trace={trace} loading={loading} />}
        {tab === "report" && <ReportView result={result} loading={loading} onMessage={setMessage} />}
      </div>
    </section>
  );
}
