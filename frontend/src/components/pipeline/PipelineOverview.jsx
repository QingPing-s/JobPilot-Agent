import { Activity, AlertTriangle, BarChart3, Briefcase, Database, Route } from "lucide-react";

import { asArray, PIPELINE_NODES, pipelineStatus, safeScore } from "../../utils/jobpilot";

const STATUS_LABELS = {
  success: "已完成",
  running: "运行中",
  error: "失败",
  pending: "等待中",
};

export function PipelineOverview({ jobs, loading, trace }) {
  const matchedJobs = asArray(jobs);
  const records = asArray(trace);
  const highestScore = matchedJobs.length
    ? Math.max(...matchedJobs.map((job) => safeScore(job.match_score)))
    : 0;
  const warningCount = records.filter(
    (record) => record.status === "warning" || record.status === "error"
  ).length;
  const completed = PIPELINE_NODES.filter(
    ([node]) => pipelineStatus(node, records, loading) === "success"
  ).length;

  const summaries = [
    [Route, "Agent 流程", `${completed}/${PIPELINE_NODES.length} 节点完成`],
    [Database, "推荐结果", `${matchedJobs.length} 个岗位`],
    [BarChart3, "最高分", `${highestScore.toFixed(1)}/100`],
    [AlertTriangle, "异常事件", `${warningCount} 个`],
  ];

  return (
    <>
      <section className="workflow-overview">
      <div className="overview-summary">
        {summaries.map(([Icon, label, value]) => (
          <div key={label}>
            <Icon size={17} />
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
      <div className="workflow-steps" aria-label="Agent 执行流程">
        {PIPELINE_NODES.map(([node, label], index) => {
          const status = pipelineStatus(node, records, loading);
          return (
            <div className={`workflow-step ${status}`} key={node}>
              <span className="step-index">{index + 1}</span>
              <div>
                <strong>{label}</strong>
                <small>{STATUS_LABELS[status]}</small>
              </div>
              {status === "running" && <Activity className="spin" size={14} />}
            </div>
          );
        })}
      </div>
      </section>
    </>
  );
}
