import { Database, FileText, RefreshCcw, Search, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";

import { asArray } from "../../utils/jobpilot";

export function JobSourceSelector({ disabled, onChange, value }) {
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

export function JobLibraryPanel({ isAdmin, jobs, loading, onDelete, onRefresh }) {
  const [query, setQuery] = useState("");
  const activeJobs = asArray(jobs);
  const filteredJobs = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return activeJobs;
    return activeJobs.filter((job) =>
      [job.title, job.company, job.location, job.salary, job.education]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(normalized))
    );
  }, [activeJobs, query]);

  return (
    <section className="job-library-panel">
      <div className="library-head">
        <div>
          <strong>岗位库</strong>
          <span>SQLite 持久化岗位；维护操作需要管理员权限</span>
        </div>
        <button disabled={loading} onClick={onRefresh} type="button">
          <RefreshCcw size={14} />
          刷新
        </button>
      </div>
      <div className="library-tools">
        <label className="library-search">
          <Search size={14} />
          <input disabled={loading} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索岗位、公司、地点" />
        </label>
        <span className="library-count">{filteredJobs.length}/{activeJobs.length} 条</span>
      </div>
      <div className="library-list">
        {filteredJobs.length ? (
          filteredJobs.map((job) => (
            <div className="library-item" key={job.job_id}>
              <div>
                <strong>{job.title || "未命名岗位"}</strong>
                <span>{job.company || "未知公司"}{job.location ? ` · ${job.location}` : ""}</span>
                <small>{job.salary || "薪资未标注"}{job.education ? ` · ${job.education}` : ""}</small>
              </div>
              {isAdmin && (
                <button
                  className="library-delete"
                  disabled={loading}
                  onClick={() => onDelete(job)}
                  title="停用岗位"
                  type="button"
                >
                  <Trash2 size={14} />
                </button>
              )}
            </div>
          ))
        ) : (
          <div className="library-empty">{activeJobs.length ? "没有匹配当前搜索条件的岗位。" : "岗位库为空，请由管理员导入 JD。"}</div>
        )}
      </div>
    </section>
  );
}
