import { FileUp, Loader2, RefreshCcw, Wand2 } from "lucide-react";

import { buildProfileJsonFromText } from "../../utils/jobpilot";

export function ProfileEditor({
  disabled,
  documentLoading,
  onChange,
  onDocument,
  onMessage,
  sampleProfile,
  status,
  targetRole,
  value,
}) {
  async function handleDocument(event) {
    const file = event.target.files?.[0];
    if (file) await onDocument(file);
    event.target.value = "";
  }

  function organize() {
    try {
      const profile = buildProfileJsonFromText(value, targetRole);
      onChange(JSON.stringify(profile, null, 2));
      onMessage("候选人画像已整理为 CandidateProfile JSON。");
    } catch (error) {
      onMessage(error.message, true);
    }
  }

  return (
    <section className="input-section open">
      <div className="section-trigger static-trigger">
        <span>候选人画像</span>
        <em>{status.label}</em>
      </div>
      <div className="section-body">
        <div className="input-actions">
          <button disabled={disabled} onClick={() => onChange(JSON.stringify(sampleProfile, null, 2))} type="button">
            <RefreshCcw size={14} />
            加载示例候选人
          </button>
          <button disabled={disabled} onClick={organize} type="button">
            <Wand2 size={14} />
            整理画像
          </button>
          <label className={`upload-action ${disabled || documentLoading ? "disabled" : ""}`}>
            {documentLoading ? <Loader2 className="spin" size={14} /> : <FileUp size={14} />}
            {documentLoading ? "解析中" : "上传简历"}
            <input
              accept=".bmp,.docx,.jpeg,.jpg,.json,.md,.pdf,.png,.txt,.webp"
              disabled={disabled || documentLoading}
              onChange={handleDocument}
              type="file"
            />
          </label>
        </div>
        <div className="field">
          <div className="field-head">
            <label>候选人画像</label>
            <div className="field-actions">
              <span className={`validation-pill ${status.status}`}>{status.label}</span>
            </div>
          </div>
          <textarea disabled={disabled} value={value} onChange={(event) => onChange(event.target.value)} />
        </div>
      </div>
    </section>
  );
}
