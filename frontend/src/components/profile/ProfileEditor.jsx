import { RefreshCcw, Upload, Wand2 } from "lucide-react";

import { buildProfileJsonFromText } from "../../utils/jobpilot";

export function ProfileEditor({
  disabled,
  onChange,
  onMessage,
  sampleProfile,
  status,
  targetRole,
  value,
}) {
  async function handleFile(event) {
    const file = event.target.files?.[0];
    if (file) onChange(await file.text());
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
        </div>
        <div className="field">
          <div className="field-head">
            <label>候选人画像</label>
            <div className="field-actions">
              <span className={`validation-pill ${status.status}`}>{status.label}</span>
              <label className="icon-button" title="上传候选人 JSON 或文本">
                <Upload size={16} />
                <input disabled={disabled} type="file" accept=".json,.txt,.md" onChange={handleFile} />
              </label>
            </div>
          </div>
          <textarea disabled={disabled} value={value} onChange={(event) => onChange(event.target.value)} />
        </div>
      </div>
    </section>
  );
}
