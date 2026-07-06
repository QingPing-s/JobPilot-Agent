import { LockKeyhole, LogIn, X } from "lucide-react";
import { useState } from "react";

import { jobPilotApi } from "../../api/jobpilot";

export function LoginPanel({ onClose, onLogin }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const result = await jobPilotApi.login("admin", password);
      onLogin(result.role);
    } catch (loginError) {
      setError(loginError.message || "登录失败。");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="admin-dialog-backdrop" role="presentation">
      <form className="login-panel admin-dialog" onSubmit={submit}>
        <button className="dialog-close" onClick={onClose} title="关闭" type="button">
          <X size={17} />
        </button>
        <LockKeyhole size={26} />
        <div>
          <h1>管理员验证</h1>
          <p>验证后可以新增或停用岗位。</p>
        </div>
        <label>
          <span>管理员密码</span>
          <input
            autoComplete="current-password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {error && <div className="error-box">{error}</div>}
        <button disabled={loading || !password} type="submit">
          <LogIn size={16} />
          {loading ? "验证中" : "进入管理员模式"}
        </button>
      </form>
    </div>
  );
}
