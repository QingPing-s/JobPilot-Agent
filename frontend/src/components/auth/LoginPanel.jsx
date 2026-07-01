import { LockKeyhole, LogIn } from "lucide-react";
import { useState } from "react";

import { jobPilotApi } from "../../api/jobpilot";

export function LoginPanel({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const result = await jobPilotApi.login(username, password);
      onLogin(result.role);
    } catch (loginError) {
      setError(loginError.message || "登录失败。");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="login-page">
      <form className="login-panel" onSubmit={submit}>
        <LockKeyhole size={26} />
        <div>
          <h1>JobPilot Agent</h1>
          <p>登录后访问岗位匹配工作台</p>
        </div>
        <label>
          <span>用户名</span>
          <input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} />
        </label>
        <label>
          <span>密码</span>
          <input
            autoComplete="current-password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {error && <div className="error-box">{error}</div>}
        <button disabled={loading || !username || !password} type="submit">
          <LogIn size={16} />
          {loading ? "登录中" : "登录"}
        </button>
      </form>
    </main>
  );
}
