const API_BASE = import.meta.env.VITE_API_BASE_URL || "";
const TOKEN_KEY = "jobpilot_access_token";
const ROLE_KEY = "jobpilot_role";

export function getAccessToken() {
  return window.localStorage.getItem(TOKEN_KEY) || "";
}

export function setAccessToken(token) {
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
}

export function getAuthRole() {
  return window.localStorage.getItem(ROLE_KEY) || "";
}

export function clearAuth() {
  setAccessToken("");
  window.localStorage.removeItem(ROLE_KEY);
}

async function request(path, options = {}) {
  const token = getAccessToken();
  const headers = new Headers(options.headers || {});
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `请求失败：HTTP ${response.status}`);
  }
  return data;
}

async function streamRunEvents(runId, onEvent, signal) {
  const headers = new Headers();
  const token = getAccessToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await fetch(`${API_BASE}/api/runs/${runId}/events`, {
    headers,
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`事件流连接失败：HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      let eventType = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      try {
        onEvent(eventType, JSON.parse(data));
      } catch {
        // Ignore malformed event frames; status polling remains the fallback.
      }
    }
  }
}

export const jobPilotApi = {
  health: () => request("/api/health"),
  login: async (username, password) => {
    const data = await request("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    setAccessToken(data.access_token);
    window.localStorage.setItem(ROLE_KEY, data.role || "user");
    return data;
  },
  logout: clearAuth,
  listJobs: () => request("/api/jobs"),
  extractProfileDocument: (file, targetRole) =>
    request(
      `/api/profile/document?filename=${encodeURIComponent(file.name)}&target_role=${encodeURIComponent(targetRole)}`,
      {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file,
      }
    ),
  saveJobs: (payload) =>
    request("/api/record-jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  deleteJob: (jobId) =>
    request(`/api/jobs/${encodeURIComponent(jobId)}`, {
      method: "DELETE",
    }),
  createRun: (payload, signal) =>
    request("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    }),
  getRun: (runId, signal) => request(`/api/runs/${runId}`, { signal }),
  streamRunEvents,
  cancelRun: (runId) => request(`/api/runs/${runId}`, { method: "DELETE" }),
  reviewRun: (runId, approved) =>
    request(`/api/runs/${runId}/review`, {
      method: "POST",
      body: JSON.stringify({ approved }),
    }),
};
