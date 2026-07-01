const normalizedBase = import.meta.env.BASE_URL.replace(/\/$/, "");

export function sitePath(path = "") {
  return `${normalizedBase}/${path.replace(/^\/+/, "")}`;
}

export const githubUrl = "https://github.com/QingPing-s/JobPilot-Agent";
export const liveDemoUrl =
  "https://jobpilot-agent.gentlefield-019d4ae8.eastasia.azurecontainerapps.io/";
