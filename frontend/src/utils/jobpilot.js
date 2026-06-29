export const PIPELINE_NODES = [
  ["profile_node", "候选人画像"],
  ["jd_parse_node", "JD 解析"],
  ["retrieve_node", "混合召回"],
  ["rerank_node", "岗位重排"],
  ["match_score_node", "匹配评分"],
  ["gap_analysis_node", "差距分析"],
  ["resume_suggestion_node", "简历建议"],
];

export function asArray(value) {
  return Array.isArray(value) ? value : [];
}

export function safeScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(100, number)) : 0;
}

export function scoreTone(score) {
  const value = safeScore(score);
  if (value >= 80) return "excellent";
  if (value >= 60) return "good";
  if (value >= 40) return "fair";
  return "poor";
}

export function matchLevel(score) {
  const value = safeScore(score);
  if (value >= 80) return "强匹配";
  if (value >= 60) return "较匹配";
  if (value >= 40) return "可投但需优化";
  return "不建议优先投递";
}

export function parseProfileInput(text) {
  const trimmed = text.trim();
  if (!trimmed) return {};
  try {
    return { user_profile_json: JSON.parse(trimmed) };
  } catch {
    return { user_profile_text: trimmed };
  }
}

export function splitJdText(text) {
  return text
    .split(/\n-{3,}\s*JOB\s*-{3,}\n/i)
    .map((item) => item.trim())
    .filter(Boolean);
}

function uniqueList(items) {
  return [...new Set(items.map((item) => String(item).trim()).filter(Boolean))];
}

function extractSkills(text) {
  const patterns = [
    ["Python", /\bpython\b/i],
    ["TypeScript", /\btypescript\b/i],
    ["React", /\breact\b/i],
    ["FastAPI", /\bfastapi\b/i],
    ["LangGraph", /\blanggraph\b/i],
    ["LangChain", /\blangchain\b/i],
    ["RAG", /\brag\b|检索增强生成/i],
    ["LLM", /\bllm\b|大模型|大语言模型/i],
    ["AI Agent", /\bai agent\b|\bagent\b|智能体/i],
    ["DeepSeek API", /\bdeepseek\b/i],
    ["ChromaDB", /\bchromadb\b|向量库|vector database/i],
    ["Prompt Engineering", /prompt engineering|提示词/i],
    ["Tool Calling", /tool calling|function calling|工具调用/i],
    ["PyTorch", /\bpytorch\b/i],
    ["Docker", /\bdocker\b/i],
    ["SQL", /\bsql\b/i],
  ];
  return patterns.filter(([, pattern]) => pattern.test(text)).map(([name]) => name);
}

function extractSoftSkills(text) {
  const patterns = [
    ["学习速度快", /学习速度快|学习能力强|快速学习|快速上手/],
    ["主动查阅资料", /主动查阅资料|主动查资料|阅读文档/],
    ["问题拆解能力", /问题拆解|拆解问题|任务拆解/],
    ["自驱力强", /自驱力|自我驱动|主动性强|积极主动/],
    ["沟通协作能力", /沟通协作|沟通能力|团队协作|团队合作/],
    ["责任心强", /责任心|责任感|认真负责/],
    ["能独立解决问题", /独立解决问题|问题解决能力/],
  ];
  return patterns.filter(([, pattern]) => pattern.test(text)).map(([name]) => name);
}

export function buildProfileJsonFromText(text, targetRole) {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("请先输入候选人文本或 JSON。");
  try {
    const parsed = JSON.parse(trimmed);
    return {
      name: parsed.name ?? null,
      education: asArray(parsed.education),
      skills: asArray(parsed.skills),
      soft_skills: asArray(parsed.soft_skills),
      projects: asArray(parsed.projects),
      internships: asArray(parsed.internships),
      target_roles: asArray(parsed.target_roles),
      preferences:
        parsed.preferences && typeof parsed.preferences === "object" && !Array.isArray(parsed.preferences)
          ? parsed.preferences
          : {},
    };
  } catch {
    const lines = trimmed.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    const education = lines
      .filter((line) => /本科|硕士|博士|计算机|软件工程|人工智能|computer science/i.test(line))
      .slice(0, 4);
    const projectLines = lines.filter((line) => /项目|project/i.test(line)).slice(0, 5);
    const skills = uniqueList(extractSkills(trimmed));
    return {
      name: null,
      education: uniqueList(education),
      skills,
      soft_skills: uniqueList(extractSoftSkills(trimmed)),
      projects: projectLines.map((line, index) => ({
        name: line.split(/[，,。.;；]/)[0] || `项目经历 ${index + 1}`,
        description: line,
        tech_stack: skills.filter((skill) => line.toLowerCase().includes(skill.toLowerCase())),
        highlights: [],
      })),
      internships: [],
      target_roles: targetRole ? [targetRole] : [],
      preferences: { source: "frontend_local_parser", raw_text: trimmed },
    };
  }
}

export function validateProfileInput(text) {
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

export function summarizeProfileInput(text) {
  try {
    const parsed = JSON.parse(text);
    return {
      mode: "JSON 画像",
      skillCount: asArray(parsed.skills).length,
      projectCount: asArray(parsed.projects).length,
      educationCount: asArray(parsed.education).length,
    };
  } catch {
    return {
      mode: text.trim() ? "文本画像" : "未填写",
      skillCount: 0,
      projectCount: 0,
      educationCount: 0,
    };
  }
}

export function formatDuration(ms) {
  if (!ms) return "尚未运行";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function isTraceWarning(record) {
  const status = record?.status || record?.event_type || "";
  const message = `${record?.message || ""} ${record?.error_message || ""}`.toLowerCase();
  return (
    status === "error" ||
    status === "warning" ||
    message.includes("fallback") ||
    message.includes("兜底") ||
    message.includes("回退") ||
    message.includes("失败")
  );
}

export function pipelineStatus(node, trace, loading) {
  const records = asArray(trace).filter((record) => record.node === node);
  if (records.some((record) => record.status === "error" || record.event_type === "error")) return "error";
  if (records.some((record) => ["success", "end"].includes(record.status || record.event_type))) return "success";
  if (loading) {
    const completed = new Set(
      asArray(trace)
        .filter((record) => ["success", "end"].includes(record.status || record.event_type))
        .map((record) => record.node)
    );
    return PIPELINE_NODES.find(([candidate]) => !completed.has(candidate))?.[0] === node ? "running" : "pending";
  }
  return "pending";
}
