import assert from "node:assert/strict";
import test from "node:test";

import {
  buildProfileJsonFromText,
  matchLevel,
  parseProfileInput,
  splitJdText,
} from "./jobpilot.js";

test("parseProfileInput preserves structured JSON", () => {
  assert.deepEqual(parseProfileInput('{"skills":["Python"]}'), {
    user_profile_json: { skills: ["Python"] },
  });
});

test("splitJdText separates multiple job descriptions", () => {
  assert.deepEqual(splitJdText("JD one\n---JOB---\nJD two"), ["JD one", "JD two"]);
});

test("local profile parser extracts technical and soft skills", () => {
  const profile = buildProfileJsonFromText(
    "人工智能研究生，熟悉 Python 和 LangGraph，学习能力强，能够独立解决问题。",
    "AI Agent Intern"
  );
  assert.ok(profile.skills.includes("Python"));
  assert.ok(profile.skills.includes("LangGraph"));
  assert.ok(profile.soft_skills.includes("学习速度快"));
  assert.equal(profile.target_roles[0], "AI Agent Intern");
});

test("match level follows score thresholds", () => {
  assert.equal(matchLevel(85), "强匹配");
  assert.equal(matchLevel(20), "不建议优先投递");
});
