import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");
const screenshotPath = path.join(repoRoot, "outputs", "jobpilot_ui_smoke.png");
const appUrl = process.env.JOBPILOT_UI_URL || "http://127.0.0.1:5173";

async function launchBrowser() {
  try {
    return await chromium.launch({ channel: "msedge", headless: true });
  } catch (error) {
    const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
    return chromium.launch({ executablePath: edgePath, headless: true });
  }
}

async function main() {
  await mkdir(path.dirname(screenshotPath), { recursive: true });

  const browser = await launchBrowser();
  const context = await browser.newContext({
    acceptDownloads: true,
    permissions: ["clipboard-read", "clipboard-write"],
    viewport: { width: 1440, height: 1100 },
  });
  const page = await context.newPage();

  try {
    await page.goto(appUrl, { waitUntil: "networkidle", timeout: 30_000 });
    await page.getByRole("heading", { name: "JobPilot Agent 工作台" }).waitFor({ timeout: 15_000 });
    await page.locator(".stats-bar .stat-card").first().waitFor({ timeout: 15_000 });
    await page.getByRole("button", { name: /加载示例 JD/ }).click();
    await page.getByRole("button", { name: /保存 JD 到岗位库/ }).click();
    await page.getByText(/已保存 .* 条岗位到岗位库/).waitFor({ timeout: 30_000 });
    await page.getByRole("button", { name: "运行 Agent" }).click();

    await page.locator(".job-card").first().waitFor({ timeout: 120_000 });
    const jobCount = await page.locator(".job-card").count();
    if (jobCount < 1) {
      throw new Error("Expected at least one matched job card after running the agent.");
    }
    const statCount = await page.locator(".stat-card").count();
    if (statCount < 7) {
      throw new Error("Expected the enhanced StatsBar to expose seven stat cards.");
    }
    await page.locator(".score-breakdown").first().waitFor({ timeout: 30_000 });

    const tabs = page.locator("nav.tabs");

    await tabs.getByRole("button", { name: /差距分析/ }).click();
    await page.locator(".plain-section").first().waitFor({ timeout: 30_000 });

    await tabs.getByRole("button", { name: /简历建议/ }).click();
    await page.locator(".plain-section").first().waitFor({ timeout: 30_000 });

    await tabs.getByRole("button", { name: /执行轨迹/ }).click();
    await page.locator(".trace-event.success").first().waitFor({ timeout: 30_000 });
    const traceRows = await page.locator(".trace-event").count();

    await tabs.getByRole("button", { name: /报告/ }).click();
    await page.getByRole("button", { name: /导出 Markdown 报告/ }).waitFor({ timeout: 30_000 });
    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 30_000 }),
      page.getByRole("button", { name: /导出 Markdown 报告/ }).click(),
    ]);
    if (!download.suggestedFilename().endsWith(".md")) {
      throw new Error("Expected Markdown report download.");
    }
    await page.getByRole("button", { name: /复制 Top1 简历建议/ }).click();
    await page.getByText("Top1 简历建议已复制").waitFor({ timeout: 30_000 });

    await tabs.getByRole("button", { name: /岗位推荐/ }).click();
    await page.locator(".score-breakdown").first().waitFor({ timeout: 30_000 });
    await page.getByText(/可投但需优化|不建议优先投递|较匹配|强匹配/).first().waitFor({ timeout: 30_000 });
    await page.screenshot({ path: screenshotPath, fullPage: true });

    const summary = {
      appUrl,
      jobCount,
      statCount,
      traceRows,
      screenshotPath,
    };
    console.log(JSON.stringify(summary, null, 2));
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
