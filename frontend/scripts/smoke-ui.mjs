import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");
const screenshotPath = path.join(repoRoot, "outputs", "jobpilot_ui_smoke.png");
const mobileScreenshotPath = path.join(repoRoot, "outputs", "jobpilot_ui_mobile.png");
const failureScreenshotPath = path.join(repoRoot, "outputs", "jobpilot_ui_smoke_failure.png");
const appUrl = process.env.JOBPILOT_UI_URL || "http://127.0.0.1:5173";

async function launchBrowser() {
  try {
    return await chromium.launch({ headless: true });
  } catch {
    try {
      return await chromium.launch({ channel: "msedge", headless: true });
    } catch {
      const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
      return chromium.launch({ executablePath: edgePath, headless: true });
    }
  }
}

async function clickIfVisible(locator) {
  if (await locator.isVisible().catch(() => false)) {
    await locator.click();
    return true;
  }
  return false;
}

async function ensureRunnable(page) {
  const runButton = page.locator(".run-button");
  await runButton.waitFor({ timeout: 15_000 });

  if (await runButton.isEnabled()) {
    return;
  }

  await clickIfVisible(page.getByRole("button", { name: /加载示例候选人/ }));
  if (await clickIfVisible(page.getByRole("button", { name: /使用岗位库/ }))) {
    await page.waitForFunction(
      () => {
        const button = document.querySelector(".run-button");
        return button && !button.disabled;
      },
      null,
      { timeout: 8_000 }
    ).catch(() => {});
  }

  if (await runButton.isDisabled()) {
    await clickIfVisible(page.getByRole("button", { name: /手动输入 JD/ }));
    await clickIfVisible(page.getByRole("button", { name: /加载示例 JD/ }));
  }

  await runButton.waitFor({ state: "visible", timeout: 15_000 });
  if (await runButton.isDisabled()) {
    throw new Error("Run button is still disabled after loading profile and JD source.");
  }
}

async function useDeterministicSettings(page) {
  const settingsGrid = page.locator(".settings-grid");
  if (!(await settingsGrid.isVisible().catch(() => false))) {
    await page.getByRole("button", { name: /运行设置/ }).click();
  }

  await settingsGrid.waitFor({ timeout: 15_000 });
  await settingsGrid.locator('input[type="number"]').fill("5");

  const checkboxes = settingsGrid.locator('input[type="checkbox"]');
  const count = await checkboxes.count();
  for (let index = 0; index < count; index += 1) {
    const checkbox = checkboxes.nth(index);
    if (await checkbox.isChecked()) {
      await checkbox.uncheck();
    }
  }
}

async function collectDebug(page) {
  const activeTabText = await page.locator(".tab-body").innerText({ timeout: 2_000 }).catch(() => "");
  const errorText = await page.locator(".error-box").innerText({ timeout: 2_000 }).catch(() => "");
  const loadingText = await page.locator(".loading-state").innerText({ timeout: 2_000 }).catch(() => "");
  return {
    url: page.url(),
    errorText,
    loadingText,
    activeTabText: activeTabText.slice(0, 800),
  };
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
    await page.getByRole("heading", { name: /JobPilot Agent 工作台/ }).waitFor({ timeout: 15_000 });
    await page.locator(".stats-bar .stat-card").first().waitFor({ timeout: 15_000 });

    const statLabels = await page.locator(".stat-card span").evaluateAll((nodes) => nodes.map((node) => node.textContent));
    const requiredStats = ["匹配岗位数量", "平均匹配分", "Top1 匹配分", "缺失技能总数", "警告节点", "运行耗时"];
    for (const label of requiredStats) {
      if (!statLabels.includes(label)) {
        throw new Error(`Expected stat card: ${label}`);
      }
    }

    await ensureRunnable(page);
    await useDeterministicSettings(page);
    await page.locator(".run-button").click();

    try {
      await page.locator(".job-card").first().waitFor({ timeout: 120_000 });
    } catch (error) {
      const debug = await collectDebug(page);
      await page.screenshot({ path: failureScreenshotPath, fullPage: true }).catch(() => {});
      throw new Error(`Timed out waiting for job cards. Debug: ${JSON.stringify(debug, null, 2)}`, { cause: error });
    }

    const jobCount = await page.locator(".job-card").count();
    if (jobCount < 1) {
      throw new Error("Expected at least one matched job card after running the agent.");
    }

    const statCount = await page.locator(".stat-card").count();
    if (statCount < 8) {
      throw new Error(`Expected at least eight stat cards, got ${statCount}.`);
    }

    await page.locator(".score-breakdown").first().waitFor({ timeout: 30_000 });

    const tabs = page.locator("nav.tabs");
    await tabs.getByRole("button", { name: /差距分析/ }).click();
    await page.locator(".plain-section").first().waitFor({ timeout: 30_000 });

    await tabs.getByRole("button", { name: /简历建议/ }).click();
    await page.locator(".plain-section").first().waitFor({ timeout: 30_000 });

    await tabs.getByRole("button", { name: /执行轨迹/ }).click();
    await page.locator(".trace-event").first().waitFor({ timeout: 30_000 });
    const traceRows = await page.locator(".trace-event").count();
    const warningBadgeCount = await page.locator(".trace-status.warning").count();

    await tabs.getByRole("button", { name: /报告/ }).click();
    await page.getByRole("button", { name: /导出 Markdown 报告/ }).waitFor({ timeout: 30_000 });
    const [download] = await Promise.all([
      page.waitForEvent("download", { timeout: 30_000 }),
      page.getByRole("button", { name: /导出 Markdown 报告/ }).click(),
    ]);
    if (!download.suggestedFilename().endsWith(".md")) {
      throw new Error("Expected Markdown report download.");
    }

    await tabs.getByRole("button", { name: /岗位推荐/ }).click();
    await page.locator(".score-breakdown").first().waitFor({ timeout: 30_000 });
    await page.screenshot({ path: screenshotPath, fullPage: true });

    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(300);
    const hasHorizontalOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth + 1
    );
    if (hasHorizontalOverflow) {
      throw new Error("Mobile layout has horizontal overflow.");
    }
    await page.screenshot({ path: mobileScreenshotPath, fullPage: false });

    const summary = {
      appUrl,
      jobCount,
      statCount,
      traceRows,
      warningBadgeCount,
      screenshotPath,
      mobileScreenshotPath,
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
