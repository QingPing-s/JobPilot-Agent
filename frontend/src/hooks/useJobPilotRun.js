import { useCallback, useEffect, useRef, useState } from "react";

import { jobPilotApi } from "../api/jobpilot";

const TERMINAL = new Set(["completed", "failed", "cancelled", "timed_out", "awaiting_review"]);

function wait(ms, signal) {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        window.clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true }
    );
  });
}

export function useJobPilotRun() {
  const [result, setResult] = useState(null);
  const [runId, setRunId] = useState("");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [durationMs, setDurationMs] = useState(0);
  const [events, setEvents] = useState([]);
  const controllerRef = useRef(null);

  const reset = useCallback(() => {
    controllerRef.current?.abort();
    setResult(null);
    setRunId("");
    setStatus("idle");
    setError("");
    setDurationMs(0);
    setEvents([]);
  }, []);

  const run = useCallback(async (payload) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const started = performance.now();
    setResult(null);
    setError("");
    setStatus("submitting");
    setEvents([]);

    try {
      const created = await jobPilotApi.createRun(payload, controller.signal);
      setRunId(created.run_id);
      setStatus(created.status);
      const eventStream = jobPilotApi
        .streamRunEvents(
          created.run_id,
          (eventType, event) => {
            if (eventType === "node") {
              setEvents((current) => [...current, event]);
            } else if (eventType === "status" && event.status) {
              setStatus(event.status);
            }
          },
          controller.signal
        )
        .catch((streamError) => {
          if (streamError.name !== "AbortError") {
            console.warn("JobPilot SSE unavailable; polling remains active.", streamError);
          }
        });
      let record = created;
      if (TERMINAL.has(record.status)) {
        record = await jobPilotApi.getRun(created.run_id, controller.signal);
        setStatus(record.status);
      }
      while (!TERMINAL.has(record.status)) {
        await wait(700, controller.signal);
        record = await jobPilotApi.getRun(created.run_id, controller.signal);
        setStatus(record.status);
      }
      if (record.status === "failed") throw new Error(record.error || "Agent 运行失败。");
      await eventStream;
      setResult(record.result || null);
      setDurationMs(performance.now() - started);
      return record;
    } catch (runError) {
      if (runError.name !== "AbortError") {
        setError(runError.message || "Agent 请求失败。");
        setStatus("failed");
      }
      setDurationMs(performance.now() - started);
      return null;
    }
  }, []);

  const cancel = useCallback(async () => {
    if (!runId) return;
    try {
      const record = await jobPilotApi.cancelRun(runId);
      setStatus(record.status);
    } catch (cancelError) {
      setError(cancelError.message || "取消运行失败。");
    }
  }, [runId]);

  useEffect(() => () => controllerRef.current?.abort(), []);

  return {
    cancel,
    durationMs,
    error,
    events,
    loading: ["submitting", "queued", "running", "cancelling"].includes(status),
    reset,
    result,
    run,
    runId,
    status,
  };
}
