"use client";

import { useCallback, useRef, useState } from "react";

export interface Step {
  step_id: string;
  step_name: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  tool_name?: string;
  output_summary?: string;
  duration_ms?: number;
  error?: string;
}

export interface CaseSubmission {
  patient_text: string;
  include_drug_check?: boolean;
  include_guidelines?: boolean;
  use_langgraph?: boolean;
}

export interface UseAgentWebSocketReturn {
  steps: Step[];
  report: any | null;
  isRunning: boolean;
  isWarmingUp: boolean;
  warmUpMessage: string | null;
  error: string | null;
  submitCase: (submission: CaseSubmission) => void;
  reset: () => void;
}

function getWsUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_WS_URL;
  if (envUrl) return envUrl;

  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/ws/agent`;
  }

  return "ws://localhost:8002/ws/agent";
}

export function useAgentWebSocket(): UseAgentWebSocketReturn {
  const [steps, setSteps] = useState<Step[]>([]);
  const [report, setReport] = useState<any | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isWarmingUp, setIsWarmingUp] = useState(false);
  const [warmUpMessage, setWarmUpMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const submitCase = useCallback((submission: CaseSubmission) => {
    setSteps([]);
    setReport(null);
    setError(null);
    setIsRunning(true);
    setIsWarmingUp(false);
    setWarmUpMessage(null);

    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(getWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify(submission));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case "ack":
            break;

          case "warming_up":
            setIsWarmingUp(true);
            setWarmUpMessage(data.message);
            break;

          case "model_ready":
            setIsWarmingUp(false);
            setWarmUpMessage(null);
            break;

          case "step_update":
            setSteps((prev) => {
              const existing = prev.findIndex(
                (s) => s.step_id === data.step.step_id
              );
              if (existing >= 0) {
                const updated = [...prev];
                updated[existing] = data.step;
                return updated;
              }
              return [...prev, data.step];
            });
            break;

          case "report":
            setReport(data.report);
            break;

          case "complete":
            setIsRunning(false);
            break;

          case "error":
            setError(data.message);
            setIsRunning(false);
            break;
        }
      } catch (err) {
        console.error("Failed to parse WebSocket message:", err);
      }
    };

    ws.onerror = () => {
      setError("WebSocket connection failed. Verify that backend is running on port 8002.");
      setIsRunning(false);
    };

    ws.onclose = () => {
      setIsRunning(false);
    };
  }, []);

  const reset = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setSteps([]);
    setReport(null);
    setError(null);
    setIsRunning(false);
    setIsWarmingUp(false);
    setWarmUpMessage(null);
  }, []);

  return { steps, report, isRunning, isWarmingUp, warmUpMessage, error, submitCase, reset };
}
