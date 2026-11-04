"use client";

import { useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  Clock,
  AlertCircle,
  MinusCircle,
  Loader2,
  Bot,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  Sparkles,
} from "lucide-react";

export interface Step {
  step_id: string;
  step_name: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  tool_name?: string;
  output_summary?: string;
  duration_ms?: number;
  error?: string;
}

interface AgentPipelineProps {
  steps: Step[];
  isRunning: boolean;
}

const STEP_SUBTITLES: Record<string, string> = {
  parse: "Structuring demographics, chief complaint, HPI, medications, and labs",
  context_extraction: "Normalizing clinical entities, timeline & missing critical fields",
  reason: "Synthesizing differential diagnoses & risk-stratified workup plan",
  drugs: "Auditing OpenFDA & RxNorm for severe drug-drug interactions",
  evidence_retrieval: "Hybrid BM25 + Dense ChromaDB + live PubMed guidelines search",
  guidelines: "Retrieving clinical practice guidelines & standard of care pathways",
  evidence_grading: "Applying GRADE study design hierarchy & recency evaluation",
  conflicts: "Detecting care omissions, contraindications & treatment gaps",
  synthesize: "Synthesizing comprehensive Clinical Decision Support dossier",
  verify: "Extracting atomic claims & validating via NLI literature entailment",
  uncertainty: "Quantifying evidence sufficiency & evaluating safety abstention threshold",
};

export function AgentPipeline({ steps, isRunning }: AgentPipelineProps) {
  const [expandedStepId, setExpandedStepId] = useState<string | null>(null);
  const activeStepRef = useRef<HTMLDivElement | null>(null);

  // Automatically scroll running step into view on mobile/tablet
  useEffect(() => {
    if (activeStepRef.current) {
      activeStepRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [steps]);

  // Compute progress percentage
  const completedCount = steps.filter((s) => s.status === "completed").length;
  const totalCount = steps.length || 1;
  const progressPercent = Math.round((completedCount / totalCount) * 100);

  const runningStep = steps.find((s) => s.status === "running");

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-card overflow-hidden sticky top-6">
      {/* Header with Live Progress */}
      <div className="p-4 sm:p-5 bg-gradient-to-b from-slate-50 to-white border-b border-slate-100">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-sm">
              <Bot className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Agentic Pipeline Execution
              </h3>
              <p className="text-[11px] text-slate-500">
                {isRunning ? (
                  <span className="text-indigo-600 font-medium flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-indigo-600 animate-ping inline-block" />
                    Executing: {runningStep?.step_name || "Reasoning"}...
                  </span>
                ) : completedCount === steps.length && steps.length > 0 ? (
                  <span className="text-emerald-600 font-medium">Pipeline execution finalized</span>
                ) : (
                  "Multi-agent diagnostic workflow"
                )}
              </p>
            </div>
          </div>

          <div className="text-right">
            <span className="text-xs font-bold text-slate-700">
              {completedCount}/{totalCount}
            </span>
            <span className="text-[10px] text-slate-400 block font-medium">
              {progressPercent}% Complete
            </span>
          </div>
        </div>

        {/* Progress Bar Track */}
        <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden mt-3">
          <div
            className={`h-full transition-all duration-500 ease-out rounded-full ${
              completedCount === steps.length && steps.length > 0
                ? "bg-emerald-500"
                : "bg-indigo-600"
            }`}
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Steps List */}
      <div className="p-4 sm:p-5 max-h-[calc(100vh-280px)] overflow-y-auto space-y-3">
        {steps.map((step, index) => {
          const isCurrentRunning = step.status === "running";
          const isCompleted = step.status === "completed";
          const isFailed = step.status === "failed";
          const isSkipped = step.status === "skipped";
          const isPending = step.status === "pending";

          const isExpanded = expandedStepId === step.step_id || (isFailed && !expandedStepId);

          return (
            <div
              key={step.step_id}
              ref={isCurrentRunning ? activeStepRef : null}
              className={`rounded-xl border transition-all duration-200 overflow-hidden ${
                isCurrentRunning
                  ? "bg-indigo-50/50 border-indigo-400 ring-2 ring-indigo-500/10 shadow-sm"
                  : isCompleted
                  ? "bg-white border-slate-200/90 hover:border-slate-300"
                  : isFailed
                  ? "bg-rose-50/40 border-rose-300"
                  : isSkipped
                  ? "bg-slate-50/60 border-slate-200 opacity-60"
                  : "bg-slate-50/40 border-slate-200/60 opacity-60"
              }`}
            >
              <div
                onClick={() => setExpandedStepId(isExpanded ? null : step.step_id)}
                className="p-3 sm:p-3.5 flex items-start gap-3 cursor-pointer select-none"
              >
                {/* Step Status Icon */}
                <div className="mt-0.5 shrink-0">
                  {isCompleted && (
                    <div className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    </div>
                  )}
                  {isCurrentRunning && (
                    <div className="w-5 h-5 rounded-full bg-indigo-100 text-indigo-600 flex items-center justify-center">
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    </div>
                  )}
                  {isFailed && (
                    <div className="w-5 h-5 rounded-full bg-rose-100 text-rose-600 flex items-center justify-center">
                      <AlertCircle className="w-3.5 h-3.5" />
                    </div>
                  )}
                  {isSkipped && (
                    <div className="w-5 h-5 rounded-full bg-slate-200 text-slate-500 flex items-center justify-center">
                      <MinusCircle className="w-3.5 h-3.5" />
                    </div>
                  )}
                  {isPending && (
                    <div className="w-5 h-5 rounded-full bg-slate-100 text-slate-400 flex items-center justify-center">
                      <Clock className="w-3 h-3" />
                    </div>
                  )}
                </div>

                {/* Step Details */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-1">
                    <span className="text-xs font-semibold text-slate-900 truncate">
                      {index + 1}. {step.step_name}
                    </span>

                    {/* Step Duration / Status Badge */}
                    <div className="flex items-center gap-1.5 shrink-0">
                      {step.duration_ms !== undefined && step.duration_ms !== null && (
                        <span className="text-[10px] text-slate-400 font-mono">
                          {step.duration_ms >= 1000
                            ? `${(step.duration_ms / 1000).toFixed(1)}s`
                            : `${step.duration_ms}ms`}
                        </span>
                      )}
                      {step.tool_name && (
                        <span className="text-[9px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-500 font-mono hidden sm:inline">
                          {step.tool_name}
                        </span>
                      )}
                    </div>
                  </div>

                  <p className="text-[11px] text-slate-500 mt-0.5 line-clamp-1">
                    {step.output_summary || STEP_SUBTITLES[step.step_id] || "Step pending execution"}
                  </p>
                </div>

                {/* Accordion Caret */}
                <button
                  type="button"
                  className="text-slate-400 hover:text-slate-600 p-0.5"
                  aria-label="Toggle details"
                >
                  {isExpanded ? (
                    <ChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>

              {/* Expanded details container */}
              {isExpanded && (
                <div className="px-3.5 pb-3 pt-1 border-t border-slate-100 text-[11px] space-y-2 bg-slate-50/50">
                  {step.output_summary && (
                    <div>
                      <span className="font-semibold text-slate-600 block mb-0.5">Summary Output:</span>
                      <p className="text-slate-700 bg-white p-2 rounded border border-slate-200/80 leading-relaxed">
                        {step.output_summary}
                      </p>
                    </div>
                  )}

                  {step.error && (
                    <div>
                      <span className="font-semibold text-rose-600 block mb-0.5">Failure Diagnostic:</span>
                      <p className="text-rose-700 bg-rose-50 p-2 rounded border border-rose-200 font-mono text-[10px]">
                        {step.error}
                      </p>
                    </div>
                  )}

                  <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1">
                    <span>Tool identifier: <code className="text-slate-600">{step.tool_name || step.step_id}</code></span>
                    <span>State: <strong className="uppercase text-slate-600">{step.status}</strong></span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
