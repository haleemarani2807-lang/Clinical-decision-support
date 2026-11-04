"use client";

import { useState } from "react";
import {
  FileText,
  AlertTriangle,
  ShieldAlert,
  CheckCircle2,
  Stethoscope,
  Pill,
  BookOpen,
  ArrowRight,
  Download,
  Copy,
  Check,
  Printer,
  ChevronDown,
  ChevronUp,
  Info,
  ExternalLink,
  ShieldCheck,
} from "lucide-react";
import { reportToMarkdown } from "@/lib/reportToMarkdown";

export interface Report {
  patient_summary: string;
  differential_diagnosis: Array<{
    diagnosis: string;
    icd10_code?: string;
    likelihood: string;
    likelihood_raw?: string;
    supporting_evidence: string[];
    reasoning: string;
  }>;
  drug_interaction_warnings: Array<{
    drug_a: string;
    drug_b: string;
    severity: string;
    description: string;
  }>;
  conflicts: Array<{
    conflict_type: string;
    severity: string;
    guideline_source: string;
    guideline_text: string;
    patient_data: string;
    description: string;
    suggested_resolution?: string;
  }>;
  guideline_recommendations: string[];
  suggested_next_steps: Array<{
    action: string;
    priority: string;
    rationale: string;
  }>;
  caveats: string[];
  sources_cited: string[];
  generated_at?: string;
}

interface CDSReportProps {
  report: Report;
}

const LIKELIHOOD_THEME: Record<string, { badge: string; border: string; bg: string }> = {
  "very high": {
    badge: "bg-rose-100 text-rose-800 border-rose-200",
    border: "border-l-rose-500",
    bg: "bg-rose-50/20",
  },
  high: {
    badge: "bg-rose-50 text-rose-700 border-rose-200",
    border: "border-l-rose-500",
    bg: "bg-rose-50/15",
  },
  moderate: {
    badge: "bg-amber-50 text-amber-700 border-amber-200",
    border: "border-l-amber-500",
    bg: "bg-amber-50/15",
  },
  low: {
    badge: "bg-sky-50 text-sky-700 border-sky-200",
    border: "border-l-sky-400",
    bg: "bg-sky-50/10",
  },
  "very low": {
    badge: "bg-slate-100 text-slate-600 border-slate-200",
    border: "border-l-slate-300",
    bg: "bg-slate-50/10",
  },
};

const PRIORITY_BADGES: Record<string, string> = {
  critical: "bg-rose-100 text-rose-800 border-rose-200 font-bold",
  high: "bg-amber-100 text-amber-800 border-amber-200 font-semibold",
  moderate: "bg-sky-100 text-sky-800 border-sky-200 font-medium",
  low: "bg-slate-100 text-slate-700 border-slate-200 font-normal",
};

export function CDSReport({ report }: CDSReportProps) {
  const [copied, setCopied] = useState(false);
  const [expandedDx, setExpandedDx] = useState<Record<number, boolean>>({ 0: true, 1: true });
  const [checkedActions, setCheckedActions] = useState<Record<number, boolean>>({});

  const handleCopy = () => {
    const md = reportToMarkdown(report);
    navigator.clipboard.writeText(md);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const md = reportToMarkdown(report);
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cds-report-${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const toggleDx = (idx: number) => {
    setExpandedDx((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const toggleAction = (idx: number) => {
    setCheckedActions((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <div className="space-y-6">
      {/* Top Report Header & Export Actions */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-card p-5 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 ring-4 ring-emerald-100" />
              <h2 className="text-lg font-bold text-slate-900 tracking-tight">
                Clinical Decision Support Dossier
              </h2>
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200 font-semibold">
                GRADE Verified
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Evidence-verified clinical diagnostic assessment and management pathway
            </p>
          </div>

          {/* Quick Toolbar */}
          <div className="flex items-center gap-2 self-start sm:self-auto">
            <button
              onClick={handleCopy}
              className="px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg transition-colors flex items-center gap-1.5"
              title="Copy markdown to clipboard"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? "Copied" : "Copy"}
            </button>
            <button
              onClick={handleDownload}
              className="px-3 py-1.5 text-xs font-medium text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg transition-colors flex items-center gap-1.5"
              title="Download report as Markdown file"
            >
              <Download className="w-3.5 h-3.5" />
              Export .md
            </button>
          </div>
        </div>

        {/* Patient Synthesis Strip */}
        <div className="mt-4 pt-1">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5 text-indigo-500" />
            Synthesized Clinical Presentation
          </h3>
          <p className="text-sm text-slate-800 leading-relaxed bg-slate-50/70 p-3.5 rounded-xl border border-slate-200/80">
            {report.patient_summary}
          </p>
        </div>
      </div>

      {/* Safety Critical Alerts & Contradiction Gaps */}
      {report.conflicts && report.conflicts.length > 0 && (
        <div className="bg-gradient-to-r from-rose-50/80 via-white to-rose-50/40 rounded-2xl border-2 border-rose-300 shadow-card p-5">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-7 h-7 rounded-lg bg-rose-600 text-white flex items-center justify-center">
              <ShieldAlert className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-rose-900">
                Care Gaps &amp; Clinical Conflicts Detected ({report.conflicts.length})
              </h3>
              <p className="text-[11px] text-rose-600">
                Discrepancies identified between clinical guidelines and current patient orders
              </p>
            </div>
          </div>

          <div className="mt-3 space-y-3">
            {report.conflicts.map((conflict, i) => (
              <div
                key={i}
                className="p-4 rounded-xl border border-rose-200 bg-white/95 shadow-sm space-y-2"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-bold text-rose-800 uppercase tracking-wide">
                    {conflict.conflict_type.replace(/_/g, " ")}
                  </span>
                  <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-rose-100 text-rose-700">
                    {conflict.severity} Acuity
                  </span>
                </div>

                <p className="text-xs sm:text-sm font-medium text-slate-800">
                  {conflict.description}
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs pt-1">
                  <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                    <span className="font-semibold text-slate-500 block mb-0.5">
                      Guideline Standard:
                    </span>
                    <span className="text-slate-700">{conflict.guideline_text}</span>
                    <span className="block text-[11px] text-slate-400 mt-1 italic">
                      Source: {conflict.guideline_source}
                    </span>
                  </div>
                  <div className="bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                    <span className="font-semibold text-slate-500 block mb-0.5">
                      Patient Specific Finding:
                    </span>
                    <span className="text-slate-700">{conflict.patient_data}</span>
                  </div>
                </div>

                {conflict.suggested_resolution && (
                  <div className="p-2.5 bg-emerald-50 border border-emerald-200 rounded-lg text-xs flex items-start gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold text-emerald-800">
                        Suggested Resolution:{" "}
                      </span>
                      <span className="text-emerald-900">
                        {conflict.suggested_resolution}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ranked Differential Diagnosis */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-card p-5 sm:p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-indigo-50 text-indigo-600 flex items-center justify-center">
              <Stethoscope className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Differential Diagnosis ({report.differential_diagnosis.length})
              </h3>
              <p className="text-[11px] text-slate-400">
                Ranked by clinical likelihood with diagnostic evidence mapping
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {report.differential_diagnosis.map((dx, i) => {
            const theme = LIKELIHOOD_THEME[dx.likelihood.toLowerCase()] || LIKELIHOOD_THEME.low;
            const isExpanded = !!expandedDx[i];

            return (
              <div
                key={i}
                className={`rounded-xl border border-slate-200 overflow-hidden transition-all duration-200 border-l-4 ${theme.border} ${theme.bg}`}
              >
                <div
                  onClick={() => toggleDx(i)}
                  className="p-3.5 sm:p-4 flex items-center justify-between gap-3 cursor-pointer select-none bg-white/80 hover:bg-white"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className="w-5 h-5 rounded-full bg-slate-100 text-slate-700 font-bold text-xs flex items-center justify-center shrink-0">
                      {i + 1}
                    </span>
                    <h4 className="text-sm font-bold text-slate-900 truncate">
                      {dx.diagnosis}
                    </h4>
                    {dx.icd10_code && (
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">
                        {dx.icd10_code}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <span
                      className={`text-[11px] font-semibold uppercase px-2.5 py-0.5 rounded-full border ${theme.badge}`}
                    >
                      {dx.likelihood} Likelihood
                    </span>
                    {isExpanded ? (
                      <ChevronUp className="w-4 h-4 text-slate-400" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-slate-400" />
                    )}
                  </div>
                </div>

                {isExpanded && (
                  <div className="px-4 pb-4 pt-1 bg-white border-t border-slate-100 text-xs space-y-3">
                    <div>
                      <span className="font-semibold text-slate-600 block mb-1">
                        Diagnostic Reasoning:
                      </span>
                      <p className="text-slate-700 leading-relaxed bg-slate-50 p-3 rounded-lg border border-slate-100">
                        {dx.reasoning}
                      </p>
                    </div>

                    {dx.supporting_evidence && dx.supporting_evidence.length > 0 && (
                      <div>
                        <span className="font-semibold text-slate-600 block mb-1.5">
                          Supporting Patient Evidence:
                        </span>
                        <div className="flex flex-wrap gap-1.5">
                          {dx.supporting_evidence.map((ev, j) => (
                            <span
                              key={j}
                              className="text-[11px] bg-indigo-50 text-indigo-700 px-2 py-0.8 rounded-md border border-indigo-100"
                            >
                              ✓ {ev}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Suggested Next Steps / Clinical Workup Grid */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-card p-5 sm:p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
              <CheckCircle2 className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Recommended Workup &amp; Interventions ({report.suggested_next_steps.length})
              </h3>
              <p className="text-[11px] text-slate-400">
                Interactive clinical action items prioritized by acuity
              </p>
            </div>
          </div>
        </div>

        <div className="space-y-2.5">
          {report.suggested_next_steps.map((step, i) => {
            const isChecked = !!checkedActions[i];
            const badgeClass =
              PRIORITY_BADGES[step.priority.toLowerCase()] || PRIORITY_BADGES.moderate;

            return (
              <div
                key={i}
                onClick={() => toggleAction(i)}
                className={`p-3.5 rounded-xl border transition-all duration-150 flex items-start gap-3 cursor-pointer ${
                  isChecked
                    ? "bg-slate-50/70 border-slate-200 opacity-60 line-through"
                    : "bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/40"
                }`}
              >
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => {}}
                  className="mt-1 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                />

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className={`text-[10px] uppercase px-2 py-0.5 rounded-md border ${badgeClass}`}
                    >
                      {step.priority}
                    </span>
                    <span className="text-xs sm:text-sm font-semibold text-slate-900">
                      {step.action}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 leading-relaxed">{step.rationale}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Drug Interactions Matrix */}
      {report.drug_interaction_warnings && report.drug_interaction_warnings.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-card p-5 sm:p-6">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-7 h-7 rounded-lg bg-amber-50 text-amber-600 flex items-center justify-center">
              <Pill className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Drug-Drug Interaction Warnings ({report.drug_interaction_warnings.length})
              </h3>
              <p className="text-[11px] text-slate-400">
                OpenFDA &amp; RxNorm interaction audit
              </p>
            </div>
          </div>

          <div className="space-y-2.5">
            {report.drug_interaction_warnings.map((ix, i) => (
              <div
                key={i}
                className="p-3.5 rounded-xl border border-amber-200/90 bg-amber-50/30 space-y-1.5"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800">
                    💊 {ix.drug_a} ↔ {ix.drug_b}
                  </span>
                  <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded bg-amber-100 text-amber-800 border border-amber-200">
                    {ix.severity} Severity
                  </span>
                </div>
                <p className="text-xs text-slate-700 leading-relaxed">{ix.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Guideline Recommendations & Literature Sources */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Guidelines */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-card p-5">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-indigo-500" />
            Guideline-Concordant Pathways
          </h3>
          <ul className="space-y-2 text-xs text-slate-700">
            {report.guideline_recommendations && report.guideline_recommendations.length > 0 ? (
              report.guideline_recommendations.map((rec, i) => (
                <li key={i} className="flex items-start gap-2 bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                  <span className="text-indigo-600 font-bold shrink-0">•</span>
                  <span className="leading-relaxed">{rec}</span>
                </li>
              ))
            ) : (
              <li className="text-slate-400 italic">No specific guideline recommendations cited.</li>
            )}
          </ul>
        </div>

        {/* Sources */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-card p-5">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
            <ExternalLink className="w-3.5 h-3.5 text-indigo-500" />
            Referenced Medical Literature
          </h3>
          <ul className="space-y-2 text-xs text-slate-600">
            {report.sources_cited && report.sources_cited.length > 0 ? (
              report.sources_cited.map((source, i) => (
                <li key={i} className="flex items-center gap-2 bg-slate-50 p-2.5 rounded-lg border border-slate-100 truncate">
                  <span className="text-slate-400 font-mono text-[10px]">[{i + 1}]</span>
                  <span className="truncate">{source}</span>
                </li>
              ))
            ) : (
              <li className="text-slate-400 italic">No specific literature references indexed.</li>
            )}
          </ul>
        </div>
      </div>

      {/* Disclaimers & Regulatory Notice */}
      <div className="bg-slate-50 rounded-2xl border border-slate-200/80 p-4 text-xs text-slate-500 space-y-1.5">
        <div className="flex items-center gap-1.5 font-semibold text-slate-700">
          <Info className="w-3.5 h-3.5 text-slate-500" />
          Clinical Decision Support Disclaimer &amp; Caveats
        </div>
        {report.caveats && report.caveats.length > 0 ? (
          report.caveats.map((c, i) => (
            <p key={i} className="text-[11px] text-slate-500 leading-relaxed">
              • {c}
            </p>
          ))
        ) : (
          <p className="text-[11px] text-slate-500 leading-relaxed">
            This AI-assisted report is designed to augment clinical judgment and should not be used as the sole basis for diagnostic or therapeutic decisions.
          </p>
        )}
      </div>
    </div>
  );
}
