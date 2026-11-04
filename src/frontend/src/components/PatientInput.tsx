"use client";

import { useState } from "react";
import {
  Heart,
  Activity,
  Flame,
  Stethoscope,
  Sparkles,
  RotateCcw,
  Sliders,
  ShieldCheck,
  CheckCircle2,
  FileText,
  AlertTriangle,
  ArrowRight,
} from "lucide-react";

export const SAMPLE_CASES = [
  {
    id: "adhf",
    label: "Acute Heart Failure (68F)",
    badge: "Cardiology · High Acuity",
    badgeColor: "bg-rose-50 text-rose-700 border-rose-200",
    icon: Heart,
    summary: "68F with 3-day orthopnea, PND, 8lb weight gain, and bilateral 3+ pitting edema.",
    text: `68-year-old female presenting with worsening shortness of breath over 3 days, now dyspneic at rest.
Associated with bilateral lower extremity edema, orthopnea (uses 3 pillows), and paroxysmal nocturnal dyspnea (PND).
Weight gain of 8 lbs in the past week.

PMH: CHF (reduced EF 30%), Atrial fibrillation, CKD Stage 3, COPD
Medications: Carvedilol 25mg BID, Furosemide 40mg daily, Warfarin 5mg daily, Lisinopril 10mg daily, Albuterol PRN
Allergies: Sulfa drugs (rash)

Vitals: BP 142/88, HR 110 irregular, Temp 98.2°F, RR 28, SpO2 88% on room air
Labs: BNP 1850 pg/mL (ref <100), Creatinine 2.1 (baseline 1.5), K+ 5.2, Hgb 10.2
CXR: Bilateral pleural effusions, cardiomegaly, pulmonary vascular congestion`,
  },
  {
    id: "acs",
    label: "Acute Coronary Syndrome (55M)",
    badge: "Emergency · Critical",
    badgeColor: "bg-red-50 text-red-700 border-red-200",
    icon: Flame,
    summary: "55M with 2hr crushing substernal chest pain radiating to left arm with diaphoresis.",
    text: `55-year-old male presenting to the ED with acute onset substernal chest pain for the past 2 hours.
Pain is described as pressure-like, 8/10 severity, radiating to left arm and jaw. Associated with diaphoresis and nausea.

PMH: Hypertension, Type 2 Diabetes, Hyperlipidemia, Former smoker (30 pack-years, quit 5 years ago)
Medications: Metformin 1000mg BID, Lisinopril 20mg daily, Atorvastatin 40mg daily, Aspirin 81mg daily
Allergies: Penicillin (anaphylaxis)

Vitals: BP 165/95, HR 98, Temp 98.6°F, RR 22, SpO2 95% on room air
Labs: Troponin I 0.45 ng/mL (ref <0.04), BMP normal, CBC unremarkable
ECG: ST elevation in leads II, III, aVF`,
  },
  {
    id: "dka",
    label: "Diabetic Ketoacidosis (42M)",
    badge: "Endocrinology · Urgent",
    badgeColor: "bg-amber-50 text-amber-700 border-amber-200",
    icon: Activity,
    summary: "42M altered mental status, missed insulin for 48h, Kussmaul respirations, BG 485.",
    text: `42-year-old male brought by EMS with altered mental status. Found confused at home by family.
History of poorly controlled Type 1 Diabetes. Has not been taking insulin for 2 days due to running out.
Reports polyuria, polydipsia, abdominal pain, and vomiting for the past day.

PMH: Type 1 Diabetes (diagnosed age 12), Depression
Medications: Insulin glargine 30 units nightly, Insulin lispro sliding scale, Sertraline 100mg daily
Allergies: NKDA

Vitals: BP 100/60, HR 120, Temp 99.1°F, RR 32 (Kussmaul breathing), SpO2 98%
Labs: Glucose 485 mg/dL, pH 7.15, Bicarb 8, Anion gap 28, Beta-hydroxybutyrate 6.2, K+ 5.8, Na+ 128, BUN 35, Cr 1.8
Urinalysis: Large ketones, glucose >1000`,
  },
  {
    id: "cap",
    label: "Severe Pneumonia (74F)",
    badge: "Pulmonology · Inpatient",
    badgeColor: "bg-sky-50 text-sky-700 border-sky-200",
    icon: Stethoscope,
    summary: "74F with fever, productive rust-colored cough, right lower lobe crackles, SpO2 89%.",
    text: `74-year-old female presenting from nursing facility with 4 days of high fever, chills, and productive rust-colored cough.
Daughter notes increasing confusion and lethargy since yesterday morning.

PMH: Osteoarthritis, Mild cognitive impairment, Osteoporosis
Medications: Acetaminophen 500mg TID, Calcium with Vitamin D, Donepezil 10mg daily
Allergies: Codeine (nausea)

Vitals: BP 96/60, HR 108, Temp 102.4°F, RR 26, SpO2 89% on room air
Labs: WBC 18,400 with 15% bands, Lactate 2.4 mmol/L, BUN 28, Creatinine 1.2
CXR: Dense right lower lobe alveolar consolidation with air bronchograms`,
  },
];

interface PatientInputProps {
  onSubmit: (patientText: string, options: { useLangGraph: boolean; includeDrugs: boolean; includeGuidelines: boolean }) => void;
  isLoading: boolean;
}

export function PatientInput({ onSubmit, isLoading }: PatientInputProps) {
  const [text, setText] = useState("");
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [useLangGraph, setUseLangGraph] = useState(true);
  const [includeDrugs, setIncludeDrugs] = useState(true);
  const [includeGuidelines, setIncludeGuidelines] = useState(true);
  const [showConfig, setShowConfig] = useState(false);

  const handleSelectCase = (caseObj: typeof SAMPLE_CASES[0]) => {
    setSelectedCaseId(caseObj.id);
    setText(caseObj.text);
  };

  const handleClear = () => {
    setText("");
    setSelectedCaseId(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !isLoading) {
      onSubmit(text.trim(), {
        useLangGraph,
        includeDrugs,
        includeGuidelines,
      });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (text.trim() && !isLoading) {
        onSubmit(text.trim(), {
          useLangGraph,
          includeDrugs,
          includeGuidelines,
        });
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Sample Cases Selector Header */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
            Clinical Test Vignettes
          </label>
          <span className="text-xs text-slate-400">Select to pre-populate</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {SAMPLE_CASES.map((sample) => {
            const Icon = sample.icon;
            const isSelected = selectedCaseId === sample.id;
            return (
              <button
                key={sample.id}
                type="button"
                onClick={() => handleSelectCase(sample)}
                className={`text-left p-3.5 rounded-xl border transition-all duration-200 group flex items-start gap-3 ${
                  isSelected
                    ? "bg-indigo-50/70 border-indigo-500 ring-2 ring-indigo-500/20 shadow-sm"
                    : "bg-white border-slate-200/80 hover:border-slate-300 hover:bg-slate-50/70 shadow-card"
                }`}
              >
                <div
                  className={`p-2 rounded-lg shrink-0 transition-colors ${
                    isSelected
                      ? "bg-indigo-600 text-white"
                      : "bg-slate-100 text-slate-600 group-hover:bg-indigo-100 group-hover:text-indigo-600"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-1 mb-1">
                    <h4 className="text-xs font-semibold text-slate-900 truncate">
                      {sample.label}
                    </h4>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded-md border font-medium whitespace-nowrap ${sample.badgeColor}`}
                    >
                      {sample.badge.split(" · ")[0]}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 line-clamp-2 leading-relaxed">
                    {sample.summary}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Clinical Note Input Area */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-card p-5 transition-shadow focus-within:shadow-soft focus-within:border-indigo-400">
        <div className="flex items-center justify-between mb-3 pb-2.5 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-slate-900">
                Patient Clinical Description / EHR Note
              </h3>
              <p className="text-[11px] text-slate-400">
                Unstructured HPI, physical exam, vitals, labs, and active medication list
              </p>
            </div>
          </div>

          {text && (
            <button
              type="button"
              onClick={handleClear}
              className="text-xs text-slate-400 hover:text-slate-600 flex items-center gap-1 px-2 py-1 rounded-md hover:bg-slate-100 transition-colors"
              title="Clear note input"
            >
              <RotateCcw className="w-3 h-3" />
              Clear
            </button>
          )}
        </div>

        <div className="relative">
          <textarea
            id="patient-text"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              if (selectedCaseId) setSelectedCaseId(null);
            }}
            onKeyDown={handleKeyDown}
            rows={13}
            className="w-full p-3.5 bg-slate-50/50 rounded-xl border border-slate-200/90 text-slate-800 text-xs sm:text-sm font-mono leading-relaxed focus:bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all placeholder:text-slate-400 resize-y"
            placeholder="Paste patient clinical case text here (e.g., 68yo female with acute dyspnea at rest, orthopnea, bilateral lower extremity edema, labs, vitals)..."
          />
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
          <div className="flex items-center gap-3">
            <span>
              <strong>{text.trim() ? text.trim().split(/\s+/).length : 0}</strong> words
            </span>
            <span className="text-slate-300">|</span>
            <span>
              <strong>{text.length}</strong> characters
            </span>
          </div>
          <span className="text-[11px] text-slate-400 hidden sm:inline">
            Press <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-slate-600 border border-slate-200 font-mono text-[10px]">Ctrl</kbd> + <kbd className="px-1.5 py-0.5 bg-slate-100 rounded text-slate-600 border border-slate-200 font-mono text-[10px]">Enter</kbd> to execute
          </span>
        </div>
      </div>

      {/* Advanced Orchestration & Pipeline Controls */}
      <div className="bg-slate-50/80 rounded-xl border border-slate-200/80 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-indigo-600" />
            <span className="text-xs font-semibold text-slate-800">
              Orchestrator Pipeline Configuration
            </span>
            <span className="text-[10px] bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full font-semibold">
              {useLangGraph ? "Track B: LangGraph Multi-Agent (10-Step)" : "Track A: Baseline Linear (6-Step)"}
            </span>
          </div>

          <button
            type="button"
            onClick={() => setShowConfig(!showConfig)}
            className="text-xs text-indigo-600 hover:text-indigo-800 font-medium"
          >
            {showConfig ? "Collapse Options" : "Customize Tools"}
          </button>
        </div>

        {/* Toggle options */}
        <div className={`mt-3 pt-3 border-t border-slate-200/60 grid grid-cols-1 sm:grid-cols-3 gap-3 ${showConfig ? "block" : "hidden sm:grid"}`}>
          {/* Mode Selector */}
          <label className="flex items-start gap-2.5 p-2 rounded-lg bg-white border border-slate-200 cursor-pointer hover:border-indigo-300 transition-colors">
            <input
              type="checkbox"
              checked={useLangGraph}
              onChange={(e) => setUseLangGraph(e.target.checked)}
              className="mt-0.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
            />
            <div>
              <span className="text-xs font-medium text-slate-900 block">LangGraph DAG</span>
              <span className="text-[10px] text-slate-500 block leading-tight">GRADE & claim NLI check</span>
            </div>
          </label>

          {/* Drug Interactions */}
          <label className="flex items-start gap-2.5 p-2 rounded-lg bg-white border border-slate-200 cursor-pointer hover:border-indigo-300 transition-colors">
            <input
              type="checkbox"
              checked={includeDrugs}
              onChange={(e) => setIncludeDrugs(e.target.checked)}
              className="mt-0.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
            />
            <div>
              <span className="text-xs font-medium text-slate-900 block">Drug Interactions</span>
              <span className="text-[10px] text-slate-500 block leading-tight">OpenFDA & RxNorm audit</span>
            </div>
          </label>

          {/* Guideline RAG */}
          <label className="flex items-start gap-2.5 p-2 rounded-lg bg-white border border-slate-200 cursor-pointer hover:border-indigo-300 transition-colors">
            <input
              type="checkbox"
              checked={includeGuidelines}
              onChange={(e) => setIncludeGuidelines(e.target.checked)}
              className="mt-0.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
            />
            <div>
              <span className="text-xs font-medium text-slate-900 block">Evidence Retrieval</span>
              <span className="text-[10px] text-slate-500 block leading-tight">PubMed + Vector Corpus</span>
            </div>
          </label>
        </div>
      </div>

      {/* Submit Action Button */}
      <div className="flex items-center justify-end gap-3 pt-2">
        <button
          type="submit"
          disabled={!text.trim() || isLoading}
          className={`w-full sm:w-auto px-6 py-3.5 rounded-xl font-medium text-sm text-white shadow-soft transition-all duration-200 flex items-center justify-center gap-2 ${
            !text.trim() || isLoading
              ? "bg-slate-300 cursor-not-allowed text-slate-500 shadow-none"
              : "bg-indigo-600 hover:bg-indigo-700 active:scale-[0.99] shadow-indigo-500/25 hover:shadow-indigo-500/35"
          }`}
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Initializing Agent Pipeline...</span>
            </>
          ) : (
            <>
              <span>Execute Clinical Decision Support</span>
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>
    </form>
  );
}
