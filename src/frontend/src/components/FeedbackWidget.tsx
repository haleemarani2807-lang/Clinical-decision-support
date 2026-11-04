"use client";

import { useState, useRef, useEffect } from "react";
import { MessageSquare, X, Send, CheckCircle2, HeartHandshake, Star } from "lucide-react";

export function FeedbackWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [contact, setContact] = useState("");
  const [rating, setRating] = useState<number | null>(null);
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const panelRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [isOpen]);

  const handleSubmit = async () => {
    if (!message.trim()) return;
    setStatus("sending");

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: `${rating ? `[Rating: ${rating}/5] ` : ""}${message.trim()}`,
          contact: contact.trim() || undefined,
          page_url: window.location.href,
          user_agent: navigator.userAgent,
        }),
      });

      if (res.ok) {
        setStatus("sent");
        setMessage("");
        setContact("");
        setRating(null);
        setTimeout(() => {
          setIsOpen(false);
          setStatus("idle");
        }, 2200);
      } else {
        setStatus("error");
      }
    } catch {
      setStatus("error");
    }
  };

  return (
    <div className="fixed bottom-12 right-6 z-50" ref={panelRef}>
      {/* Feedback panel */}
      {isOpen && (
        <div className="mb-3 w-88 bg-white rounded-2xl shadow-float border border-slate-200 overflow-hidden animate-slide-up">
          <div className="bg-gradient-to-r from-indigo-600 to-indigo-700 px-5 py-4 flex items-center justify-between text-white">
            <div className="flex items-center gap-2">
              <HeartHandshake className="w-5 h-5 text-indigo-200" />
              <div>
                <h3 className="font-bold text-sm">Clinician Feedback</h3>
                <p className="text-[11px] text-indigo-100">Help improve reasoning quality &amp; safety</p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="p-1 rounded-lg hover:bg-white/10 text-white/80 hover:text-white transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="p-5 space-y-3.5">
            {status === "sent" ? (
              <div className="text-center py-6 space-y-2">
                <div className="w-10 h-10 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center mx-auto">
                  <CheckCircle2 className="w-6 h-6" />
                </div>
                <h4 className="text-sm font-bold text-slate-800">Feedback Submitted</h4>
                <p className="text-xs text-slate-500">Thank you for contributing to clinical evaluation!</p>
              </div>
            ) : (
              <>
                {/* Rating selection */}
                <div>
                  <label className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1.5">
                    How was the reasoning accuracy?
                  </label>
                  <div className="flex gap-1.5">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        type="button"
                        onClick={() => setRating(star)}
                        className={`p-1.5 rounded-lg border text-xs font-semibold flex-1 transition-all ${
                          rating === star
                            ? "bg-indigo-50 border-indigo-500 text-indigo-700 ring-2 ring-indigo-500/10"
                            : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100"
                        }`}
                      >
                        ★ {star}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1.5">
                    Clinical Comments or Observations
                  </label>
                  <textarea
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Enter impressions on differential ranking, workup accuracy, or missing contraindications..."
                    className="w-full px-3.5 py-2.5 border border-slate-200 rounded-xl text-xs sm:text-sm text-slate-800 resize-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all placeholder:text-slate-400"
                    rows={3}
                    maxLength={1000}
                    autoFocus
                  />
                </div>

                <div>
                  <input
                    type="text"
                    value={contact}
                    onChange={(e) => setContact(e.target.value)}
                    placeholder="Your email or institution (optional)"
                    className="w-full px-3.5 py-2 border border-slate-200 rounded-xl text-xs text-slate-800 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 outline-none transition-all placeholder:text-slate-400"
                    maxLength={200}
                  />
                </div>

                {status === "error" && (
                  <p className="text-xs text-rose-600 bg-rose-50 p-2 rounded-lg border border-rose-200">
                    Failed to submit feedback. Check that backend is running.
                  </p>
                )}

                <button
                  onClick={handleSubmit}
                  disabled={!message.trim() || status === "sending"}
                  className={`w-full py-2.5 px-4 rounded-xl text-xs font-semibold text-white transition-all flex items-center justify-center gap-2 ${
                    !message.trim() || status === "sending"
                      ? "bg-slate-300 cursor-not-allowed text-slate-500"
                      : "bg-indigo-600 hover:bg-indigo-700 active:scale-[0.99] shadow-sm"
                  }`}
                >
                  <Send className="w-3.5 h-3.5" />
                  {status === "sending" ? "Submitting..." : "Send Clinical Feedback"}
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Floating Pill Trigger */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2.5 rounded-full bg-slate-900 text-white hover:bg-indigo-600 shadow-float transition-all duration-200 group active:scale-95 border border-slate-700/50"
      >
        <MessageSquare className="w-4 h-4 text-indigo-400 group-hover:text-white transition-colors" />
        <span className="text-xs font-semibold">Feedback</span>
      </button>
    </div>
  );
}
