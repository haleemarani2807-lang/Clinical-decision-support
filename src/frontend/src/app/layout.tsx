import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Aegis CDSS — Evidence-Verified Clinical Decision Support",
  description:
    "Production-grade agentic clinical decision support system with GRADE evidence grading, atomic claim verification, and calibrated abstention.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%234f46e5' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3'/><path d='M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4'/><circle cx='20' cy='10' r='2'/></svg>" />
      </head>
      <body className="bg-slate-50 text-slate-900 antialiased selection:bg-indigo-100 selection:text-indigo-900">
        {children}
      </body>
    </html>
  );
}
