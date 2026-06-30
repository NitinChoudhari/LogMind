export type DocType = "statutory" | "planning" | "advisory" | "web";

export interface SourceChunk {
  id: string;
  docId: string;
  docTitle: string;
  docType: DocType;
  excerpt: string;
  score: number;
  section?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
  feedback?: "up" | "down" | null;
  createdAt: number;
  /** Manager route for this turn — drives the thinking-timeline search step. */
  route?: "kb" | "web" | "general";
  /** Router's one-line reason — headline fallback when there's no reasoning text. */
  reason?: string;
  /** Planned sub-queries (kb route), shown under the search step. */
  subQueries?: string[];
  /** Model identifier reported by the backend (e.g. "gpt-4o-mini", a HF repo id). */
  model?: string;
  /** Raw <think> reasoning text, only present for providers that emit one. */
  thinking?: string;
  /** Seconds spent in the thinking phase, reported by the backend. */
  thinkingSeconds?: number;
  /** Total wall-clock seconds for the request, measured client-side. */
  durationSeconds?: number;
  /** Answer token count, reported by the backend (exact for huggingface, tiktoken-estimated otherwise). */
  tokens?: number;
  /** Generation throughput (tokens / generation seconds), reported by the backend. */
  tokensPerSec?: number;
}

export interface ChatThreadData {
  id: string;
  title: string;
  updatedAt: number;
  messages: ChatMessage[];
}

export interface KnowledgeDoc {
  id: string;
  title: string;
  docType: DocType;
  chunks: number;
  ingestedAt: number;
  sourceFile: string;
  section?: string;
  preview: string;
}

const now = 1719360000000;
const day = 86400000;

export const SAMPLE_SOURCES: SourceChunk[] = [
  {
    id: "s1",
    docId: "d1",
    docTitle: "Income Tax Act, 1961 — Section 80C",
    docType: "statutory",
    section: "Chapter VI-A",
    score: 0.92,
    excerpt:
      "In computing the total income of an assessee, being an individual or a Hindu undivided family, there shall be deducted, in accordance with and subject to the provisions of this section, the whole of the amount paid or deposited in the previous year, as does not exceed one hundred and fifty thousand rupees...",
  },
  {
    id: "s2",
    docId: "d2",
    docTitle: "ICAI Tax Planning Guide — Chapter 4: Salaried Individuals",
    docType: "planning",
    section: "4.2 Investment-linked deductions",
    score: 0.87,
    excerpt:
      "Salaried taxpayers should evaluate the optimal mix of ELSS, PPF, and EPF contributions to maximise the ₹1.5L 80C ceiling while balancing liquidity needs. ELSS offers the shortest lock-in (3 years) with equity-linked returns...",
  },
  {
    id: "s3",
    docId: "d3",
    docTitle: "HRA Exemption Advisory — CBDT Circular Reference",
    docType: "advisory",
    section: "Rule 2A",
    score: 0.81,
    excerpt:
      "HRA exemption is computed as the least of: (a) actual HRA received, (b) 50% of salary for metro / 40% for non-metro, (c) rent paid minus 10% of salary. Salary here includes basic + DA (forming part of retirement benefits)...",
  },
  {
    id: "s4",
    docId: "d4",
    docTitle: "Income Tax Act — Section 24(b) Home Loan Interest",
    docType: "statutory",
    section: "Income from House Property",
    score: 0.78,
    excerpt:
      "Where the property has been acquired, constructed, repaired, renewed or reconstructed with borrowed capital, the amount of any interest payable on such capital shall be allowed as a deduction, subject to a ceiling of ₹2,00,000 for self-occupied property...",
  },
];

export const SAMPLE_DOCS: KnowledgeDoc[] = [
  {
    id: "d1",
    title: "Income Tax Act, 1961 — Section 80C",
    docType: "statutory",
    chunks: 24,
    ingestedAt: now - 12 * day,
    sourceFile: "ITA-1961-S80C.pdf",
    section: "Chapter VI-A",
    preview: "Deductions in respect of LIC premium, PF, ELSS, tuition fees up to ₹1.5L cap.",
  },
  {
    id: "d2",
    title: "ICAI Tax Planning Guide — Chapter 4",
    docType: "planning",
    chunks: 38,
    ingestedAt: now - 8 * day,
    sourceFile: "icai-tax-planning-2024-ch4.pdf",
    section: "Salaried Individuals",
    preview: "Investment optimisation, regime selection, salary structuring for salaried taxpayers.",
  },
  {
    id: "d3",
    title: "HRA Exemption Advisory",
    docType: "advisory",
    chunks: 9,
    ingestedAt: now - 5 * day,
    sourceFile: "hra-advisory-2024.md",
    section: "Rule 2A",
    preview: "Calculation methodology, documentation requirements, common pitfalls for HRA claims.",
  },
  {
    id: "d4",
    title: "Income Tax Act — Section 24(b)",
    docType: "statutory",
    chunks: 14,
    ingestedAt: now - 18 * day,
    sourceFile: "ITA-1961-S24.pdf",
    section: "House Property",
    preview: "Home loan interest deduction up to ₹2L for self-occupied property.",
  },
  {
    id: "d5",
    title: "New vs Old Tax Regime — Comparative Planning Note",
    docType: "planning",
    chunks: 17,
    ingestedAt: now - 3 * day,
    sourceFile: "regime-comparison-fy25.pdf",
    preview: "Break-even analysis across income brackets between old and new regimes.",
  },
  {
    id: "d6",
    title: "TDS on Rent — Section 194-IB Advisory",
    docType: "advisory",
    chunks: 6,
    ingestedAt: now - 22 * day,
    sourceFile: "tds-194ib.txt",
    preview: "Individual tenants paying rent > ₹50k/month must deduct 5% TDS once per year.",
  },
  {
    id: "d7",
    title: "Capital Gains — Section 54 / 54F Exemptions",
    docType: "statutory",
    chunks: 21,
    ingestedAt: now - 30 * day,
    sourceFile: "ITA-1961-S54.pdf",
    preview: "Reinvestment in residential property to claim LTCG exemption.",
  },
];

export const SAMPLE_THREADS: ChatThreadData[] = [];

export const ANALYTICS = {
  totalDocs: SAMPLE_DOCS.length,
  totalChunks: SAMPLE_DOCS.reduce((s, d) => s + d.chunks, 0),
  queriesToday: 47,
  avgResponseMs: 1840,
  queryVolume: Array.from({ length: 14 }, (_, i) => ({
    day: new Date(now - (13 - i) * day).toLocaleDateString("en", { month: "short", day: "numeric" }),
    queries: Math.round(20 + Math.random() * 40 + i * 1.4),
  })),
  docTypeSplit: [
    { name: "Statutory", value: SAMPLE_DOCS.filter((d) => d.docType === "statutory").length, color: "var(--statutory)" },
    { name: "Planning", value: SAMPLE_DOCS.filter((d) => d.docType === "planning").length, color: "var(--planning)" },
    { name: "Advisory", value: SAMPLE_DOCS.filter((d) => d.docType === "advisory").length, color: "var(--advisory)" },
  ],
  recentQueries: [
    { ts: now - 5 * 60000, query: "Section 80C limit and qualifying instruments", ms: 1620, feedback: "up", sources: 3 },
    { ts: now - 22 * 60000, query: "HRA exemption for Bangalore rent ₹28k", ms: 2010, feedback: "up", sources: 2 },
    { ts: now - 51 * 60000, query: "TDS rate on rent above 50k", ms: 1480, feedback: null, sources: 2 },
    { ts: now - 95 * 60000, query: "Capital gains exemption under Section 54", ms: 2240, feedback: "up", sources: 4 },
    { ts: now - 140 * 60000, query: "Old vs new regime breakeven point", ms: 1980, feedback: "down", sources: 3 },
    { ts: now - 200 * 60000, query: "NPS additional 50k deduction eligibility", ms: 1520, feedback: "up", sources: 2 },
  ] as { ts: number; query: string; ms: number; feedback: "up" | "down" | null; sources: number }[],
  avgRelevance: 0.84,
  hitRate: 0.91,
};
