import type { SourceChunk, DocType, KnowledgeDoc } from "./mockData";

interface BackendSource {
  n: number;
  source: string;
  section: string;
  snippet: string;
  similarity: number | null;
  kind?: string;
  title?: string | null;
}

function mapDocType(source: string): DocType {
  const lower = source.toLowerCase();
  if (lower.includes("act")) return "statutory";
  if (lower.includes("final_tax_law_book")) return "advisory";
  return "planning";
}

export function mapSources(srcs: BackendSource[]): SourceChunk[] {
  return srcs.map((s) => {
    if (s.kind === "web") {
      return {
        id: `s${s.n}`,
        docId: s.source,
        docTitle: s.title || s.source,
        docType: "web",
        excerpt: s.snippet,
        score: s.similarity ?? 0,
        section: s.section || undefined,
      };
    }
    const filename = s.source.split(/[/\\]/).pop() ?? s.source;
    const title = filename.replace(/\.[^.]+$/, "");
    return {
      id: `s${s.n}`,
      docId: s.source,
      docTitle: title,
      docType: mapDocType(s.source),
      excerpt: s.snippet,
      score: s.similarity ?? 0,
      section: s.section || undefined,
    };
  });
}

export type DoneInfo = {
  model?: string;
  tokens?: number;
  tokensPerSec?: number;
};

type StreamCallbacks = {
  onRoute?: (route: string, reason: string) => void;
  onTrace?: (line: string) => void;
  onSubqueries?: (items: string[]) => void;
  onSources?: (sources: BackendSource[]) => void;
  onThinking?: (text: string) => void;
  onThinkingDone?: (seconds: number) => void;
  onToken?: (text: string) => void;
  onDone?: (info: DoneInfo) => void;
  onError?: (msg: string) => void;
};

export async function streamQuery(question: string, callbacks: StreamCallbacks): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/api/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    callbacks.onError?.("Network error — is the backend running?");
    return;
  }

  const ctype = res.headers.get("content-type") ?? "";
  if (!res.ok || !res.body || ctype.includes("application/json")) {
    let msg = `HTTP ${res.status}`;
    try {
      const d = await res.json() as Record<string, unknown>;
      if (d?.error) msg = String(d.error);
    } catch {}
    callbacks.onError?.(msg);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const payload = raw.replace(/^data: ?/, "").trim();
      if (!payload) continue;
      let evt: Record<string, unknown>;
      try {
        evt = JSON.parse(payload) as Record<string, unknown>;
      } catch {
        continue;
      }
      switch (evt.type) {
        case "route":
          callbacks.onRoute?.(evt.route as string, (evt.reason as string) ?? "");
          break;
        case "trace":
          callbacks.onTrace?.(evt.line as string);
          break;
        case "subqueries":
          callbacks.onSubqueries?.(evt.items as string[]);
          break;
        case "sources":
          callbacks.onSources?.(evt.sources as BackendSource[]);
          break;
        case "thinking":
          callbacks.onThinking?.(evt.text as string);
          break;
        case "thinking_done":
          callbacks.onThinkingDone?.(evt.seconds as number);
          break;
        case "token":
          callbacks.onToken?.(evt.text as string);
          break;
        case "done":
          callbacks.onDone?.({
            model: evt.model as string | undefined,
            tokens: evt.tokens as number | undefined,
            tokensPerSec: (evt.tokens_per_sec as number | null | undefined) ?? undefined,
          });
          return;
        case "error":
          callbacks.onError?.(evt.message as string);
          return;
      }
    }
  }
  // Stream ended without explicit done event
  callbacks.onDone?.({});
}

// --------------------------------------------------------------------------- #
// Documents API
// --------------------------------------------------------------------------- #
interface BackendDoc {
  filename: string;
  title: string;
  doc_type: string;
  exam_board: string;
  chunks: number;
  size_bytes: number;
  modified_ts: number;
  preview: string;
}

interface DocsResponse {
  documents: BackendDoc[];
  total_chunks: number;
  index_ready: boolean;
}

function mapBackendDocType(doc_type: string, filename: string): DocType {
  if (doc_type === "act" || filename.toLowerCase().includes("act")) return "statutory";
  if (doc_type === "study_book") return "advisory";
  return "planning";
}

export async function fetchDocuments(): Promise<{ docs: KnowledgeDoc[]; totalChunks: number }> {
  const res = await fetch("/api/documents");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as DocsResponse;
  const docs: KnowledgeDoc[] = data.documents.map((d) => ({
    id: d.filename,
    title: d.title,
    docType: mapBackendDocType(d.doc_type, d.filename),
    chunks: d.chunks,
    ingestedAt: Math.round(d.modified_ts * 1000),
    sourceFile: d.filename,
    preview: d.preview,
  }));
  return { docs, totalChunks: data.total_chunks };
}

// --------------------------------------------------------------------------- #
// Analytics API
// --------------------------------------------------------------------------- #
interface AnalyticsResponse {
  total_docs: number;
  total_chunks: number;
  queries_today: number;
  avg_response_ms: number;
  query_volume: { day: string; queries: number }[];
  doc_type_split: { name: string; value: number }[];
  recent_queries: {
    query: string;
    ts: number;
    ms: number;
    sources: number;
    feedback: "up" | "down" | null;
  }[];
  avg_relevance: number;
  hit_rate: number;
}

const DOC_TYPE_LABELS: Record<string, string> = {
  study_chapter: "Study Chapters",
  act: "Income Tax Act",
  study_book: "Study Book",
};

const DOC_TYPE_COLORS: Record<string, string> = {
  study_chapter: "oklch(0.46 0.11 50)",
  act: "oklch(0.55 0.14 230)",
  study_book: "oklch(0.52 0.13 160)",
};

export type AnalyticsData = {
  totalDocs: number;
  totalChunks: number;
  queriesToday: number;
  avgResponseMs: number;
  queryVolume: { day: string; queries: number }[];
  docTypeSplit: { name: string; value: number; color: string }[];
  recentQueries: {
    query: string;
    ts: number;
    ms: number;
    sources: number;
    feedback: "up" | "down" | null;
  }[];
  avgRelevance: number;
  hitRate: number;
};

export async function fetchAnalytics(): Promise<AnalyticsData> {
  const res = await fetch("/api/analytics");
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = (await res.json()) as AnalyticsResponse;
  return {
    totalDocs: data.total_docs,
    totalChunks: data.total_chunks,
    queriesToday: data.queries_today,
    avgResponseMs: data.avg_response_ms,
    queryVolume: data.query_volume,
    docTypeSplit: data.doc_type_split.map((s) => ({
      name: DOC_TYPE_LABELS[s.name] ?? s.name,
      value: s.value,
      color: DOC_TYPE_COLORS[s.name] ?? "oklch(0.5 0.1 50)",
    })),
    recentQueries: data.recent_queries,
    avgRelevance: data.avg_relevance,
    hitRate: data.hit_rate,
  };
}
