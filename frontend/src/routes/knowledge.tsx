import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Upload, Search, FileText, ChevronDown, ChevronRight, X, Loader2, AlertCircle } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { DocBadge } from "@/components/DocBadge";
import { fetchDocuments } from "@/lib/api";
import type { DocType, KnowledgeDoc } from "@/lib/mockData";

type Filter = "all" | DocType;

const STAGES = ["Parsing", "Chunking", "Embedding", "Indexing"] as const;

export function KnowledgePage() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["documents"],
    queryFn: fetchDocuments,
    staleTime: 30_000,
  });

  // Locally-added docs from the simulated upload modal
  const [uploadedDocs, setUploadedDocs] = useState<KnowledgeDoc[]>([]);
  const docs = [...uploadedDocs, ...(data?.docs ?? [])];
  const totalChunks = (data?.totalChunks ?? 0) + uploadedDocs.reduce((s, d) => s + d.chunks, 0);

  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [ingestStage, setIngestStage] = useState<number | null>(null);

  const filtered = useMemo(
    () =>
      docs.filter(
        (d) =>
          (filter === "all" || d.docType === filter) &&
          (query === "" ||
            d.title.toLowerCase().includes(query.toLowerCase()) ||
            d.sourceFile.toLowerCase().includes(query.toLowerCase())),
      ),
    [docs, filter, query],
  );

  const runIngest = async (newDoc: KnowledgeDoc) => {
    for (let i = 0; i < STAGES.length; i++) {
      setIngestStage(i);
      await new Promise((r) => setTimeout(r, 700));
    }
    setIngestStage(null);
    setUploadedDocs((prev) => [newDoc, ...prev]);
    setUploadOpen(false);
  };

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-6xl px-4 py-6 md:px-8 md:py-10">
          <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
            <div className="min-w-0">
              <h1 className="text-2xl font-semibold tracking-tight">Knowledge Base</h1>
              <p className="mt-1 text-sm text-muted-foreground">
                {isLoading ? (
                  <span className="inline-flex items-center gap-1.5">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Loading index…
                  </span>
                ) : (
                  `${docs.length} documents · ${totalChunks} indexed chunks`
                )}
              </p>
            </div>
            <button
              onClick={() => setUploadOpen(true)}
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90"
            >
              <Upload className="h-4 w-4" />
              Upload document
            </button>
          </div>

          <div className="mb-4 flex flex-wrap items-center gap-2">
            <div className="relative flex-1 min-w-[200px]">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search documents…"
                className="w-full rounded-lg border border-border bg-surface/60 py-2 pl-9 pr-3 text-sm placeholder:text-muted-foreground focus:border-primary focus:outline-none"
              />
            </div>
            {(["all", "statutory", "planning", "advisory"] as Filter[]).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`rounded-full border px-3 py-1.5 text-xs font-medium capitalize transition ${
                  filter === f
                    ? "border-primary bg-primary/15 text-primary"
                    : "border-border bg-surface/40 text-muted-foreground hover:text-foreground"
                }`}
              >
                {f}
              </button>
            ))}
          </div>

          <div className="overflow-hidden rounded-xl border border-border bg-surface/40">
            {isLoading ? (
              <div className="flex items-center justify-center gap-2 px-6 py-16 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading documents from index…
              </div>
            ) : isError ? (
              <div className="flex items-center gap-2 px-6 py-16 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 shrink-0" />
                {error instanceof Error ? error.message : "Failed to load documents"}
              </div>
            ) : filtered.length === 0 ? (
              <div className="px-6 py-16 text-center text-sm text-muted-foreground">
                {docs.length === 0
                  ? "No documents indexed yet — run python ingest.py in the backend."
                  : "No documents match those filters."}
              </div>
            ) : (
              filtered.map((d) => (
                <div key={d.id} className="border-b border-border last:border-0">
                  <button
                    onClick={() => setExpanded((p) => (p === d.id ? null : d.id))}
                    className="flex w-full items-center gap-4 px-4 py-3.5 text-left hover:bg-muted/30"
                  >
                    <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-surface-elevated text-muted-foreground">
                      <FileText className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="truncate text-sm font-medium">{d.title}</span>
                        <DocBadge type={d.docType} />
                      </div>
                      <div className="mt-0.5 truncate text-[11px] text-muted-foreground">
                        {d.sourceFile} · {d.chunks} chunks · {new Date(d.ingestedAt).toLocaleDateString("en-GB")}
                      </div>
                    </div>
                    {expanded === d.id ? (
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-muted-foreground" />
                    )}
                  </button>
                  {expanded === d.id && d.preview && (
                    <div className="border-t border-border bg-surface/30 px-4 py-4">
                      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                        Preview
                      </div>
                      <p className="rounded-lg border border-border bg-surface p-3 text-xs leading-relaxed text-muted-foreground italic">
                        "{d.preview}…"
                      </p>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {uploadOpen && (
        <UploadModal
          onClose={() => setUploadOpen(false)}
          onUpload={runIngest}
          ingestStage={ingestStage}
        />
      )}
    </AppShell>
  );
}

function UploadModal({
  onClose,
  onUpload,
  ingestStage,
}: {
  onClose: () => void;
  onUpload: (d: KnowledgeDoc) => void;
  ingestStage: number | null;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [docType, setDocType] = useState<DocType>("statutory");
  const [section, setSection] = useState("");

  const submit = () => {
    if (!file) return;
    onUpload({
      id: `d${Date.now()}`,
      title: file.name.replace(/\.[^.]+$/, ""),
      docType,
      chunks: Math.floor(8 + Math.random() * 30),
      ingestedAt: Date.now(),
      sourceFile: file.name,
      section: section || undefined,
      preview: "Recently uploaded document — preview generated after indexing.",
    });
  };

  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4">
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-6">
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold">Upload document</h2>
            <p className="text-xs text-muted-foreground">
              PDF, TXT, or Markdown · add to{" "}
              <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">backend/data/</code>{" "}
              and re-run{" "}
              <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">ingest.py</code>{" "}
              to index it
            </p>
          </div>
          <button onClick={onClose} className="rounded-md p-1.5 hover:bg-muted" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        {ingestStage === null ? (
          <>
            <label className="mb-4 block cursor-pointer rounded-xl border-2 border-dashed border-border bg-surface-elevated/40 px-4 py-8 text-center hover:border-primary/40">
              <Upload className="mx-auto h-6 w-6 text-muted-foreground" />
              <div className="mt-2 text-sm font-medium">
                {file ? file.name : "Drop file or click to browse"}
              </div>
              <input
                type="file"
                accept=".pdf,.txt,.md"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>

            <div className="mb-3">
              <label className="mb-1 block text-xs font-medium">Document type</label>
              <div className="flex gap-2">
                {(["statutory", "planning", "advisory"] as DocType[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setDocType(t)}
                    className={`flex-1 rounded-lg border px-3 py-2 text-xs font-medium capitalize ${
                      docType === t
                        ? "border-primary bg-primary/15 text-primary"
                        : "border-border text-muted-foreground"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            <div className="mb-5">
              <label className="mb-1 block text-xs font-medium">Section / category (optional)</label>
              <input
                value={section}
                onChange={(e) => setSection(e.target.value)}
                placeholder="e.g. Chapter VI-A"
                className="w-full rounded-lg border border-border bg-surface-elevated/40 px-3 py-2 text-sm focus:border-primary focus:outline-none"
              />
            </div>

            <button
              onClick={submit}
              disabled={!file}
              className="w-full rounded-lg bg-primary py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-40"
            >
              Preview ingest
            </button>
          </>
        ) : (
          <div className="py-6">
            <div className="mb-4 text-sm font-medium">Simulating ingest for {file?.name}…</div>
            <div className="space-y-2">
              {STAGES.map((s, i) => (
                <div key={s} className="flex items-center gap-3">
                  <div
                    className={`h-2 flex-1 overflow-hidden rounded-full bg-muted ${
                      i <= ingestStage ? "" : "opacity-40"
                    }`}
                  >
                    <div
                      className="h-full bg-primary transition-all"
                      style={{
                        width: i < ingestStage ? "100%" : i === ingestStage ? "60%" : "0%",
                      }}
                    />
                  </div>
                  <span className="w-20 text-right text-[11px] text-muted-foreground">{s}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
