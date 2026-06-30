import { useEffect, useRef, type ReactNode } from "react";
import {
  ChevronDown, Brain, Globe, BookOpen, FileText, Clock, Check, ExternalLink,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { DocBadge } from "@/components/DocBadge";
import type { SourceChunk } from "@/lib/mockData";

type Route = "kb" | "web" | "general";

export function ThoughtBlock({
  reasoning,
  route,
  reason,
  subQueries,
  sources,
  seconds,
  streaming,
  open,
  onOpenChange,
  activeSourceIdx,
  idPrefix,
}: {
  reasoning: string;
  route?: Route;
  reason?: string;
  subQueries?: string[];
  sources?: SourceChunk[];
  seconds?: number;
  streaming: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  activeSourceIdx?: number;
  idPrefix: string;
}) {
  const userToggled = useRef(false);
  const hadSeconds = useRef(seconds !== undefined);

  /* Auto-collapse once the thinking duration lands — unless the user toggled it. */
  useEffect(() => {
    if (!hadSeconds.current && seconds !== undefined && !userToggled.current) {
      onOpenChange(false);
    }
    hadSeconds.current = seconds !== undefined;
  }, [seconds, onOpenChange]);

  const hasSearch =
    (route === "kb" || route === "web") && (sources?.length ?? 0) > 0;
  const hasContent = Boolean(reasoning) || hasSearch;

  if (!hasContent && !streaming) return null;

  const headline = deriveHeadline({ reasoning, route, reason, streaming, seconds });

  return (
    <div className="mb-3 w-full overflow-hidden rounded-xl border border-border bg-surface transition-all">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <button
        onClick={() => { userToggled.current = true; onOpenChange(!open); }}
        className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left transition hover:bg-muted/20"
        aria-expanded={open}
      >
        <Brain className={cn(
          "h-3.5 w-3.5 shrink-0",
          streaming && seconds === undefined ? "text-primary pulse-dot" : "text-muted-foreground/50",
        )} />
        <span className="flex-1 truncate text-[12px] font-medium text-muted-foreground">{headline}</span>
        {seconds != null && (
          <span className="mr-1.5 tabular-nums text-[10px] text-muted-foreground/50">{seconds.toFixed(1)}s</span>
        )}
        <ChevronDown className={cn(
          "h-3.5 w-3.5 shrink-0 text-muted-foreground/35 transition-transform duration-200",
          open && "rotate-180",
        )} />
      </button>

      {/* ── Timeline body ──────────────────────────────────────────── */}
      {open && hasContent && (
        <div className="relative border-t border-border px-4 py-3.5">
          {/* connector rail behind the step icons */}
          <div className="absolute left-7 top-6 bottom-6 w-px bg-border/70" />

          <div className="space-y-4">
            {/* Search / retrieval step */}
            {hasSearch && (
              <Step icon={route === "web" ? Globe : BookOpen}>
                <div className="flex flex-wrap items-center gap-x-1.5 text-[12px] font-medium text-foreground/80">
                  {route === "web" ? "Searched the web" : "Searched your documents"}
                  <span className="text-muted-foreground/50">· {sources!.length} {sources!.length === 1 ? "result" : "results"}</span>
                </div>
                {subQueries && subQueries.length > 0 && (
                  <div className="mt-1 space-y-0.5">
                    {subQueries.map((q, i) => (
                      <div key={i} className="truncate text-[11px] text-muted-foreground/55">↳ {q}</div>
                    ))}
                  </div>
                )}
                <div className="mt-2 space-y-1.5">
                  {sources!.map((s, i) => (
                    <InlineSourceCard
                      key={s.id}
                      source={s}
                      index={i}
                      active={i === activeSourceIdx}
                      idPrefix={idPrefix}
                    />
                  ))}
                </div>
              </Step>
            )}

            {/* Thinking step */}
            {reasoning && (
              <Step icon={Clock}>
                <div className="text-[12px] font-medium text-foreground/80">Thinking</div>
                <div className="mt-1.5 max-h-64 overflow-y-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-foreground">
                  {reasoning}
                </div>
              </Step>
            )}

            {/* Done step */}
            {!streaming && (
              <Step icon={Check}>
                <span className="text-[12px] font-medium text-muted-foreground/70">Done</span>
              </Step>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Timeline step: icon rail + content ───────────────────────────── */
function Step({
  icon: Icon,
  children,
}: {
  icon: typeof Clock;
  children: ReactNode;
}) {
  return (
    <div className="relative flex gap-3">
      <span className="z-10 mt-px grid h-6 w-6 shrink-0 place-items-center rounded-full bg-surface text-muted-foreground/70 ring-1 ring-border">
        <Icon className="h-3 w-3" />
      </span>
      <div className="min-w-0 flex-1 pt-0.5">{children}</div>
    </div>
  );
}

/* ── Compact inline source card ───────────────────────────────────── */
function InlineSourceCard({
  source: s,
  index: i,
  active,
  idPrefix,
}: {
  source: SourceChunk;
  index: number;
  active: boolean;
  idPrefix: string;
}) {
  const isWeb = s.docType === "web";
  const sub = isWeb ? safeDomain(s.docId) : s.section;
  const pct = Math.round((s.score ?? 0) * 100);

  return (
    <div
      id={`src-${idPrefix}-${i + 1}`}
      className={cn(
        "scroll-mt-4 rounded-lg border px-3 py-2 transition-colors",
        active
          ? "border-primary/50 bg-primary/[0.06]"
          : "border-border bg-card/50 hover:border-primary/30",
      )}
    >
      <div className="flex items-center gap-2">
        <span className="grid h-[18px] w-[18px] shrink-0 place-items-center rounded bg-muted text-[9px] font-bold text-muted-foreground">
          {i + 1}
        </span>
        {isWeb
          ? <Globe className="h-3 w-3 shrink-0 text-muted-foreground/60" />
          : <FileText className="h-3 w-3 shrink-0 text-muted-foreground/60" />}
        {isWeb ? (
          <a
            href={s.docId}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className="inline-flex min-w-0 items-center gap-1 text-[12px] font-medium text-foreground hover:text-primary hover:underline"
          >
            <span className="truncate">{s.docTitle}</span>
            <ExternalLink className="h-2.5 w-2.5 shrink-0 opacity-50" />
          </a>
        ) : (
          <span className="truncate text-[12px] font-medium text-foreground">{s.docTitle}</span>
        )}
        <span className="ml-auto shrink-0 tabular-nums text-[10px] text-muted-foreground/45">{pct}%</span>
      </div>

      <div className="mt-1 flex items-center gap-1.5 pl-[26px]">
        <DocBadge type={s.docType} />
        {sub && <span className="truncate text-[10px] text-muted-foreground/55">{sub}</span>}
      </div>

      {s.excerpt && (
        <p className="mt-1 pl-[26px] text-[11px] italic leading-relaxed text-muted-foreground/55 line-clamp-2">
          "{s.excerpt}"
        </p>
      )}
    </div>
  );
}

/* ── Helpers ──────────────────────────────────────────────────────── */
function deriveHeadline({
  reasoning,
  route,
  reason,
  streaming,
  seconds,
}: {
  reasoning: string;
  route?: Route;
  reason?: string;
  streaming: boolean;
  seconds?: number;
}): string {
  if (streaming && seconds === undefined) return "Thinking…";

  const fromReasoning = firstSentence(reasoning);
  if (fromReasoning) return fromReasoning;
  if (reason) return clamp(reason.trim());
  if (route === "web") return "Searched the web";
  if (route === "kb") return "Searched your documents";
  if (route === "general") return "Answered from general knowledge";
  return "Thought process";
}

function firstSentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return "";
  const m = trimmed.match(/^[\s\S]*?[.!?](?:\s|$)/);
  const sentence = (m ? m[0] : trimmed.split(/\n/)[0]).trim().replace(/[.!?]+$/, "");
  return clamp(sentence);
}

function clamp(s: string, max = 72): string {
  return s.length > max ? s.slice(0, max - 1).trimEnd() + "…" : s;
}

function safeDomain(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}
