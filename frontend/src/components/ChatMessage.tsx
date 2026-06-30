import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ThumbsUp, ThumbsDown, Copy,
  Check, RotateCcw, Clock, Zap, Hash,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import type { ChatMessage as Msg, SourceChunk } from "@/lib/mockData";
import { cn } from "@/lib/utils";
import { ThoughtBlock } from "@/components/ThoughtBlock";

function modelSlug(model: string): string {
  const last = model.split(/[/\\]/).pop() ?? model;
  return last.toLowerCase().replace(/[^a-z0-9.-]+/g, "-");
}

/* ── Shared LogMind avatar ──────────────────────────────────────────── */
function LogMindAvatar({ size = "sm" }: { size?: "sm" | "md" }) {
  const px = size === "md" ? "h-9 w-9" : "h-7 w-7";
  const icon = size === "md" ? "text-sm" : "text-[11px]";
  return (
    <div className={`${px} relative shrink-0 overflow-hidden rounded-full`}>
      <div className="absolute inset-0" style={{ background: "var(--gradient-brand)" }} />
      {/* "L" monogram — same as Claude's "C" treatment */}
      <span className={`relative grid h-full w-full place-items-center font-serif font-semibold text-white ${icon}`}>
        L
      </span>
    </div>
  );
}

export function ChatMessage({
  msg,
  streaming,
  onFeedback,
  onRegenerate,
  canRegenerate,
}: {
  msg: Msg;
  streaming?: boolean;
  onFeedback?: (id: string, value: "up" | "down") => void;
  onRegenerate?: (id: string) => void;
  canRegenerate?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const [timelineOpen, setTimelineOpen] = useState(!!streaming);
  const [activeSourceIdx, setActiveSourceIdx] = useState<number | undefined>(undefined);

  /* Citation click → expand the thinking timeline and scroll to that source card. */
  const handleCitation = (idx: number) => {
    setTimelineOpen(true);
    setActiveSourceIdx(idx);
    requestAnimationFrame(() => {
      document.getElementById(`src-${msg.id}-${idx + 1}`)?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    });
  };

  /* ── User message ───────────────────────────────────────────────── */
  if (msg.role === "user") {
    return (
      <div className="fade-up flex justify-end gap-3">
        {/* User bubble — soft surface, no border drama */}
        <div className="max-w-[78%] rounded-3xl rounded-br-md bg-surface px-4 py-3 text-[14px] leading-relaxed text-foreground/90 shadow-[var(--shadow-xs)]">
          {msg.content}
        </div>
        {/* User avatar — plain initials */}
        <div className="mt-1 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-muted text-[11px] font-bold text-muted-foreground ring-1 ring-border">
          U
        </div>
      </div>
    );
  }

  /* ── AI response — Claude-style: no bubble, text flows on background ── */
  return (
    <div className="fade-up flex gap-4">
      <LogMindAvatar />

      <div className="min-w-0 flex-1 pb-2">
        {/* Model label */}
        {msg.model && (
          <p className="mb-2 text-[10.5px] font-medium text-muted-foreground/50 tracking-tight">
            logmind/{modelSlug(msg.model)}
          </p>
        )}

        {/* Thinking timeline (search step + reasoning + done) */}
        <ThoughtBlock
          reasoning={msg.thinking ?? ""}
          route={msg.route}
          reason={msg.reason}
          subQueries={msg.subQueries}
          sources={msg.sources}
          seconds={msg.thinkingSeconds}
          streaming={!!streaming}
          open={timelineOpen}
          onOpenChange={setTimelineOpen}
          activeSourceIdx={activeSourceIdx}
          idPrefix={msg.id}
        />

        {/* Answer text — no card/bubble, breathes on the background.
            While streaming, hidden until the first token so the thinking
            timeline isn't trailed by an empty glowing box. */}
        {(msg.content || !streaming) && (
          <div className={cn(
            "text-[14.5px] leading-[1.75] text-foreground/88",
            streaming && "streaming-glow rounded-xl px-4 py-3 border border-primary/20",
          )}>
            <MarkdownWithCitations
              content={msg.content}
              sources={msg.sources ?? []}
              onCitationClick={handleCitation}
            />
            {streaming && (
              <span className="cursor-blink ml-0.5 inline-block h-3.5 w-0.5 translate-y-0.5 rounded-full bg-primary" />
            )}
          </div>
        )}

        {/* Metric pills */}
        {!streaming && (msg.durationSeconds != null || msg.tokens != null || msg.tokensPerSec != null) && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            {msg.tokensPerSec != null && (
              <MetricPill icon={<Zap  className="h-2.5 w-2.5" />} label={`${msg.tokensPerSec.toFixed(1)} tok/s`} />
            )}
            {msg.tokens != null && (
              <MetricPill icon={<Hash  className="h-2.5 w-2.5" />} label={`${msg.tokens} tok`} />
            )}
            {msg.durationSeconds != null && (
              <MetricPill icon={<Clock className="h-2.5 w-2.5" />} label={`${msg.durationSeconds.toFixed(1)}s`} />
            )}
          </div>
        )}

        {/* Action row */}
        {!streaming && (
          <div className="mt-2 flex items-center gap-0.5 text-muted-foreground">
            {onRegenerate && canRegenerate && (
              <IconBtn onClick={() => onRegenerate(msg.id)} label="Regenerate">
                <RotateCcw className="h-3.5 w-3.5" />
              </IconBtn>
            )}
            <div className="mx-1.5 h-3.5 w-px bg-border" />
            <IconBtn
              onClick={() => onFeedback?.(msg.id, "up")}
              active={msg.feedback === "up"}
              label="Helpful"
            >
              <ThumbsUp className="h-3.5 w-3.5" />
            </IconBtn>
            <IconBtn
              onClick={() => onFeedback?.(msg.id, "down")}
              active={msg.feedback === "down"}
              label="Not helpful"
            >
              <ThumbsDown className="h-3.5 w-3.5" />
            </IconBtn>
            <IconBtn
              onClick={() => {
                navigator.clipboard.writeText(msg.content);
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              }}
              label="Copy"
            >
              {copied
                ? <Check className="h-3.5 w-3.5 text-accent" />
                : <Copy  className="h-3.5 w-3.5" />
              }
            </IconBtn>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricPill({ icon, label }: { icon: ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-muted/60 px-2 py-0.5 text-[10px] font-medium text-muted-foreground tabular-nums">
      {icon}
      {label}
    </span>
  );
}

function IconBtn({
  children, onClick, active, label,
}: {
  children: ReactNode;
  onClick?: () => void;
  active?: boolean;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className={cn(
        "rounded-lg p-1.5 transition hover:bg-muted hover:text-foreground",
        active && "bg-primary/15 text-primary hover:bg-primary/22",
      )}
    >
      {children}
    </button>
  );
}

/* ─── Markdown + inline citation chips ─────────────────────────────── */
function MarkdownWithCitations({
  content,
  sources,
  onCitationClick,
}: {
  content: string;
  sources: SourceChunk[];
  onCitationClick?: (idx: number) => void;
}) {
  const processed = content.replace(/\[(\d+)\]/g, (_, n) => `§CITE${n}§`);

  return (
    <div className="prose-logmind [&_h1]:mt-4 [&_h1]:mb-2 [&_h1]:text-[15px] [&_h1]:font-semibold [&_h1]:text-foreground [&_h2]:mt-4 [&_h2]:mb-2 [&_h2]:text-[14px] [&_h2]:font-semibold [&_h2]:text-foreground [&_h3]:mt-3 [&_h3]:mb-1.5 [&_h3]:text-[13.5px] [&_h3]:font-medium [&_p]:my-2 [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5 [&_li]:leading-relaxed [&_strong]:font-semibold [&_strong]:text-foreground [&_code]:rounded-md [&_code]:bg-muted/80 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[12px] [&_blockquote]:my-3 [&_blockquote]:border-l-2 [&_blockquote]:border-primary/40 [&_blockquote]:pl-3.5 [&_blockquote]:italic [&_blockquote]:text-muted-foreground [&_table]:my-3 [&_table]:w-full [&_table]:border-collapse [&_table]:text-[13px] [&_th]:border [&_th]:border-border [&_th]:bg-muted/50 [&_th]:px-3 [&_th]:py-2 [&_th]:text-left [&_th]:font-semibold [&_td]:border [&_td]:border-border [&_td]:px-3 [&_td]:py-2 [&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2 [&>:first-child]:mt-0 [&>:last-child]:mb-0">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p:  ({ children }) => <p>{renderCitations(children, sources, onCitationClick)}</p>,
          li: ({ children }) => <li>{renderCitations(children, sources, onCitationClick)}</li>,
          td: ({ children }) => <td>{renderCitations(children, sources, onCitationClick)}</td>,
        }}
      >
        {processed}
      </ReactMarkdown>
    </div>
  );
}

function renderCitations(
  children: ReactNode,
  sources: SourceChunk[],
  onClick?: (i: number) => void,
): ReactNode {
  if (typeof children === "string") {
    return children.split(/(§CITE\d+§)/g).map((p, i) => {
      const m = p.match(/^§CITE(\d+)§$/);
      if (!m) return p;
      const n = parseInt(m[1], 10);
      if (n < 1 || n > sources.length) return `[${n}]`;
      return (
        <button
          key={i}
          onClick={() => onClick?.(n - 1)}
          className="mx-0.5 inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-md border border-primary/35 bg-primary/14 px-1 text-[10px] font-bold text-primary align-middle transition hover:bg-primary/24 hover:border-primary/55"
        >
          {n}
        </button>
      );
    });
  }
  if (Array.isArray(children)) {
    return children.map((c, i) => <span key={i}>{renderCitations(c, sources, onClick)}</span>);
  }
  return children;
}
