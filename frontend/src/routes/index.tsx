import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, ChevronDown, Brain, Search, Sparkles, BookOpen, Scale } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { ChatMessage } from "@/components/ChatMessage";
import { streamQuery, mapSources } from "@/lib/api";
import type { ChatThreadData, ChatMessage as Msg, SourceChunk } from "@/lib/mockData";

type Route = "kb" | "web" | "general";

type PipelineStage = "idle" | "querying" | "researching" | "synthesizing" | "complete";

const MODES = [
  { id: "auto",      label: "Auto",      desc: "Let LogMind choose the best retrieval mode" },
  { id: "statutory", label: "Statutory", desc: "Focus on act & regulation text" },
  { id: "advisory",  label: "Advisory",  desc: "Focus on study material & examples" },
] as const;

export function ChatPage() {
  const [threads, setThreads]               = useState<ChatThreadData[]>([]);
  const [activeId, setActiveId]             = useState<string | null>(null);
  const [input, setInput]                   = useState("");
  const [stage, setStage]                   = useState<PipelineStage>("idle");
  const [streamingText, setStreamingText]   = useState("");
  const [streamingSubQueries, setStreamingSubQueries] = useState<string[]>([]);
  const [streamingThinking, setStreamingThinking]     = useState("");
  const [streamingThinkingSeconds, setStreamingThinkingSeconds] = useState<number | undefined>(undefined);
  const [streamingRoute, setStreamingRoute] = useState<Route | undefined>(undefined);
  const [streamingReason, setStreamingReason] = useState("");
  const [streamingSources, setStreamingSources] = useState<SourceChunk[]>([]);

  const [mode, setMode]         = useState<(typeof MODES)[number]["id"]>("auto");
  const [modeOpen, setModeOpen] = useState(false);

  const scrollRef      = useRef<HTMLDivElement>(null);
  const inputRef       = useRef<HTMLTextAreaElement>(null);
  const streamTextRef  = useRef("");
  const thinkingTextRef = useRef("");
  const requestStartRef = useRef(0);

  const activeThread = useMemo(
    () => threads.find((t) => t.id === activeId) ?? null,
    [threads, activeId],
  );

  const lastAssistantId = useMemo(() => {
    const msgs = activeThread?.messages ?? [];
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "assistant") return msgs[i].id;
    }
    return null;
  }, [activeThread]);

  useEffect(() => { inputRef.current?.focus(); }, [activeId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [activeThread?.messages.length, streamingText, streamingThinking, stage]);

  const newThread = () => {
    const t: ChatThreadData = { id: `t${Date.now()}`, title: "New chat", updatedAt: Date.now(), messages: [] };
    setThreads((prev) => [t, ...prev]);
    setActiveId(t.id);
  };

  const deleteThread = (id: string) => {
    setThreads((prev) => prev.filter((t) => t.id !== id));
    if (activeId === id) setActiveId(threads.find((t) => t.id !== id)?.id ?? null);
  };

  const setFeedback = (msgId: string, value: "up" | "down") => {
    setThreads((prev) =>
      prev.map((t) => t.id !== activeId ? t : {
        ...t,
        messages: t.messages.map((m) => m.id === msgId ? { ...m, feedback: m.feedback === value ? null : value } : m),
      }),
    );
  };

  const resetStreaming = () => {
    streamTextRef.current = "";
    thinkingTextRef.current = "";
    setStreamingText("");
    setStreamingThinking("");
    setStreamingThinkingSeconds(undefined);
    setStreamingSubQueries([]);
    setStreamingRoute(undefined);
    setStreamingReason("");
    setStreamingSources([]);
  };

  const runQuery = async (text: string, threadId: string) => {
    resetStreaming();
    requestStartRef.current = Date.now();
    setStage("querying");

    let pendingSources: SourceChunk[] = [];
    let pendingThinkingSeconds: number | undefined;
    let pendingRoute: Route | undefined;
    let pendingReason = "";
    let pendingSubQueries: string[] = [];
    let sawFirstToken = false;

    await streamQuery(text, {
      onRoute: (route, reason) => {
        pendingRoute = route as Route;
        pendingReason = reason;
        setStreamingRoute(route as Route);
        setStreamingReason(reason);
      },
      onSubqueries: (items) => {
        pendingSubQueries = items;
        setStreamingSubQueries(items);
        setStage("researching");
      },
      onSources: (srcs) => {
        pendingSources = mapSources(srcs);
        setStreamingSources(pendingSources);
        setStage("synthesizing");
      },
      onThinking: (tok) => {
        thinkingTextRef.current += tok;
        setStreamingThinking(thinkingTextRef.current);
      },
      onThinkingDone: (seconds) => {
        pendingThinkingSeconds = seconds;
        setStreamingThinkingSeconds(seconds);
      },
      onToken: (tok) => {
        if (!sawFirstToken) {
          sawFirstToken = true;
          if (pendingThinkingSeconds === undefined) {
            pendingThinkingSeconds = (Date.now() - requestStartRef.current) / 1000;
            setStreamingThinkingSeconds(pendingThinkingSeconds);
          }
        }
        streamTextRef.current += tok;
        setStreamingText(streamTextRef.current);
      },
      onDone: ({ model, tokens, tokensPerSec }) => {
        if (pendingThinkingSeconds === undefined) {
          pendingThinkingSeconds = (Date.now() - requestStartRef.current) / 1000;
        }
        const aiMsg: Msg = {
          id:              `a${Date.now()}`,
          role:            "assistant",
          content:         streamTextRef.current || "No response received.",
          sources:         pendingSources.length > 0 ? pendingSources : undefined,
          route:           pendingRoute,
          reason:          pendingReason || undefined,
          subQueries:      pendingSubQueries.length > 0 ? pendingSubQueries : undefined,
          createdAt:       Date.now(),
          model,
          thinking:        thinkingTextRef.current || undefined,
          thinkingSeconds: pendingThinkingSeconds,
          durationSeconds: (Date.now() - requestStartRef.current) / 1000,
          tokens,
          tokensPerSec,
        };
        setThreads((prev) =>
          prev.map((t) => t.id !== threadId ? t : { ...t, updatedAt: Date.now(), messages: [...t.messages, aiMsg] }),
        );
        resetStreaming();
        setStage("idle");
        inputRef.current?.focus();
      },
      onError: (msg) => {
        const errMsg: Msg = { id: `a${Date.now()}`, role: "assistant", content: `**Error:** ${msg}`, createdAt: Date.now() };
        setThreads((prev) =>
          prev.map((t) => t.id !== threadId ? t : { ...t, messages: [...t.messages, errMsg] }),
        );
        resetStreaming();
        setStage("idle");
      },
    });
  };

  const send = async () => {
    const text = input.trim();
    if (!text || stage !== "idle") return;
    setInput("");

    const userMsg: Msg = { id: `u${Date.now()}`, role: "user", content: text, createdAt: Date.now() };
    let threadId = activeId;

    if (!threadId || !activeThread) {
      const t: ChatThreadData = { id: `t${Date.now()}`, title: text.slice(0, 60), updatedAt: Date.now(), messages: [userMsg] };
      setThreads((prev) => [t, ...prev]);
      threadId = t.id;
      setActiveId(threadId);
    } else {
      setThreads((prev) =>
        prev.map((t) => t.id !== threadId ? t : {
          ...t,
          title: t.messages.length === 0 ? text.slice(0, 60) : t.title,
          updatedAt: Date.now(),
          messages: [...t.messages, userMsg],
        }),
      );
    }

    await runQuery(text, threadId);
  };

  const regenerate = async (assistantId: string) => {
    if (!activeThread || !activeId || stage !== "idle") return;
    const idx = activeThread.messages.findIndex((m) => m.id === assistantId);
    if (idx <= 0) return;
    const userMsg = activeThread.messages[idx - 1];
    if (userMsg.role !== "user") return;

    const threadId = activeId;
    setThreads((prev) =>
      prev.map((t) => t.id !== threadId ? t : { ...t, messages: t.messages.filter((m) => m.id !== assistantId) }),
    );
    await runQuery(userMsg.content, threadId);
  };

  const isStreaming = stage !== "idle" && stage !== "complete";

  return (
    <AppShell threads={threads} activeThreadId={activeId} onSelectThread={setActiveId} onNewThread={newThread} onDeleteThread={deleteThread}>
      <div className="relative flex min-h-0 flex-1 overflow-hidden">

        {/* ── Chat column ──────────────────────────────────────────────── */}
        <div className="flex min-w-0 flex-1 flex-col">

          {/* Scrollable messages area */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto">
            <div className="mx-auto w-full max-w-2xl px-4 py-6 md:px-6 md:py-10">

              {(!activeThread || activeThread.messages.length === 0) && stage === "idle" ? (
                <EmptyState onPick={(q) => setInput(q)} />
              ) : (
                <div className="space-y-6">
                  {activeThread?.messages.map((m) => (
                    <ChatMessage
                      key={m.id}
                      msg={m}
                      onFeedback={setFeedback}
                      onRegenerate={regenerate}
                      canRegenerate={stage === "idle" && m.id === lastAssistantId}
                    />
                  ))}

                  {/* ── Live streaming turn — same ChatMessage, fed from streaming state ── */}
                  {isStreaming && (
                    <ChatMessage
                      key="streaming"
                      streaming
                      msg={{
                        id:              "streaming",
                        role:            "assistant",
                        content:         streamingText,
                        createdAt:       requestStartRef.current,
                        route:           streamingRoute,
                        reason:          streamingReason || undefined,
                        subQueries:      streamingSubQueries.length > 0 ? streamingSubQueries : undefined,
                        sources:         streamingSources.length > 0 ? streamingSources : undefined,
                        thinking:        streamingThinking || undefined,
                        thinkingSeconds: streamingThinkingSeconds,
                      }}
                    />
                  )}
                </div>
              )}
            </div>
          </div>

          {/* ── Composer ─────────────────────────────────────────────── */}
          <div className="relative z-10 border-t border-border bg-surface px-4 py-3 md:px-6">
            <div className="mx-auto w-full max-w-2xl">
              <div className="glass relative overflow-hidden rounded-2xl ring-1 ring-transparent focus-within:ring-primary/35 transition-shadow duration-200">
                <textarea
                  ref={inputRef}
                  rows={1}
                  value={input}
                  onChange={(e) => {
                    setInput(e.target.value);
                    e.target.style.height = "auto";
                    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); }
                  }}
                  placeholder="Ask anything about tax law, deductions, compliance…"
                  className="block w-full resize-none bg-transparent px-4 pt-3.5 pb-1 text-sm leading-relaxed text-foreground placeholder:text-muted-foreground/60 focus:outline-none"
                />
                <div className="flex items-center gap-2 px-3 pb-3 pt-1">
                  {/* Mode selector */}
                  <div className="relative">
                    <button
                      onClick={() => setModeOpen((v) => !v)}
                      className="inline-flex items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-[11px] font-medium text-muted-foreground transition hover:border-primary/30 hover:text-foreground"
                    >
                      {MODES.find((m) => m.id === mode)?.label}
                      <ChevronDown className="h-2.5 w-2.5" />
                    </button>
                    {modeOpen && (
                      <div className="glass-strong absolute bottom-full left-0 mb-2 w-52 overflow-hidden rounded-xl shadow-[var(--shadow-md)]">
                        {MODES.map((m) => (
                          <button
                            key={m.id}
                            onClick={() => { setMode(m.id); setModeOpen(false); }}
                            className={`block w-full px-3.5 py-2.5 text-left transition hover:bg-primary/8 ${
                              mode === m.id ? "text-primary" : "text-foreground"
                            }`}
                          >
                            <div className="text-[12.5px] font-medium">{m.label}</div>
                            <div className="text-[11px] text-muted-foreground/70 mt-0.5">{m.desc}</div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Send button */}
                  <button
                    onClick={() => void send()}
                    disabled={!input.trim() || stage !== "idle"}
                    className="group relative ml-auto inline-flex h-8 w-8 items-center justify-center overflow-hidden rounded-full text-primary-foreground shadow-sm transition active:scale-95 disabled:cursor-not-allowed disabled:opacity-25"
                    aria-label="Send"
                  >
                    <div className="absolute inset-0 transition group-hover:opacity-90" style={{ background: "var(--gradient-brand)" }} />
                    {stage !== "idle" ? (
                      <span className="relative pulse-dot h-3 w-3 rounded-full border-2 border-white/60" />
                    ) : (
                      <ArrowUp className="relative h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>

              <p className="mt-2 text-center text-[10.5px] text-muted-foreground/50">
                Answers cite indexed documents — verify important matters with a qualified professional.
              </p>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}


/* ─── Empty / welcome state ─────────────────────────────────────────── */
const SUGGESTIONS = [
  {
    icon: <Scale    className="h-4 w-4" />,
    text: "Old vs new tax regime — which saves more at ₹18 LPA?",
    tag:  "Comparison",
  },
  {
    icon: <BookOpen className="h-4 w-4" />,
    text: "How much can I save under Section 80C this year?",
    tag:  "Deductions",
  },
  {
    icon: <Search   className="h-4 w-4" />,
    text: "Calculate HRA exemption for Bangalore rent of ₹28,000",
    tag:  "HRA",
  },
  {
    icon: <Sparkles className="h-4 w-4" />,
    text: "Home loan interest deduction for self-occupied property",
    tag:  "PGBP & HP",
  },
];

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="flex min-h-[72vh] flex-col items-center justify-center px-4 text-center">

      {/* Avatar — same gradient as sidebar + chat, larger */}
      <div className="relative mb-7 grid h-16 w-16 place-items-center overflow-hidden rounded-2xl shadow-[var(--shadow-md)]">
        <div className="absolute inset-0" style={{ background: "var(--gradient-brand)" }} />
        <Brain className="relative h-8 w-8 text-white" />
      </div>

      {/* Headline — Playfair serif, Claude-like warmth */}
      <h1 className="font-serif text-[2rem] font-semibold leading-tight tracking-tight text-foreground">
        How can I help?
      </h1>
      <p className="mt-3 max-w-sm text-[14px] leading-relaxed text-muted-foreground">
        Ask anything about Indian tax law and I'll find the answer in your indexed documents, with sources.
      </p>

      {/* Suggestion cards — clean, no glass */}
      <div className="mt-9 grid w-full max-w-[560px] grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map(({ icon, text, tag }) => (
          <button
            key={text}
            onClick={() => onPick(text)}
            className="group flex flex-col items-start gap-3 rounded-xl border border-border bg-card p-4 text-left shadow-[var(--shadow-xs)] transition hover:border-primary/25 hover:shadow-[var(--shadow-sm)] active:scale-[0.99]"
          >
            <div className="flex items-center gap-2">
              <span className="grid h-7 w-7 place-items-center rounded-lg bg-primary/10 text-primary transition group-hover:bg-primary/18">
                {icon}
              </span>
              <span className="rounded-full bg-muted px-2 py-[3px] text-[9.5px] font-semibold uppercase tracking-wide text-muted-foreground/70">
                {tag}
              </span>
            </div>
            <p className="text-[13px] leading-snug text-foreground/75 transition group-hover:text-foreground">
              {text}
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}
