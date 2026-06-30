import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { FileText, Database, MessageSquare, Clock, ThumbsUp, ThumbsDown, Loader2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { AppShell } from "@/components/AppShell";
import { fetchAnalytics, type AnalyticsData } from "@/lib/api";

const EMPTY: AnalyticsData = {
  totalDocs: 0,
  totalChunks: 0,
  queriesToday: 0,
  avgResponseMs: 0,
  queryVolume: [],
  docTypeSplit: [],
  recentQueries: [],
  avgRelevance: 0,
  hitRate: 0,
};

export function AnalyticsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["analytics"],
    queryFn: fetchAnalytics,
    staleTime: 10_000,
    refetchInterval: 30_000,
  });

  const A = data ?? EMPTY;
  const nowRef = Date.now();

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-6xl px-4 py-6 md:px-8 md:py-10">
          <div className="mb-6 flex items-end justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">Analytics</h1>
              <p className="mt-1 text-sm text-muted-foreground">
                Usage and retrieval quality across the LogMind knowledge base.
              </p>
            </div>
            {isLoading && (
              <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Fetching…
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat icon={<FileText className="h-4 w-4" />} label="Documents" value={A.totalDocs} />
            <Stat icon={<Database className="h-4 w-4" />} label="Indexed chunks" value={A.totalChunks} />
            <Stat icon={<MessageSquare className="h-4 w-4" />} label="Queries today" value={A.queriesToday} />
            <Stat
              icon={<Clock className="h-4 w-4" />}
              label="Avg response"
              value={A.avgResponseMs > 0 ? `${(A.avgResponseMs / 1000).toFixed(2)}s` : "—"}
            />
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            <div className="relative overflow-hidden rounded-xl border border-border bg-surface/60 p-4 shadow-[var(--shadow-md)] lg:col-span-2">
              <div
                className="absolute inset-x-0 top-0 h-px"
                style={{ background: "var(--gradient-primary)" }}
              />
              <div className="mb-3 flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold tracking-tight">Query volume</h2>
                  <p className="text-[11px] text-muted-foreground">Last 14 days</p>
                </div>
                <div className="flex items-center gap-1.5 rounded-full bg-primary/10 px-2.5 py-1 text-[11px] font-medium text-primary">
                  <span className="h-1.5 w-1.5 rounded-full bg-primary pulse-dot" />
                  live
                </div>
              </div>
              <div className="h-64">
                {A.queryVolume.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={A.queryVolume} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                      <defs>
                        <linearGradient id="qvFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="oklch(0.46 0.11 50)" stopOpacity={0.45} />
                          <stop offset="100%" stopColor="oklch(0.46 0.11 50)" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="qvStroke" x1="0" y1="0" x2="1" y2="0">
                          <stop offset="0%" stopColor="oklch(0.46 0.11 50)" />
                          <stop offset="100%" stopColor="oklch(0.62 0.13 55)" />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="2 4" stroke="oklch(0.22 0.025 55 / 10%)" vertical={false} />
                      <XAxis dataKey="day" stroke="oklch(0.22 0.025 55 / 45%)" fontSize={10} tickLine={false} axisLine={false} />
                      <YAxis stroke="oklch(0.22 0.025 55 / 45%)" fontSize={10} tickLine={false} axisLine={false} width={32} allowDecimals={false} />
                      <Tooltip
                        cursor={{ stroke: "oklch(0.46 0.11 50 / 40%)", strokeWidth: 1, strokeDasharray: "3 3" }}
                        contentStyle={{
                          background: "oklch(0.98 0.008 80)",
                          border: "1px solid oklch(0.22 0.025 55 / 14%)",
                          borderRadius: 10,
                          fontSize: 12,
                          color: "oklch(0.22 0.025 55)",
                          boxShadow: "0 10px 30px -12px oklch(0.22 0.025 55 / 25%)",
                          padding: "8px 12px",
                        }}
                        labelStyle={{ fontWeight: 600, marginBottom: 2 }}
                        itemStyle={{ color: "oklch(0.46 0.11 50)" }}
                      />
                      <Area
                        type="monotone"
                        dataKey="queries"
                        stroke="url(#qvStroke)"
                        strokeWidth={2.5}
                        fill="url(#qvFill)"
                        dot={false}
                        activeDot={{ r: 5, fill: "oklch(0.46 0.11 50)", stroke: "oklch(0.98 0.008 80)", strokeWidth: 2 }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                    No queries yet
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-xl border border-border bg-surface/40 p-4">
              <h2 className="mb-3 text-sm font-semibold">Doc type distribution</h2>
              {A.docTypeSplit.length > 0 ? (
                <>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={A.docTypeSplit}
                          dataKey="value"
                          innerRadius={45}
                          outerRadius={70}
                          paddingAngle={2}
                        >
                          {A.docTypeSplit.map((s, i) => (
                            <Cell key={i} fill={s.color} stroke="transparent" />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-2 space-y-1.5">
                    {A.docTypeSplit.map((s) => (
                      <div key={s.name} className="flex items-center gap-2 text-xs">
                        <span className="h-2 w-2 rounded-full" style={{ background: s.color }} />
                        <span className="flex-1 text-muted-foreground">{s.name}</span>
                        <span className="font-medium">{s.value}</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="flex h-48 items-center justify-center text-sm text-muted-foreground">
                  No index data
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 grid gap-4 lg:grid-cols-3">
            <div className="rounded-xl border border-border bg-surface/40 p-4 lg:col-span-2">
              <h2 className="mb-3 text-sm font-semibold">Recent queries</h2>
              {A.recentQueries.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-left text-[10px] uppercase tracking-wider text-muted-foreground">
                        <th className="pb-2 pr-3 font-medium">When</th>
                        <th className="pb-2 pr-3 font-medium">Query</th>
                        <th className="pb-2 pr-3 font-medium">Time</th>
                        <th className="pb-2 pr-3 font-medium">Sources</th>
                        <th className="pb-2 font-medium">Feedback</th>
                      </tr>
                    </thead>
                    <tbody>
                      {A.recentQueries.map((q, i) => (
                        <tr key={i} className="border-t border-border">
                          <td className="py-2 pr-3 text-muted-foreground">{relTime(q.ts, nowRef)}</td>
                          <td className="max-w-xs truncate py-2 pr-3">{q.query}</td>
                          <td className="py-2 pr-3 text-muted-foreground">
                            {q.ms > 0 ? `${(q.ms / 1000).toFixed(2)}s` : "—"}
                          </td>
                          <td className="py-2 pr-3 text-muted-foreground">{q.sources}</td>
                          <td className="py-2">
                            {q.feedback === "up" && (
                              <ThumbsUp className="h-3.5 w-3.5 text-[var(--success)]" />
                            )}
                            {q.feedback === "down" && (
                              <ThumbsDown className="h-3.5 w-3.5 text-destructive" />
                            )}
                            {q.feedback === null && (
                              <span className="text-muted-foreground">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="flex h-24 items-center justify-center text-sm text-muted-foreground">
                  No queries logged yet
                </div>
              )}
            </div>

            <div className="rounded-xl border border-border bg-surface/40 p-4">
              <h2 className="mb-4 text-sm font-semibold">Retrieval quality</h2>
              <MetricBar label="Avg relevance score" value={A.avgRelevance} />
              <div className="h-4" />
              <MetricBar label="Top-k hit rate" value={A.hitRate} />
              <p className="mt-4 text-[11px] leading-relaxed text-muted-foreground">
                Hit rate = share of queries where the top-3 retrieval included a chunk used in the
                final synthesized answer.
              </p>
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

function Stat({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface/40 p-4">
      <div className="flex items-center gap-2 text-muted-foreground">
        {icon}
        <span className="text-[11px] font-medium uppercase tracking-wider">{label}</span>
      </div>
      <div className="mt-2 text-2xl font-semibold tracking-tight">{value}</div>
    </div>
  );
}

function MetricBar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-medium">{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-primary transition-all" style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

function relTime(ts: number, nowRef: number) {
  const diff = nowRef - ts;
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}
