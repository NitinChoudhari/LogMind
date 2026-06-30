import { useEffect, useState, type ReactNode } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import {
  MessageSquarePlus, Library, BarChart3,
  Menu, Trash2, Brain, X, Sun, Moon, Sparkles,
} from "lucide-react";
import { SAMPLE_THREADS, type ChatThreadData } from "@/lib/mockData";
import { cn } from "@/lib/utils";

interface AppShellProps {
  children: ReactNode;
  threads?: ChatThreadData[];
  activeThreadId?: string | null;
  onSelectThread?: (id: string) => void;
  onNewThread?: () => void;
  onDeleteThread?: (id: string) => void;
  showSidebar?: boolean;
}

function groupThreads(threads: ChatThreadData[]) {
  const day = 86400000;
  const now = Date.now();
  const groups: Record<string, ChatThreadData[]> = {
    Today: [], Yesterday: [], "Past 7 days": [], Older: [],
  };
  for (const t of threads) {
    const diff = now - t.updatedAt;
    if      (diff < day)       groups.Today.push(t);
    else if (diff < 2 * day)   groups.Yesterday.push(t);
    else if (diff < 7 * day)   groups["Past 7 days"].push(t);
    else                        groups.Older.push(t);
  }
  return groups;
}

function useDarkMode() {
  const [dark, setDark] = useState(() =>
    document.documentElement.classList.contains("dark"),
  );
  function toggle() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try { localStorage.setItem("logmind-theme", next ? "dark" : "light"); } catch (_) {}
  }
  return { dark, toggle };
}

export function AppShell({
  children,
  threads = SAMPLE_THREADS,
  activeThreadId,
  onSelectThread,
  onNewThread,
  onDeleteThread,
  showSidebar = true,
}: AppShellProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const grouped = groupThreads(threads);
  const { dark, toggle } = useDarkMode();

  useEffect(() => { setMobileOpen(false); }, [pathname]);

  const navItems = [
    { to: "/",          icon: <MessageSquarePlus className="h-4 w-4" />, label: "Chat"           },
    { to: "/knowledge", icon: <Library           className="h-4 w-4" />, label: "Knowledge Base" },
    { to: "/analytics", icon: <BarChart3         className="h-4 w-4" />, label: "Analytics"      },
  ];

  const sidebar = (
    <aside className="flex h-full w-[240px] shrink-0 flex-col bg-surface border-r border-border">

      {/* ── Brand ──────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2.5 px-4 py-[18px]">
        {/* Gradient monogram — like Claude's "C" icon */}
        <div className="relative grid h-7 w-7 shrink-0 place-items-center overflow-hidden rounded-lg">
          <div className="absolute inset-0" style={{ background: "var(--gradient-brand)" }} />
          <Brain className="relative h-3.5 w-3.5 text-white" />
        </div>
        <span className="font-serif text-[15px] font-semibold tracking-tight gradient-text select-none">
          LogMind
        </span>
        {/* Mobile close */}
        <button
          className="ml-auto rounded-md p-1 text-muted-foreground transition hover:bg-muted hover:text-foreground md:hidden"
          onClick={() => setMobileOpen(false)}
          aria-label="Close"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* ── New conversation ─────────────────────────────────────────── */}
      <div className="px-3 pb-3">
        <button
          onClick={() => { onNewThread?.(); setMobileOpen(false); }}
          className="group relative flex w-full items-center gap-2.5 overflow-hidden rounded-xl px-3.5 py-2.5 text-[13px] font-semibold text-primary-foreground transition active:scale-[0.98]"
        >
          <div className="absolute inset-0" style={{ background: "var(--gradient-brand)" }} />
          <div className="absolute inset-0 opacity-0 transition group-hover:opacity-20 bg-white" />
          <MessageSquarePlus className="relative h-4 w-4" />
          <span className="relative">New conversation</span>
        </button>
      </div>

      {/* ── Thread history ────────────────────────────────────────────── */}
      <nav className="flex-1 min-h-0 overflow-y-auto px-2 py-1">
        {pathname === "/" &&
          Object.entries(grouped).map(([label, items]) =>
            items.length === 0 ? null : (
              <div key={label} className="mb-5">
                <p className="px-3 pb-1.5 pt-1 text-[9.5px] font-semibold uppercase tracking-[0.12em] text-muted-foreground/40">
                  {label}
                </p>
                <ul className="space-y-px">
                  {items.map((t) => (
                    <li key={t.id}>
                      <div
                        className={cn(
                          "group flex cursor-pointer items-center gap-2.5 rounded-lg px-3 py-[7px] text-[13px] transition",
                          activeThreadId === t.id
                            ? "bg-background text-foreground shadow-[var(--shadow-xs)]"
                            : "text-muted-foreground hover:bg-background/60 hover:text-foreground",
                        )}
                        onClick={() => { onSelectThread?.(t.id); setMobileOpen(false); }}
                      >
                        <Sparkles className={cn(
                          "h-3 w-3 shrink-0",
                          activeThreadId === t.id ? "text-primary" : "text-muted-foreground/30",
                        )} />
                        <span className="min-w-0 flex-1 truncate font-medium">{t.title}</span>
                        <button
                          className="shrink-0 opacity-0 transition group-hover:opacity-100 hover:text-destructive"
                          onClick={(e) => { e.stopPropagation(); onDeleteThread?.(t.id); }}
                          aria-label="Delete"
                        >
                          <Trash2 className="h-3 w-3" />
                        </button>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            ),
          )}
        {pathname === "/" && threads.length === 0 && (
          <p className="px-4 py-8 text-center text-[12px] leading-relaxed text-muted-foreground/45">
            Your conversations will appear here
          </p>
        )}
        {pathname !== "/" && (
          <p className="px-4 py-8 text-center text-[12px] leading-relaxed text-muted-foreground/45">
            Open a chat to see history
          </p>
        )}
      </nav>

      {/* ── Bottom nav ───────────────────────────────────────────────── */}
      <div className="border-t border-border p-2 space-y-px">
        {navItems.map(({ to, icon, label }) => (
          <Link
            key={to}
            to={to}
            onClick={() => setMobileOpen(false)}
            className={cn(
              "flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium transition",
              pathname === to
                ? "bg-background text-foreground shadow-[var(--shadow-xs)]"
                : "text-muted-foreground hover:bg-background/60 hover:text-foreground",
            )}
          >
            {icon}
            {label}
          </Link>
        ))}

        <div className="pt-px mt-px border-t border-border">
          <button
            onClick={toggle}
            className="flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium text-muted-foreground transition hover:bg-background/60 hover:text-foreground"
          >
            {dark
              ? <Sun  className="h-4 w-4 text-amber-400" />
              : <Moon className="h-4 w-4" />
            }
            {dark ? "Light mode" : "Dark mode"}
          </button>
        </div>
      </div>
    </aside>
  );

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground">
      {showSidebar && <div className="hidden md:flex">{sidebar}</div>}

      {/* Mobile drawer */}
      {showSidebar && mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={() => setMobileOpen(false)} />
          <div className="absolute inset-y-0 left-0 slide-in-right">{sidebar}</div>
        </div>
      )}

      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        {/* Mobile top bar */}
        {showSidebar && (
          <div className="flex items-center gap-2 border-b border-border bg-surface px-4 py-3 md:hidden">
            <button
              className="rounded-lg p-1.5 text-muted-foreground transition hover:bg-muted"
              onClick={() => setMobileOpen(true)}
              aria-label="Open menu"
            >
              <Menu className="h-4 w-4" />
            </button>
            <span className="font-serif text-sm font-semibold gradient-text">LogMind</span>
          </div>
        )}
        {children}
      </main>
    </div>
  );
}
