import type { DocType } from "@/lib/mockData";
import { cn } from "@/lib/utils";

const styles: Record<DocType, string> = {
  statutory: "bg-[var(--statutory)]/15 text-[var(--statutory)] border-[var(--statutory)]/30",
  planning: "bg-[var(--planning)]/15 text-[var(--planning)] border-[var(--planning)]/30",
  advisory: "bg-[var(--advisory)]/15 text-[var(--advisory)] border-[var(--advisory)]/30",
  web: "bg-[var(--web)]/15 text-[var(--web)] border-[var(--web)]/30",
};

export function DocBadge({ type, className }: { type: DocType; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
        styles[type],
        className,
      )}
    >
      {type}
    </span>
  );
}
