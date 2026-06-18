import { useState } from "react";

export default function AskBar({ onAsk, loading }) {
  const [q, setQ] = useState("");

  function submit() {
    const trimmed = q.trim();
    if (trimmed.length >= 3 && !loading) onAsk(trimmed);
  }

  return (
    <div className="askbar">
      <span className="askbar__prompt">›</span>
      <input
        className="askbar__input"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
        placeholder="Ask about the indexed manuals, error codes, or policies…"
        autoFocus
      />
      <button className="askbar__btn" onClick={submit} disabled={loading}>
        {loading ? "Working…" : "Ask"}
      </button>
    </div>
  );
}
