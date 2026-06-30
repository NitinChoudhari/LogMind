import { useState } from "react";

export default function AskBar({ onAsk, loading }) {
  const [q, setQ] = useState("");

  function submit() {
    const trimmed = q.trim();
    if (trimmed.length >= 3 && !loading) {
      onAsk(trimmed);
      setQ("");
    }
  }

  return (
    <div className="askbar">
      <input
        className="askbar__input"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
        placeholder="Ask about your tax knowledge base…"
        autoFocus
      />
      <button
        className="askbar__btn"
        onClick={submit}
        disabled={loading}
        title={loading ? "Working…" : "Ask"}
        aria-label={loading ? "Working…" : "Ask"}
      />
    </div>
  );
}
