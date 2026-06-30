import { useState } from "react";

export default function SourceList({ sources, active, turnId }) {
  const [open, setOpen] = useState(false); // open by default so citation clicks always have a target to scroll to
  if (!sources || !sources.length) return null;
  return (
    <div className="sources">
      <button className="tracecon__toggle" onClick={() => setOpen((o) => !o)}>
        <span className="tracecon__caret">{open ? "▾" : "▸"}</span>
        retrieved sources ({sources.length})
      </button>

      {open &&
        sources.map((s) => (
          <div
            key={s.n}
            id={`src-${turnId}-${s.n}`}
            className={"source" + (active === s.n ? " source--active" : "")}
          >
            <div className="source__head">
              <span className="source__n">{s.n}</span>
              <span className="source__file">{s.source}</span>
              {s.section && <span className="source__section">{s.section}</span>}
            </div>
            <p className="source__snippet">{s.snippet}…</p>
          </div>
        ))}
    </div>
  );
}