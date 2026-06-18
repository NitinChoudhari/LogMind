import { useState } from "react";

export default function TraceConsole({ text, streaming }) {
  const [open, setOpen] = useState(false);
  if (!text) return null;
  const show = open || streaming;
  const lines = text.split("\n").length;
  return (
    <div className="tracecon">
      <button className="tracecon__toggle" onClick={() => setOpen((o) => !o)}>
        <span className="tracecon__caret">{show ? "▾" : "▸"}</span>
        agent trace · verbose ({lines} lines){streaming ? " · live" : ""}
      </button>
      {show && <pre className="tracecon__body">{text}</pre>}
    </div>
  );
}
