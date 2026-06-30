import { useEffect, useRef, useState } from "react";
import Header from "./components/Header.jsx";
import AskBar from "./components/AskBar.jsx";
import Turn from "./components/Turn.jsx";
import { getHealth, getDocuments, streamQuery } from "./api.js";

let nextTurnId = 1;

function emptyTurn(question) {
  return {
    id: nextTurnId++,
    question,
    subQueries: [],
    sources: [],
    answer: "",
    traceText: "",
    meta: null,
    error: "",
    streaming: true,
    active: null,
  };
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [docs, setDocs] = useState(null);
  const [turns, setTurns] = useState([]);
  const bottomRef = useRef(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getDocuments().then(setDocs).catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  function patchTurn(id, patch) {
    setTurns((prev) =>
      prev.map((t) => (t.id === id ? { ...t, ...(typeof patch === "function" ? patch(t) : patch) } : t))
    );
  }

  async function ask(question) {
    const turn = emptyTurn(question);
    setTurns((prev) => [...prev, turn]);

    await streamQuery(question, {
      onTrace: (line) => patchTurn(turn.id, (t) => ({ traceText: t.traceText + line + "\n" })),
      onSubqueries: (items) => patchTurn(turn.id, { subQueries: items }),
      onSources: (srcs) => patchTurn(turn.id, { sources: srcs }),
      onToken: (text) => patchTurn(turn.id, (t) => ({ answer: t.answer + text })),
      onDone: (e) => patchTurn(turn.id, { meta: { provider: e.provider, model: e.model }, streaming: false }),
      onError: (m) => patchTurn(turn.id, { error: m, streaming: false }),
    });
  }

  function onCite(turnId, n) {
    patchTurn(turnId, { active: n });
    const el = document.getElementById(`src-${turnId}-${n}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const streaming = turns.length > 0 && turns[turns.length - 1].streaming;

  return (
    <div className="app">
      <Header health={health} docs={docs} />

      <main className="main">
        <div className="thread">
          <div className="thread__inner">
            {turns.length === 0 && (
              <div className="empty">
                <p>Ask a question and LogMind plans sub-queries, retrieves grounded passages, and streams a cited answer — with the agent trace live as it works.</p>
              </div>
            )}
            {turns.map((turn) => (
              <Turn key={turn.id} turn={turn} onCite={onCite} />
            ))}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="composer">
          <div className="composer__inner">
            <AskBar onAsk={ask} loading={streaming} />
            <p className="hint">LogMind answers only from the indexed documents and cites every claim.</p>
          </div>
        </div>
      </main>
    </div>
  );
}
