import { useEffect, useState } from "react";
import Header from "./components/Header.jsx";
import AskBar from "./components/AskBar.jsx";
import AgentTrace from "./components/AgentTrace.jsx";
import AnswerCard from "./components/AnswerCard.jsx";
import SourceList from "./components/SourceList.jsx";
import TraceConsole from "./components/TraceConsole.jsx";
import { getHealth, getDocuments, streamQuery } from "./api.js";

export default function App() {
  const [health, setHealth] = useState(null);
  const [docs, setDocs] = useState(null);

  const [streaming, setStreaming] = useState(false);
  const [asked, setAsked] = useState("");
  const [error, setError] = useState("");
  const [active, setActive] = useState(null);

  // streamed pieces
  const [subQueries, setSubQueries] = useState([]);
  const [sources, setSources] = useState([]);
  const [answer, setAnswer] = useState("");
  const [traceText, setTraceText] = useState("");
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => {});
    getDocuments().then(setDocs).catch(() => {});
  }, []);

  async function ask(question) {
    setStreaming(true);
    setError("");
    setActive(null);
    setAsked(question);
    setSubQueries([]);
    setSources([]);
    setAnswer("");
    setTraceText("");
    setMeta(null);

    await streamQuery(question, {
      onTrace: (line) => setTraceText((prev) => prev + line + "\n"),
      onSubqueries: (items) => setSubQueries(items),
      onSources: (srcs) => setSources(srcs),
      onToken: (t) => setAnswer((prev) => prev + t),
      onDone: (e) => {
        setMeta({ provider: e.provider, model: e.model });
        setStreaming(false);
      },
      onError: (m) => {
        setError(m);
        setStreaming(false);
      },
    });
  }

  function onCite(n) {
    setActive(n);
    const el = document.getElementById(`src-${n}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const showAnswer = answer.length > 0 || (streaming && sources.length > 0);
  const thinking = streaming && subQueries.length === 0;

  return (
    <div className="app">
      <Header health={health} docs={docs} />

      <main className="main">
        <AskBar onAsk={ask} loading={streaming} />

        {asked && (
          <p className="asked">
            <span className="asked__q">›</span> {asked}
          </p>
        )}

        {error && <div className="errorbox">{error}</div>}

        {thinking && (
          <div className="loader">
            <span className="loader__spin" />
            <span className="loader__stages">
              <span className="loader__stage is-on">contacting agents…</span>
            </span>
          </div>
        )}

        {(subQueries.length > 0 || showAnswer || traceText) && (
          <div className="result">
            <AgentTrace subQueries={subQueries} />
            {showAnswer && (
              <AnswerCard answer={answer} onCite={onCite} streaming={streaming} />
            )}
            <SourceList sources={sources} active={active} />
            <TraceConsole text={traceText} streaming={streaming} />
            {meta && !streaming && (
              <div className="meta">
                {meta.provider} · {meta.model} · {sources.length} sources
              </div>
            )}
          </div>
        )}

        {!streaming && !answer && !error && !asked && (
          <div className="empty">
            <p>Ask a question and LogMind plans sub-queries, retrieves grounded passages, and streams a cited answer — with the agent trace live as it works.</p>
          </div>
        )}
      </main>
    </div>
  );
}
