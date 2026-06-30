import AgentTrace from "./AgentTrace.jsx";
import AnswerCard from "./AnswerCard.jsx";
import SourceList from "./SourceList.jsx";
import TraceConsole from "./TraceConsole.jsx";

// One question + answer in the thread: a right-aligned user bubble, then the
// assistant's content below it (answer, citations, and the sub-queries /
// sources / trace disclosure rows).
export default function Turn({ turn, onCite }) {
  const { question, subQueries, sources, answer, traceText, meta, error, streaming, active } = turn;

  const showAnswer = answer.length > 0 || (streaming && sources.length > 0);
  const thinking = streaming && subQueries.length === 0;

  return (
    <div className="turn">
      <div className="turn__user">
        <div className="bubble--user">{question}</div>
      </div>

      <div className="turn__assistant">
        {error && <div className="errorbox">{error}</div>}

        {thinking && (
          <div className="loader">
            <span className="loader__spin" />
            <span className="loader__stages">
              <span className="loader__stage is-on">contacting agents…</span>
            </span>
          </div>
        )}

        {subQueries.length > 0 && <AgentTrace subQueries={subQueries} />}

        {showAnswer && (
          <AnswerCard answer={answer} onCite={(n) => onCite(turn.id, n)} streaming={streaming} />
        )}

        <SourceList sources={sources} active={active} turnId={turn.id} />
        <TraceConsole text={traceText} streaming={streaming} />

        {meta && !streaming && (
          <div className="meta">
            {meta.provider} · {meta.model} · {sources.length} sources
          </div>
        )}
      </div>
    </div>
  );
}
