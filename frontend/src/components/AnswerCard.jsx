// Renders the answer text, turning [n] citation markers into clickable chips
// that highlight the matching source.
export default function AnswerCard({ answer, onCite, streaming }) {
  const parts = String(answer).split(/(\[\d+\])/g);
  return (
    <div className="answer">
      <span className="answer__label">answer{streaming ? " · streaming" : ""}</span>
      <div className="answer__body">
        {parts.map((part, i) => {
          const m = part.match(/^\[(\d+)\]$/);
          if (m) {
            const n = Number(m[1]);
            return (
              <button key={i} className="cite" onClick={() => onCite(n)} title={`Source ${n}`}>
                {n}
              </button>
            );
          }
          return <span key={i}>{part}</span>;
        })}
        {streaming && <span className="answer__cursor" />}
      </div>
    </div>
  );
}
