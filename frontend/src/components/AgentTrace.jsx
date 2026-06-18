export default function AgentTrace({ subQueries }) {
  if (!subQueries || !subQueries.length) return null;
  const single = subQueries.length === 1;
  return (
    <div className="trace">
      <span className="trace__label">
        {single ? "Strategist · direct query" : `Strategist · planned ${subQueries.length} sub-queries`}
      </span>
      <div className="trace__chips">
        {subQueries.map((q, i) => (
          <span className="trace__chip" key={i}>{q}</span>
        ))}
      </div>
    </div>
  );
}
