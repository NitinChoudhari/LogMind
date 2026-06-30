export default function Header({ health, docs }) {
  return (
    <header className="header">
      <div className="brand">
        <span className="brand__mark">LOG<span className="brand__accent">MIND</span></span>
        {/* <span className="brand__sub">ask your documents</span> */}
      </div>
      <div className="status">
        {health && (
          <span className="pill" title="Set by PROVIDER in the backend .env">
            <span className="pill__dot" />
            {health.provider} · {health.model}
            {/* {health.reranker && health.reranker !== "none" && ` · rerank: ${health.reranker}`} */}
          </span>
        )}
        {docs && docs.index_ready ? (
          <span className="pill pill--quiet">
            {docs.documents.length} docs · {docs.chunks} chunks
          </span>
        ) : (
          <span className="pill pill--warn">index empty — run ingest.py</span>
        )}
      </div>
    </header>
  );
}
