// Thin API client. In dev, Vite proxies /api -> http://localhost:8000.

async function jsonOrThrow(res) {
  const data = await res.json();
  if (data && data.error) throw new Error(data.error);
  return data;
}

export async function getHealth() {
  return jsonOrThrow(await fetch("/api/health"));
}

export async function getDocuments() {
  return jsonOrThrow(await fetch("/api/documents"));
}

export async function postQuery(question) {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  return jsonOrThrow(res);
}

// Streaming query over Server-Sent Events. Calls handler callbacks as events
// arrive: onTrace, onSubqueries, onSources, onToken, onDone, onError.
export async function streamQuery(question, handlers = {}) {
  const res = await fetch("/api/query/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  // Validation/index errors come back as plain JSON, not a stream.
  const ctype = res.headers.get("content-type") || "";
  if (!res.ok || !res.body || ctype.includes("application/json")) {
    let msg = "Stream failed.";
    try {
      const d = await res.json();
      if (d && d.error) msg = d.error;
    } catch {}
    handlers.onError?.(msg);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const payload = raw.replace(/^data: ?/, "").trim();
      if (!payload) continue;
      let evt;
      try {
        evt = JSON.parse(payload);
      } catch {
        continue;
      }
      switch (evt.type) {
        case "trace": handlers.onTrace?.(evt.line); break;
        case "subqueries": handlers.onSubqueries?.(evt.items); break;
        case "sources": handlers.onSources?.(evt.sources); break;
        case "token": handlers.onToken?.(evt.text); break;
        case "done": handlers.onDone?.(evt); break;
        case "error": handlers.onError?.(evt.message); break;
        default: break;
      }
    }
  }
}
