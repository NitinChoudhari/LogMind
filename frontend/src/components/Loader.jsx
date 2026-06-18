import { useEffect, useState } from "react";

const STAGES = ["Planning sub-queries", "Retrieving evidence", "Synthesizing answer"];

export default function Loader() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setI((x) => Math.min(x + 1, STAGES.length - 1)), 1400);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="loader">
      <span className="loader__spin" />
      <div className="loader__stages">
        {STAGES.map((s, idx) => (
          <span key={s} className={"loader__stage" + (idx <= i ? " is-on" : "")}>
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}
