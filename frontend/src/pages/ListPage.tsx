import { useMemo, useState } from "react";
import { Link, useOutletContext } from "react-router-dom";
import type { ProblemStatus, ProblemType } from "../api";
import type { OutletContext } from "../App";

const TYPE_LABEL: Record<ProblemType, string> = {
  code: "Code",
  critical_thinking: "Critical thinking",
};

export function ListPage() {
  const { problems } = useOutletContext<OutletContext>();
  const [query, setQuery] = useState("");
  const [topic, setTopic] = useState("all");
  const [kind, setKind] = useState("all");
  const [status, setStatus] = useState("all");
  const topics = useMemo(
    () => [...new Set(problems.map((problem) => problem.topic))].sort(),
    [problems],
  );
  const visible = problems.filter((problem) => {
    const haystack = `${problem.number} ${problem.title} ${problem.topic}`.toLowerCase();
    if (query && !haystack.includes(query.toLowerCase())) return false;
    if (topic !== "all" && problem.topic !== topic) return false;
    if (kind !== "all" && problem.type !== kind) return false;
    if (status !== "all" && problem.status !== status) return false;
    return true;
  });

  return (
    <main className="list-page">
      <div className="filters">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search problems"
          aria-label="Search problems"
        />
        <select value={topic} onChange={(event) => setTopic(event.target.value)} aria-label="Topic">
          <option value="all">All topics</option>
          {topics.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        <select value={kind} onChange={(event) => setKind(event.target.value)} aria-label="Type">
          <option value="all">All types</option>
          <option value="code">Code</option>
          <option value="critical_thinking">Critical thinking</option>
        </select>
        <select value={status} onChange={(event) => setStatus(event.target.value)} aria-label="Status">
          <option value="all">All statuses</option>
          <option value="todo">Todo</option>
          <option value="attempted">Attempted</option>
          <option value="solved">Solved</option>
        </select>
        <span className="filter-count">{visible.length} shown</span>
      </div>
      <div className="table">
        <div className="row head">
          <span />
          <span>#</span>
          <span>Title</span>
          <span className="meta">Topic</span>
          <span className="meta">Type</span>
        </div>
        {visible.map((problem) => (
          <Link key={problem.id} to={`/problems/${problem.id}`} className="row">
            <StatusDot status={problem.status} />
            <span className="number">{problem.number}</span>
            <span className="title">{problem.title}</span>
            <span className="meta topic">
              {problem.topic}
              <span className="type-label">{TYPE_LABEL[problem.type]}</span>
            </span>
            <span className="meta type">{TYPE_LABEL[problem.type]}</span>
          </Link>
        ))}
      </div>
    </main>
  );
}

function StatusDot({ status }: { status: ProblemStatus }) {
  const label = status === "solved" ? "Solved" : status === "attempted" ? "Attempted" : "Todo";
  return <span className={`dot ${status}`} title={label} aria-label={label} />;
}
