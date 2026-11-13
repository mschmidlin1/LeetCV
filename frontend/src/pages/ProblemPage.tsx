import { useEffect, useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import { Link, useOutletContext, useParams } from "react-router-dom";
import { api, type GradeResult, type ProblemDetail, type ProblemStatus } from "../api";
import type { OutletContext } from "../App";
import { SplitPane } from "../SplitPane";

export function ProblemPage() {
  const { id = "" } = useParams();
  const { problems, reload } = useOutletContext<OutletContext>();
  const bodyRef = useRef<HTMLDivElement>(null);
  const [problem, setProblem] = useState<ProblemDetail | null>(null);
  const [code, setCode] = useState("");
  const [essay, setEssay] = useState("");
  const [busy, setBusy] = useState<"run" | "submit" | "save" | "solve" | null>(null);
  const [result, setResult] = useState<GradeResult | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const dirty = useRef(false);
  const executeRef = useRef<(kind: "run" | "submit") => void>(() => undefined);
  const narrow = useNarrow();

  useEffect(() => {
    bodyRef.current?.scrollTo(0, 0);
  }, [id]);

  useEffect(() => {
    let cancelled = false;
    dirty.current = false;
    setResult(null);
    setMessage("");
    setError("");
    api
      .problem(id)
      .then((detail) => {
        if (cancelled) return;
        setProblem(detail);
        setCode(detail.code);
        setEssay(detail.essay);
      })
      .catch((reason: Error) => setError(reason.message));
    return () => {
      cancelled = true;
    };
  }, [id]);

  useEffect(() => {
    if (!problem || problem.type !== "code" || !dirty.current) return;
    const problemId = problem.id;
    const handle = window.setTimeout(() => {
      api
        .saveCode(problemId, code)
        .then((saved) => applyStatus(saved.status))
        .catch(() => undefined);
    }, 700);
    return () => window.clearTimeout(handle);
    // Re-run only when the draft changes. Status updates must not schedule another save.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [code, problem?.id, problem?.type]);

  function applyStatus(status: ProblemStatus) {
    setProblem((current) => (current ? { ...current, status } : current));
    reload();
  }

  async function execute(kind: "run" | "submit") {
    if (!problem || busy) return;
    setBusy(kind);
    setError("");
    setMessage("");
    try {
      const graded = kind === "run" ? await api.run(problem.id, code) : await api.submit(problem.id, code);
      setResult(graded);
      applyStatus(graded.status);
      if (graded.passed && kind === "submit") setMessage("Accepted");
      else if (graded.passed) setMessage("Sample case passed");
      else if (graded.tests.some((test) => test.traceback.includes("Timed out"))) setMessage("Time limit exceeded");
      else if (graded.tests.some((test) => test.traceback.includes("AssertionError"))) setMessage("Wrong answer");
      else setMessage("Runtime error");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Run failed");
    } finally {
      setBusy(null);
    }
  }

  executeRef.current = (kind) => {
    void execute(kind);
  };

  async function saveEssay() {
    if (!problem) return;
    setBusy("save");
    setError("");
    try {
      const saved = await api.saveEssay(problem.id, essay);
      applyStatus(saved.status);
      setMessage("Answer saved");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Save failed");
    } finally {
      setBusy(null);
    }
  }

  async function markSolved() {
    if (!problem) return;
    setBusy("solve");
    setError("");
    try {
      const saved = await api.solve(problem.id);
      applyStatus(saved.status);
      setMessage("Marked solved");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not mark solved");
    } finally {
      setBusy(null);
    }
  }

  if (error && !problem) {
    return (
      <main className="problem-missing">
        <p>{error}</p>
        <Link to="/">Back to problems</Link>
      </main>
    );
  }
  if (!problem) return <main className="problem-missing">Loading…</main>;

  const current = problem;
  const index = problems.findIndex((item) => item.id === current.id);
  const previous = index > 0 ? problems[index - 1] : null;
  const next = index >= 0 && index < problems.length - 1 ? problems[index + 1] : null;

  function keepDraft() {
    if (current.type === "code" && dirty.current) {
      void api.saveCode(current.id, code);
    }
    if (current.type === "critical_thinking" && essay !== current.essay) {
      void api.saveEssay(current.id, essay);
    }
  }

  const editor = problem.type === "code" ? (
    <div className="editor">
      <Editor
        height="100%"
        language="python"
        theme="vs-dark"
        value={code}
        onMount={(editorInstance, monaco) => {
          editorInstance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Quote, () => executeRef.current("run"));
          editorInstance.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => executeRef.current("submit"));
        }}
        onChange={(value) => {
          dirty.current = true;
          setCode(value ?? "");
        }}
        options={{
          fontSize: 14,
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          padding: { top: 12 },
          tabSize: 4,
          automaticLayout: true,
          fontFamily: "Cascadia Code, Consolas, monospace",
        }}
      />
    </div>
  ) : (
    <div className="essay-wrap">
      <textarea
        className="essay"
        value={essay}
        onChange={(event) => setEssay(event.target.value)}
        placeholder="Write your answer. Save keeps it as attempted. Mark solved is separate."
        aria-label="Written answer"
      />
    </div>
  );

  const workspace = (
    <>
      <div className="toolbar">
        {problem.type === "code" ? (
          <>
            <button type="button" onClick={() => execute("run")} disabled={busy !== null}>
              {busy === "run" ? "Running…" : "Run"}
            </button>
            <button type="button" className="primary" onClick={() => execute("submit")} disabled={busy !== null}>
              {busy === "submit" ? "Submitting…" : "Submit"}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => {
                dirty.current = true;
                setCode(problem.starter);
              }}
              disabled={busy !== null}
            >
              Reset
            </button>
            <span className="hint">Ctrl+' run · Ctrl+Enter submit</span>
          </>
        ) : (
          <>
            <button type="button" onClick={saveEssay} disabled={busy !== null}>
              {busy === "save" ? "Saving…" : "Save"}
            </button>
            <button type="button" className="primary" onClick={markSolved} disabled={busy !== null}>
              {busy === "solve" ? "Saving…" : "Mark solved"}
            </button>
          </>
        )}
        {message && <span className={`banner ${result && !result.passed ? "bad" : "good"}`}>{message}</span>}
        {error && <span className="banner bad">{error}</span>}
      </div>
      {problem.type === "code" ? (
        <SplitPane
          direction="column"
          initial={0.62}
          storageKey="leetcv-results"
          first={editor}
          second={<Results result={result} />}
        />
      ) : (
        editor
      )}
    </>
  );

  return (
    <main className="workspace">
      <SplitPane
        direction={narrow ? "column" : "row"}
        initial={narrow ? 0.38 : 0.4}
        storageKey={narrow ? "leetcv-main-narrow" : "leetcv-main"}
        first={
          <section className="description">
            <div className="problem-body" ref={bodyRef}>
              <div className="problem-heading">
                <p className="eyebrow">
                  {problem.number}. {problem.topic}
                </p>
                <h1>{renderInline(problem.title)}</h1>
                <div className="tags">
                  <span className={`pill ${problem.status}`}>{labelStatus(problem.status)}</span>
                  <span className="pill">{problem.type === "code" ? "Code" : "Critical thinking"}</span>
                  {problem.type === "code" && <span className="pill">{problem.test_count} tests</span>}
                </div>
              </div>
              <Prompt title={problem.title} text={problem.prompt} />
            </div>
            <nav className="problem-nav" aria-label="Problems">
              <ProblemLink problem={previous} direction="previous" onLeave={keepDraft} />
              <ProblemLink problem={next} direction="next" onLeave={keepDraft} />
            </nav>
          </section>
        }
        second={<section className="work">{workspace}</section>}
      />
    </main>
  );
}

function Results({ result }: { result: GradeResult | null }) {
  if (!result) {
    return <div className="results results-empty">Run checks the sample case. Submit runs every test.</div>;
  }
  return (
    <div className="results">
      <div className="results-head">
        {result.tests.filter((test) => test.passed).length} / {result.tests.length} passed
      </div>
      {result.tests.map((test) => (
        <article key={test.name} className={test.passed ? "pass" : "fail"}>
          <header>
            <strong>{prettyName(test.name)}</strong>
            <span>{test.duration_ms} ms</span>
          </header>
          <Returns items={test.returns ?? []} />
          <Stream label="stdout" text={test.stdout ?? ""} />
          <Stream label="stderr" text={test.stderr ?? ""} />
          {test.traceback ? <Stream label="Traceback" text={test.traceback} /> : null}
        </article>
      ))}
    </div>
  );
}

function Returns({ items }: { items: { function: string; value?: string; raised?: string }[] }) {
  return (
    <div className="io">
      <p className="io-label">Returned</p>
      {items.length === 0 ? (
        <p className="io-empty">The function was not called.</p>
      ) : (
        items.map((item, index) => (
          <pre key={index}>
            {item.function}: {item.raised ? `raised ${item.raised}` : item.value}
          </pre>
        ))
      )}
    </div>
  );
}

function Stream({ label, text }: { label: string; text: string }) {
  return (
    <div className="io">
      <p className="io-label">{label}</p>
      {text.trim() ? <pre>{text}</pre> : <p className="io-empty">empty</p>}
    </div>
  );
}

function ProblemLink({
  problem,
  direction,
  onLeave,
}: {
  problem: { id: string; number: number; title: string } | null;
  direction: "previous" | "next";
  onLeave: () => void;
}) {
  const label = direction === "previous" ? "Previous" : "Next";
  if (!problem) {
    return (
      <button type="button" className={`nav-btn ${direction}`} disabled>
        {direction === "previous" && <span aria-hidden="true">←</span>}
        {label}
        {direction === "next" && <span aria-hidden="true">→</span>}
      </button>
    );
  }
  return (
    <Link
      className={`nav-btn ${direction}`}
      to={`/problems/${problem.id}`}
      aria-label={`${label}: ${problem.number}. ${problem.title}`}
      title={`${problem.number}. ${problem.title}`}
      onClick={onLeave}
    >
      {direction === "previous" && <span aria-hidden="true">←</span>}
      {label}
      {direction === "next" && <span aria-hidden="true">→</span>}
    </Link>
  );
}

function useNarrow() {
  const [narrow, setNarrow] = useState(() => window.matchMedia("(max-width: 800px)").matches);
  useEffect(() => {
    const query = window.matchMedia("(max-width: 800px)");
    const apply = () => setNarrow(query.matches);
    query.addEventListener("change", apply);
    return () => query.removeEventListener("change", apply);
  }, []);
  return narrow;
}

function labelStatus(status: ProblemStatus) {
  if (status === "solved") return "Solved";
  if (status === "attempted") return "Attempted";
  return "Todo";
}

function prettyName(name: string) {
  return name.replace(/^test_/, "").replaceAll("_", " ");
}

function Prompt({ title, text }: { title: string; text: string }) {
  const rest = text.startsWith(title) ? text.slice(title.length).trim() : text;
  const body = rest.replace(/^\.\s*/, "");
  if (!body) return null;
  return (
    <div className="prompt">
      <p>{renderInline(body)}</p>
    </div>
  );
}

function renderInline(text: string) {
  return text.split(/(`[^`]+`)/g).map((part, index) =>
    part.startsWith("`") && part.endsWith("`") ? (
      <code key={index}>{part.slice(1, -1)}</code>
    ) : (
      <span key={index}>{part}</span>
    ),
  );
}
