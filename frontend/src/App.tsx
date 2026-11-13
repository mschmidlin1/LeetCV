import { useCallback, useEffect, useState } from "react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { Route, Routes } from "react-router-dom";
import { api, type ProblemSummary } from "./api";
import { ListPage } from "./pages/ListPage";
import { ProblemPage } from "./pages/ProblemPage";

export type OutletContext = {
  problems: ProblemSummary[];
  reload: () => void;
};

export function App() {
  const [problems, setProblems] = useState<ProblemSummary[]>([]);
  const location = useLocation();
  const reload = useCallback(() => {
    api.list().then(setProblems).catch(() => setProblems([]));
  }, []);

  useEffect(() => {
    reload();
  }, [reload, location.pathname]);

  const solved = problems.filter((problem) => problem.status === "solved").length;

  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="mark">CV</span>
          LeetCV
        </Link>
        <p className="progress">
          <strong>{solved}</strong>
          <span> / {problems.length || "—"} solved</span>
        </p>
      </header>
      <Routes>
        <Route element={<Outlet context={{ problems, reload } satisfies OutletContext} />}>
          <Route index element={<ListPage />} />
          <Route path="problems/:id" element={<ProblemPage />} />
        </Route>
      </Routes>
    </div>
  );
}
