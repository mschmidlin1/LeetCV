export type ProblemType = "code" | "critical_thinking";
export type ProblemStatus = "todo" | "attempted" | "solved";

export type ProblemSummary = {
  id: string;
  number: number;
  title: string;
  topic: string;
  type: ProblemType;
  status: ProblemStatus;
  test_count: number;
};

export type ProblemDetail = ProblemSummary & {
  prompt: string;
  starter: string;
  code: string;
  essay: string;
};

export type ReturnValue = {
  function: string;
  value?: string;
  raised?: string;
};

export type TestRow = {
  name: string;
  passed: boolean;
  traceback: string;
  stdout: string;
  stderr: string;
  returns: ReturnValue[];
  duration_ms: number;
};

export type GradeResult = {
  passed: boolean;
  status: ProblemStatus;
  tests: TestRow[];
  stdout: string;
  stderr: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* keep the status text */
    }
    throw new Error(typeof detail === "string" ? detail : "Request failed");
  }
  return response.json() as Promise<T>;
}

export const api = {
  list: () => request<ProblemSummary[]>("/api/problems"),
  problem: (id: string) => request<ProblemDetail>(`/api/problems/${id}`),
  saveCode: (id: string, code: string) =>
    request<{ status: ProblemStatus }>(`/api/problems/${id}/code`, {
      method: "PUT",
      body: JSON.stringify({ code }),
    }),
  saveEssay: (id: string, essay: string) =>
    request<{ status: ProblemStatus }>(`/api/problems/${id}/essay`, {
      method: "PUT",
      body: JSON.stringify({ essay }),
    }),
  solve: (id: string) =>
    request<{ status: ProblemStatus }>(`/api/problems/${id}/solve`, { method: "POST" }),
  run: (id: string, code: string) =>
    request<GradeResult>(`/api/problems/${id}/run`, {
      method: "POST",
      body: JSON.stringify({ code }),
    }),
  submit: (id: string, code: string) =>
    request<GradeResult>(`/api/problems/${id}/submit`, {
      method: "POST",
      body: JSON.stringify({ code }),
    }),
};
