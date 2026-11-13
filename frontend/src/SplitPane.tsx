import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";

type Direction = "row" | "column";

function readRatio(key: string, fallback: number) {
  try {
    const raw = localStorage.getItem(key);
    const value = raw == null ? fallback : Number(raw);
    if (!Number.isFinite(value)) return fallback;
    return Math.min(0.85, Math.max(0.15, value));
  } catch {
    return fallback;
  }
}

export function SplitPane({
  direction,
  initial,
  storageKey,
  first,
  second,
  min = 140,
}: {
  direction: Direction;
  initial: number;
  storageKey: string;
  first: ReactNode;
  second: ReactNode;
  min?: number;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [ratio, setRatio] = useState(() => readRatio(storageKey, initial));

  useEffect(() => {
    try {
      localStorage.setItem(storageKey, String(ratio));
    } catch {
      /* private mode can reject storage */
    }
  }, [ratio, storageKey]);

  function clamp(next: number) {
    const bounds = container.current?.getBoundingClientRect();
    if (!bounds) return Math.min(0.85, Math.max(0.15, next));
    const span = direction === "row" ? bounds.width : bounds.height;
    const available = Math.max(span - 10, 1);
    const edge = Math.min(0.45, min / available);
    return Math.min(1 - edge, Math.max(edge, next));
  }

  function move(event: ReactPointerEvent<HTMLDivElement>) {
    if (!event.currentTarget.hasPointerCapture(event.pointerId) || !container.current) return;
    const bounds = container.current.getBoundingClientRect();
    const span = direction === "row" ? bounds.width : bounds.height;
    const start = direction === "row" ? bounds.left : bounds.top;
    const client = direction === "row" ? event.clientX : event.clientY;
    setRatio(clamp((client - start) / span));
  }

  return (
    <div
      ref={container}
      className={`split split-${direction}`}
      style={{ ["--pane" as string]: `calc((100% - var(--handle)) * ${ratio})` }}
    >
      <div className="split-pane">{first}</div>
      <div
        className="split-handle"
        role="separator"
        aria-label={direction === "row" ? "Resize columns" : "Resize rows"}
        aria-orientation={direction === "row" ? "vertical" : "horizontal"}
        aria-valuemin={15}
        aria-valuemax={85}
        aria-valuenow={Math.round(ratio * 100)}
        tabIndex={0}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          event.currentTarget.classList.add("active");
        }}
        onPointerMove={move}
        onPointerUp={(event) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            event.currentTarget.releasePointerCapture(event.pointerId);
          }
          event.currentTarget.classList.remove("active");
        }}
        onKeyDown={(event) => {
          const step = event.shiftKey ? 0.08 : 0.02;
          if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
            event.preventDefault();
            setRatio((current) => clamp(current - step));
          }
          if (event.key === "ArrowRight" || event.key === "ArrowDown") {
            event.preventDefault();
            setRatio((current) => clamp(current + step));
          }
        }}
      />
      <div className="split-pane">{second}</div>
    </div>
  );
}
