import { useCallback, useRef, useState } from "react";
import type { AskMode, ChunkPayload } from "../types";

export interface AskState {
  mode: AskMode | null;
  top1Score: number | null;
  chunks: ChunkPayload[];
  summary: string;
  status: "idle" | "loading" | "streaming" | "done" | "error";
  errorMessage?: string;
  elapsedMs?: number;
}

const initial: AskState = { mode: null, top1Score: null, chunks: [], summary: "", status: "idle" };

export function useAsk() {
  const [state, setState] = useState<AskState>(initial);
  const sourceRef = useRef<EventSource | null>(null);

  const abort = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
  }, []);

  const ask = useCallback((query: string) => {
    abort();
    setState({ ...initial, status: "loading" });
    const url = `/api/ask?query=${encodeURIComponent(query)}`;
    const es = new EventSource(url);
    sourceRef.current = es;

    es.addEventListener("mode", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, mode: d.mode, top1Score: d.top1_score, status: "streaming" }));
    });
    es.addEventListener("chunks", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, chunks: d }));
    });
    es.addEventListener("token", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, summary: s.summary + d }));
    });
    es.addEventListener("error", (e) => {
      const raw = (e as MessageEvent).data;
      let msg = "连接错误";
      if (raw) {
        try { msg = JSON.parse(raw).message ?? msg; } catch {}
      }
      setState((s) => ({ ...s, status: "error", errorMessage: msg }));
      es.close();
    });
    es.addEventListener("done", (e) => {
      const d = JSON.parse((e as MessageEvent).data || "{}");
      setState((s) => ({ ...s, status: "done", elapsedMs: d.elapsed_ms }));
      es.close();
    });
    es.onerror = () => {
      setState((s) => (s.status === "done" ? s : { ...s, status: "error", errorMessage: "SSE 中断" }));
      es.close();
    };
  }, [abort]);

  const reset = useCallback(() => {
    abort();
    setState(initial);
  }, [abort]);

  return { state, ask, abort, reset };
}
