import { useCallback, useEffect, useRef, useState } from "react";
import type { AskMode, ChunkPayload } from "../types";

export interface AskState {
  mode: AskMode | null;
  top1Score: number | null;
  chunks: ChunkPayload[];
  summary: string;
  status: "idle" | "loading" | "streaming" | "done" | "error";
  errorMessage?: string;
  elapsedMs?: number;
  forced: boolean; // true when last request was mode_override=resume
  lastQuery: string | null;
}

export interface UseAskOptions {
  onError?: (msg: string) => void;
}

export type AskModeOverride = "resume" | undefined;

const initial: AskState = {
  mode: null,
  top1Score: null,
  chunks: [],
  summary: "",
  status: "idle",
  forced: false,
  lastQuery: null,
};

export function useAsk(options: UseAskOptions = {}) {
  const [state, setState] = useState<AskState>(initial);
  const sourceRef = useRef<EventSource | null>(null);
  const onErrorRef = useRef(options.onError);
  useEffect(() => {
    onErrorRef.current = options.onError;
  }, [options.onError]);

  const abort = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
  }, []);

  const ask = useCallback(
    (query: string, modeOverride?: AskModeOverride) => {
      abort();
      setState({
        ...initial,
        status: "loading",
        forced: modeOverride === "resume",
        lastQuery: query,
      });
      let url = `/api/ask?query=${encodeURIComponent(query)}`;
      if (modeOverride) url += `&mode_override=${modeOverride}`;
      const es = new EventSource(url);
      sourceRef.current = es;

      es.addEventListener("mode", (e) => {
        const d = JSON.parse((e as MessageEvent).data);
        setState((s) => ({
          ...s,
          mode: d.mode,
          top1Score: d.top1_score,
          status: "streaming",
        }));
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
          try {
            msg = JSON.parse(raw).message ?? msg;
          } catch {
            /* keep default msg */
          }
        }
        onErrorRef.current?.(msg);
        setState((s) => ({ ...s, status: "error", errorMessage: msg }));
        es.close();
        sourceRef.current = null;
      });
      es.addEventListener("done", (e) => {
        const d = JSON.parse((e as MessageEvent).data || "{}");
        setState((s) => ({ ...s, status: "done", elapsedMs: d.elapsed_ms }));
        es.close();
        sourceRef.current = null;
      });
      es.onerror = () => {
        setState((s) => {
          if (s.status === "done") return s;
          onErrorRef.current?.("SSE 中断");
          return { ...s, status: "error", errorMessage: "SSE 中断" };
        });
        es.close();
        sourceRef.current = null;
      };
    },
    [abort],
  );

  const reset = useCallback(() => {
    abort();
    setState(initial);
  }, [abort]);

  return { state, ask, abort, reset };
}
