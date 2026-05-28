import type { AskMode } from "../types";

type Status = "idle" | "loading" | "streaming" | "done" | "error";

interface Props {
  mode: AskMode | null;
  summary: string;
  status: Status;
  forced: boolean;
  lastQuery: string | null;
  onAskResume: () => void;
}

export function SummaryPane({
  mode,
  summary,
  status,
  forced,
  lastQuery,
  onAskResume,
}: Props) {
  const tag = !mode
    ? null
    : forced
    ? { text: "👤 基于简历生成", cls: "bg-violet-100 text-violet-800" }
    : mode === "hit"
    ? { text: "✓ 笔记命中", cls: "bg-emerald-100 text-emerald-800" }
    : mode === "fallback"
    ? {
        text: "🟡 笔记未直接命中，基于简历回答",
        cls: "bg-amber-100 text-amber-800",
      }
    : null;

  // 显示按钮：已经有结果（有 mode 或 status=done/streaming）+ 不在加载中 + 有 lastQuery
  const canAskResume =
    !!lastQuery &&
    (status === "done" || status === "streaming" || status === "error") &&
    !!mode;

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        {tag && (
          <div className={`text-xs px-2 py-1 rounded inline-block ${tag.cls}`}>
            {tag.text}
          </div>
        )}
        {canAskResume && !forced && (
          <button
            type="button"
            onClick={onAskResume}
            disabled={status === "streaming"}
            className="ml-auto text-xs px-3 py-1 rounded bg-violet-600 text-white
                       hover:bg-violet-700 disabled:bg-zinc-400
                       disabled:cursor-not-allowed transition"
            title="忽略命中结果，让 LLM 用简历重新生成回答"
          >
            👤 基于简历生成
          </button>
        )}
        {canAskResume && forced && (
          <button
            type="button"
            onClick={onAskResume}
            disabled={status === "streaming"}
            className="ml-auto text-xs px-3 py-1 rounded border border-violet-600 text-violet-700
                       hover:bg-violet-50 disabled:opacity-50
                       disabled:cursor-not-allowed transition"
            title="再来一次"
          >
            🔁 再生成一次
          </button>
        )}
      </div>
      <pre className="whitespace-pre-wrap font-sans text-sm leading-6 flex-1">
        {summary}
      </pre>
      {status === "loading" && (
        <p className="text-zinc-400 text-sm">检索中…</p>
      )}
      {status === "streaming" && summary === "" && (
        <p className="text-zinc-400 text-sm">生成中…</p>
      )}
    </div>
  );
}
