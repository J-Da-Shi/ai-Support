import type { AskMode } from "../types";

interface Props {
  mode: AskMode | null;
  summary: string;
  status: "idle" | "loading" | "streaming" | "done" | "error";
}

export function SummaryPane({ mode, summary, status }: Props) {
  const tag =
    mode === "hit" ? { text: "✓ 笔记命中", cls: "bg-emerald-100 text-emerald-800" } :
    mode === "fallback" ? { text: "🟡 笔记未直接命中，基于简历回答", cls: "bg-amber-100 text-amber-800" } :
    null;
  return (
    <div>
      {tag && (
        <div className={`text-xs px-2 py-1 rounded inline-block mb-2 ${tag.cls}`}>{tag.text}</div>
      )}
      <pre className="whitespace-pre-wrap font-sans text-sm leading-6">{summary}</pre>
      {status === "loading" && <p className="text-zinc-400 text-sm">检索中…</p>}
      {status === "streaming" && summary === "" && <p className="text-zinc-400 text-sm">生成中…</p>}
    </div>
  );
}
