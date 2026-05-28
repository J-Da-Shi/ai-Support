import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import type { AskMode, ChunkPayload } from "../types";

interface Props {
  mode: AskMode | null;
  chunks: ChunkPayload[];
}

export function RetrievalPane({ mode, chunks }: Props) {
  if (!mode || chunks.length === 0) {
    return <div className="text-zinc-400 text-sm">等待查询…</div>;
  }
  const dim = mode === "fallback";
  return (
    <div className={dim ? "opacity-60" : ""}>
      {dim && (
        <div className="mb-2 text-xs text-amber-700 bg-amber-50 px-2 py-1 rounded">
          ⚠ 弱相关，仅供参考
        </div>
      )}
      {chunks.map((c) => (
        <div key={c.id} className="mb-4 bg-white rounded-lg shadow-sm p-3">
          <div className="text-xs text-zinc-500 mb-1 flex justify-between">
            <span>{c.heading_path.join(" > ")} · {c.file_path}</span>
            <span>score {c.score.toFixed(2)}</span>
          </div>
          <div className="text-sm">
            <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{c.text}</ReactMarkdown>
          </div>
        </div>
      ))}
    </div>
  );
}
