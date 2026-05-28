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
      {chunks.map((c) => {
        const title =
          c.heading_path[c.heading_path.length - 1] ?? c.file_path;
        const breadcrumb = c.heading_path.slice(0, -1).join(" › ");
        return (
          <div key={c.id} className="mb-4 bg-white rounded-lg shadow-sm p-4">
            <div className="mb-3">
              <div className="flex items-baseline justify-between gap-3 mb-1">
                <h3 className="text-xl font-bold text-zinc-900 leading-snug">
                  {title}
                </h3>
                <span className="shrink-0 text-xs text-zinc-500 font-mono">
                  score {c.score.toFixed(2)}
                </span>
              </div>
              <div className="text-xs text-zinc-500">
                {breadcrumb && <span>{breadcrumb} · </span>}
                <span>{c.file_path}</span>
              </div>
            </div>
            <div
              className="text-sm leading-relaxed
                [&_h1]:text-lg [&_h1]:font-bold [&_h1]:mt-3 [&_h1]:mb-1
                [&_h2]:text-base [&_h2]:font-bold [&_h2]:mt-3 [&_h2]:mb-1
                [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-1
                [&_p]:my-2
                [&_strong]:font-semibold [&_strong]:text-zinc-900
                [&_code]:bg-zinc-100 [&_code]:px-1 [&_code]:rounded
                [&_pre]:bg-zinc-50 [&_pre]:p-2 [&_pre]:rounded [&_pre]:my-2 [&_pre]:overflow-x-auto
                [&_pre_code]:bg-transparent [&_pre_code]:p-0
                [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-2
                [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:my-2"
            >
              <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                {c.text}
              </ReactMarkdown>
            </div>
          </div>
        );
      })}
    </div>
  );
}
