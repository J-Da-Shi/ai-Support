import { useEffect, useRef, useState } from "react";
import { PushToTalkButton } from "./components/PushToTalkButton";
import { QueryInput } from "./components/QueryInput";
import { RetrievalPane } from "./components/RetrievalPane";
import { SummaryPane } from "./components/SummaryPane";
import { Toast } from "./components/Toast";
import { useAsk } from "./hooks/useAsk";
import { usePushToTalk } from "./hooks/usePushToTalk";

export default function App() {
  const [toast, setToast] = useState<string | null>(null);
  const { state, ask, reset } = useAsk({ onError: setToast });
  const [draft, setDraft] = useState<string>("");
  const lastTranscriptRef = useRef<string>("");

  const { recording, start, stop } = usePushToTalk({
    onTranscript: (text) => {
      lastTranscriptRef.current = text;
      setDraft(text);
      if (text.trim()) ask(text.trim());
    },
    onError: (msg) => setToast(msg),
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        reset();
        setDraft("");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [reset]);

  return (
    <div className="h-full flex flex-col p-4 gap-4 max-w-7xl mx-auto">
      <header className="flex items-center gap-3">
        <h1 className="text-lg font-semibold">面试实时辅助</h1>
        <span className="text-xs text-zinc-500">⌘K 清空 · 按住 Space 说话</span>
      </header>
      <div className="flex gap-2 items-center">
        <PushToTalkButton recording={recording} onMouseDown={start} onMouseUp={stop} />
        <div className="flex-1">
          <QueryInput key={draft} initialValue={draft} onSubmit={ask} />
        </div>
      </div>
      <main className="grid grid-cols-2 gap-4 flex-1 min-h-0">
        <section className="overflow-auto bg-zinc-100 rounded-lg p-3">
          <h2 className="text-xs uppercase text-zinc-500 mb-2">原文片段</h2>
          <RetrievalPane mode={state.mode} chunks={state.chunks} />
        </section>
        <section className="overflow-auto bg-zinc-100 rounded-lg p-3">
          <h2 className="text-xs uppercase text-zinc-500 mb-2">延展 / 简历兜底</h2>
          <SummaryPane mode={state.mode} summary={state.summary} status={state.status} />
        </section>
      </main>
      <Toast message={toast} onClose={() => setToast(null)} />
    </div>
  );
}
