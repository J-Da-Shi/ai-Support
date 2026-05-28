import { useEffect, useState } from "react";
import { QueryInput } from "./components/QueryInput";
import { RetrievalPane } from "./components/RetrievalPane";
import { SummaryPane } from "./components/SummaryPane";
import { Toast } from "./components/Toast";
import { useAsk } from "./hooks/useAsk";

export default function App() {
  const { state, ask, reset } = useAsk();
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    if (state.status === "error" && state.errorMessage) setToast(state.errorMessage);
  }, [state.status, state.errorMessage]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        reset();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [reset]);

  return (
    <div className="h-full flex flex-col p-4 gap-4 max-w-7xl mx-auto">
      <header className="flex items-center gap-3">
        <h1 className="text-lg font-semibold">面试实时辅助</h1>
        <span className="text-xs text-zinc-500">⌘K 清空</span>
      </header>
      <QueryInput onSubmit={ask} />
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
