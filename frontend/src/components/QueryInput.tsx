import { useState } from "react";

interface Props {
  onSubmit: (q: string) => void;
  initialValue?: string;
}

export function QueryInput({ onSubmit, initialValue }: Props) {
  const [v, setV] = useState(initialValue ?? "");
  return (
    <form
      className="flex gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        const q = v.trim();
        if (q) onSubmit(q);
      }}
    >
      <input
        className="flex-1 px-3 py-2 rounded border border-zinc-300 focus:border-zinc-500 outline-none"
        placeholder="输入问题或松开热键…"
        value={v}
        onChange={(e) => setV(e.target.value)}
        autoFocus
      />
      <button className="px-4 py-2 rounded bg-zinc-900 text-white hover:bg-zinc-700" type="submit">
        提交
      </button>
    </form>
  );
}
