import { useEffect } from "react";

interface Props {
  message: string | null;
  onClose: () => void;
}

export function Toast({ message, onClose }: Props) {
  useEffect(() => {
    if (!message) return;
    const t = setTimeout(onClose, 3000);
    return () => clearTimeout(t);
  }, [message, onClose]);
  if (!message) return null;
  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-zinc-900 text-white text-sm px-4 py-2 rounded shadow-lg">
      {message}
    </div>
  );
}
