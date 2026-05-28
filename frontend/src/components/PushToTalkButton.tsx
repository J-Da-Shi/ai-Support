interface Props {
  recording: boolean;
  onMouseDown: () => void;
  onMouseUp: () => void;
}

export function PushToTalkButton({ recording, onMouseDown, onMouseUp }: Props) {
  return (
    <button
      className={`px-4 py-2 rounded text-white select-none ${
        recording ? "bg-red-600" : "bg-zinc-900 hover:bg-zinc-700"
      }`}
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onTouchStart={onMouseDown}
      onTouchEnd={onMouseUp}
      aria-pressed={recording}
    >
      {recording ? "● 录音中…" : "🎙️ 按住说话 (Space)"}
    </button>
  );
}
