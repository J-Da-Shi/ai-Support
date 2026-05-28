import { useCallback, useEffect, useRef, useState } from "react";

interface Options {
  onTranscript: (text: string) => void;
  onError: (msg: string) => void;
  hotkey?: string; // 例如 "Space"
}

function errMsg(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}

export function usePushToTalk({ onTranscript, onError, hotkey = "Space" }: Options) {
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const start = useCallback(async () => {
    if (recorderRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const rec = new MediaRecorder(stream, { mimeType: "audio/webm" });
      chunksRef.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      rec.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const fd = new FormData();
        fd.append("file", blob, "voice.webm");
        try {
          const resp = await fetch("/api/asr", { method: "POST", body: fd });
          if (!resp.ok) throw new Error(`ASR ${resp.status}`);
          const j = await resp.json();
          onTranscript(j.text || "");
        } catch (e: unknown) {
          onError(`语音识别失败：${errMsg(e)}`);
        } finally {
          streamRef.current?.getTracks().forEach((t) => t.stop());
          streamRef.current = null;
          recorderRef.current = null;
          setRecording(false);
        }
      };
      recorderRef.current = rec;
      rec.start();
      setRecording(true);
    } catch (e: unknown) {
      onError(`无法访问麦克风：${errMsg(e)}`);
    }
  }, [onTranscript, onError]);

  const stop = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
  }, []);

  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (e.code === hotkey && !e.repeat) {
        e.preventDefault();
        start();
      }
    };
    const onUp = (e: KeyboardEvent) => {
      if (e.code === hotkey) {
        e.preventDefault();
        stop();
      }
    };
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, [hotkey, start, stop]);

  return { recording, start, stop };
}
