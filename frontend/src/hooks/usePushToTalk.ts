import { useCallback, useEffect, useRef, useState } from "react";

interface Options {
  onTranscript: (text: string) => void;
  onError: (msg: string) => void;
  hotkey?: string; // 例如 "Space"
  lang?: string; // 例如 "zh-CN"
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

interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((ev: SpeechRecognitionEventLike) => void) | null;
  onerror: ((ev: { error: string; message?: string }) => void) | null;
  onend: (() => void) | null;
}

interface SpeechRecognitionEventLike {
  results: ArrayLike<{ 0: { transcript: string }; isFinal: boolean }> & {
    length: number;
  };
}

function getRecognitionCtor(): (new () => SpeechRecognitionLike) | null {
  const w = window as unknown as {
    SpeechRecognition?: new () => SpeechRecognitionLike;
    webkitSpeechRecognition?: new () => SpeechRecognitionLike;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function usePushToTalk({
  onTranscript,
  onError,
  hotkey = "Space",
  lang = "zh-CN",
}: Options) {
  const [recording, setRecording] = useState(false);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const finalTextRef = useRef<string>("");

  const start = useCallback(() => {
    if (recognitionRef.current) return;
    const Ctor = getRecognitionCtor();
    if (!Ctor) {
      onError("当前浏览器不支持语音识别（请用 Chrome / Edge）");
      return;
    }
    try {
      const rec = new Ctor();
      rec.lang = lang;
      rec.continuous = true;
      rec.interimResults = false;
      finalTextRef.current = "";
      rec.onresult = (ev) => {
        for (let i = 0; i < ev.results.length; i++) {
          const r = ev.results[i];
          if (r.isFinal) finalTextRef.current += r[0].transcript;
        }
      };
      rec.onerror = (ev) => {
        onError(`语音识别失败：${ev.error}`);
      };
      rec.onend = () => {
        const text = finalTextRef.current.trim();
        recognitionRef.current = null;
        setRecording(false);
        if (text) onTranscript(text);
      };
      recognitionRef.current = rec;
      rec.start();
      setRecording(true);
    } catch (e: unknown) {
      onError(`启动语音识别出错：${errMsg(e)}`);
      recognitionRef.current = null;
      setRecording(false);
    }
  }, [onTranscript, onError, lang]);

  const stop = useCallback(() => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        /* ignore */
      }
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
