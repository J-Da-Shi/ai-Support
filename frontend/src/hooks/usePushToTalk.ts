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
        const friendly: Record<string, string> = {
          "not-allowed":
            "麦克风权限被拒。检查：① Chrome 站点权限 chrome://settings/content/microphone ② macOS 系统设置 → 隐私 → 麦克风 + 语音识别（两个都要给 Chrome）",
          "service-not-allowed":
            "Chrome 的 Web Speech 服务被禁用。换成 Edge 试试，或在 chrome://flags 搜 'speech' 启用",
          "network":
            "Chrome Web Speech 需要联网（实际走 Google 服务），网络受限时会失败。可改用文字输入",
          "no-speech": "没有检测到语音，请说话后再松开",
          "audio-capture": "麦克风不可用或被其它应用占用",
          "aborted": "录音被中断",
        };
        const msg =
          friendly[ev.error] ?? `语音识别失败：${ev.error}${ev.message ? ` - ${ev.message}` : ""}`;
        onError(msg);
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
