export type AskMode = "hit" | "fallback" | "empty";

export interface ChunkPayload {
  id: string;
  file_path: string;
  heading_path: string[];
  text: string;
  score: number;
  line_start: number;
  line_end: number;
}

export type AskEvent =
  | { event: "mode"; data: { mode: AskMode; top1_score: number; elapsed_ms: number } }
  | { event: "chunks"; data: ChunkPayload[] }
  | { event: "token"; data: string }
  | { event: "error"; data: { stage: string; message: string; recoverable: boolean } }
  | { event: "done"; data: { elapsed_ms: number } };
