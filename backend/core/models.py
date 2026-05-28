from dataclasses import dataclass, field
from enum import Enum


class AskMode(str, Enum):
    HIT = "hit"
    FALLBACK = "fallback"
    EMPTY = "empty"


@dataclass
class Chunk:
    id: str                          # 稳定 hash
    text: str
    file_path: str                   # 相对 notes_dir
    heading_path: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float                     # 0..1 归一化混合分
    vector_score: float = 0.0
    bm25_score: float = 0.0
