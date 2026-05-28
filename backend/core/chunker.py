import hashlib
import re
from pathlib import Path

import tiktoken
from markdown_it import MarkdownIt

from core.models import Chunk

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _stable_id(file_path: str, line_start: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(file_path.encode())
    h.update(str(line_start).encode())
    h.update(text.encode())
    return h.hexdigest()[:16]


def _split_by_paragraph(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """段落级切分，超长按句切，附 overlap。"""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for p in paragraphs:
        ptok = _count_tokens(p)
        if ptok > max_tokens:
            # 句子级再切
            sentences = re.split(r"(?<=[。！？.!?])", p)
            for s in sentences:
                if not s.strip():
                    continue
                stok = _count_tokens(s)
                if buf_tok + stok > max_tokens and buf:
                    pieces.append("".join(buf))
                    buf, buf_tok = [], 0
                buf.append(s)
                buf_tok += stok
            continue
        if buf_tok + ptok > max_tokens and buf:
            pieces.append("\n\n".join(buf))
            buf, buf_tok = [], 0
        buf.append(p)
        buf_tok += ptok
    if buf:
        pieces.append("\n\n".join(buf))

    if overlap_tokens > 0 and len(pieces) > 1:
        with_overlap = [pieces[0]]
        for prev, cur in zip(pieces, pieces[1:]):
            tail_ids = _enc.encode(prev)[-overlap_tokens:]
            tail = _enc.decode(tail_ids)
            with_overlap.append(tail + cur)
        pieces = with_overlap
    return pieces


def chunk_markdown_file(
    path: Path,
    relative_path: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """按 H2 切大段；代码块整体保留；超长再细分；附 overlap。"""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    md = MarkdownIt("commonmark")
    tokens = md.parse(text)

    # 找 H1（作为面包屑顶层）
    h1: str | None = None
    for i, t in enumerate(tokens):
        if t.type == "heading_open" and t.tag == "h1":
            inline = tokens[i + 1]
            h1 = inline.content.strip()
            break

    # 找所有 H2 的位置（行号），作为分段锚点
    h2_marks: list[tuple[int, str]] = []  # (line_index_0based, title)
    for i, t in enumerate(tokens):
        if t.type == "heading_open" and t.tag == "h2":
            inline = tokens[i + 1]
            line0 = t.map[0] if t.map else 0
            h2_marks.append((line0, inline.content.strip()))

    if not h2_marks:
        # 没有 H2，整篇当一段
        body_start = 0
        if h1 is not None:
            for i, ln in enumerate(lines):
                if ln.lstrip().startswith("# "):
                    body_start = i + 1
                    break
        body = "\n".join(lines[body_start:]).strip()
        if not body:
            return []
        heading_path = [h1] if h1 else []
        return _emit_chunks(
            body=body,
            heading_path=heading_path,
            relative_path=relative_path,
            line_start=body_start + 1,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

    # 按 H2 切段
    chunks: list[Chunk] = []
    for idx, (line0, title) in enumerate(h2_marks):
        next_line0 = h2_marks[idx + 1][0] if idx + 1 < len(h2_marks) else len(lines)
        section_lines = lines[line0:next_line0]
        # 跳过 H2 自己的那一行
        body_lines = section_lines[1:]
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        heading_path = [h1, title] if h1 else [title]
        section_chunks = _emit_chunks(
            body=body,
            heading_path=heading_path,
            relative_path=relative_path,
            line_start=line0 + 2,    # 1-based，且跳过 H2 行
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        chunks.extend(section_chunks)
    return chunks


def _emit_chunks(
    body: str,
    heading_path: list[str],
    relative_path: str,
    line_start: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """把一段正文切成若干 Chunk，保护代码块完整。"""
    pieces = _protect_code_blocks_split(body, max_tokens, overlap_tokens)
    out: list[Chunk] = []
    cursor = line_start
    for p in pieces:
        line_count = p.count("\n") + 1
        c = Chunk(
            id=_stable_id(relative_path, cursor, p),
            text=p.strip(),
            file_path=relative_path,
            heading_path=heading_path,
            line_start=cursor,
            line_end=cursor + line_count - 1,
        )
        out.append(c)
        cursor += line_count
    return out


def _protect_code_blocks_split(body: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """先把代码块抽出占位，按段切，再回填。"""
    code_blocks: list[str] = []

    def stash(m):
        code_blocks.append(m.group(0))
        return f" CODE{len(code_blocks)-1} "

    masked = re.sub(r"```[\s\S]*?```", stash, body)
    pieces = _split_by_paragraph(masked, max_tokens, overlap_tokens)

    def restore(text: str) -> str:
        return re.sub(
            r" CODE(\d+) ",
            lambda m: code_blocks[int(m.group(1))],
            text,
        )

    return [restore(p) for p in pieces]
