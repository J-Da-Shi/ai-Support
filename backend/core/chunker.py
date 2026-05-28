import hashlib
import re
from pathlib import Path

import tiktoken
from markdown_it import MarkdownIt

from core.models import Chunk

_enc = tiktoken.get_encoding("cl100k_base")

# Use Unicode private-use characters as code-block placeholder framing so the
# placeholder can never collide with anything that might appear in real prose.
# U+E000 and U+E001 are reserved for private use.
_CODE_PH_OPEN = ""
_CODE_PH_CLOSE = ""
_CODE_PH_RE = re.compile(rf"{_CODE_PH_OPEN}(\d+){_CODE_PH_CLOSE}")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _stable_id(file_path: str, line_start: int, text: str) -> str:
    h = hashlib.sha256()
    h.update(file_path.encode())
    h.update(str(line_start).encode())
    h.update(text.encode())
    return h.hexdigest()[:16]


def _slice_sentence_by_tokens(sentence: str, max_tokens: int) -> list[str]:
    """Slice a single oversize sentence into pieces that each fit max_tokens."""
    ids = _enc.encode(sentence)
    if len(ids) <= max_tokens:
        return [sentence]
    out: list[str] = []
    for i in range(0, len(ids), max_tokens):
        out.append(_enc.decode(ids[i : i + max_tokens]))
    return out


def _split_by_paragraph(
    text: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """段落级切分，超长按句切，附 overlap。

    Returns the final pieces (with overlap prepended). For computing original
    pre-overlap line counts, callers should use `_split_by_paragraph_with_spans`.
    """
    return [p for p, _ in _split_by_paragraph_with_spans(text, max_tokens, overlap_tokens)]


def _split_by_paragraph_with_spans(
    text: str, max_tokens: int, overlap_tokens: int
) -> list[tuple[str, int]]:
    """Like `_split_by_paragraph` but also returns the *pre-overlap* line count
    for each piece, so callers can advance a cursor without double-counting the
    overlap that was prepended onto subsequent pieces.

    Returns: list of (piece_text_with_overlap, pre_overlap_line_count).
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    buf: list[str] = []
    buf_tok = 0

    def flush_paragraph_buf() -> None:
        nonlocal buf, buf_tok
        if buf:
            pieces.append("\n\n".join(buf))
            buf, buf_tok = [], 0

    def flush_sentence_buf() -> None:
        # Sentences are already glued by their own punctuation, so we join with
        # an empty string instead of "\n\n" to preserve user intent (I3).
        nonlocal buf, buf_tok
        if buf:
            pieces.append("".join(buf))
            buf, buf_tok = [], 0

    for p in paragraphs:
        ptok = _count_tokens(p)
        if ptok > max_tokens:
            # Paragraph is too big; before switching modes, flush whatever
            # paragraph-mode content was buffered as its own piece (I4).
            flush_paragraph_buf()
            # 句子级再切
            sentences = re.split(r"(?<=[。！？.!?])", p)
            for s in sentences:
                if not s.strip():
                    continue
                stok = _count_tokens(s)
                if stok > max_tokens:
                    # Single sentence longer than max_tokens — flush current
                    # sentence buffer, then slice the sentence by tokens (I2).
                    flush_sentence_buf()
                    for slice_text in _slice_sentence_by_tokens(s, max_tokens):
                        pieces.append(slice_text)
                    continue
                if buf_tok + stok > max_tokens and buf:
                    flush_sentence_buf()
                buf.append(s)
                buf_tok += stok
            # After sentence-mode finishes for this paragraph, flush remaining
            # sentences so the next paragraph iteration starts clean (I4).
            flush_sentence_buf()
            continue
        if buf_tok + ptok > max_tokens and buf:
            flush_paragraph_buf()
        buf.append(p)
        buf_tok += ptok
    flush_paragraph_buf()

    # Record each piece's pre-overlap line count BEFORE prepending overlap.
    pre_overlap_line_counts = [p.count("\n") + 1 for p in pieces]

    if overlap_tokens > 0 and len(pieces) > 1:
        with_overlap = [pieces[0]]
        for prev, cur in zip(pieces, pieces[1:]):
            tail_ids = _enc.encode(prev)[-overlap_tokens:]
            tail = _enc.decode(tail_ids)
            with_overlap.append(tail + cur)
        pieces = with_overlap

    return list(zip(pieces, pre_overlap_line_counts))


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

    # Determine the body_start: first line after the H1, or 0 if no H1.
    body_start = 0
    if h1 is not None:
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("# "):
                body_start = i + 1
                break

    if not h2_marks:
        # 没有 H2，整篇当一段
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

    chunks: list[Chunk] = []

    # I1: emit any pre-H2 prologue between body_start and the first H2.
    first_h2_line0 = h2_marks[0][0]
    if first_h2_line0 > body_start:
        prologue_lines = lines[body_start:first_h2_line0]
        prologue_body = "\n".join(prologue_lines).strip()
        if prologue_body:
            prologue_heading = [h1] if h1 else []
            chunks.extend(
                _emit_chunks(
                    body=prologue_body,
                    heading_path=prologue_heading,
                    relative_path=relative_path,
                    line_start=body_start + 1,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )

    # 按 H2 切段
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
    pieces_with_spans = _protect_code_blocks_split_with_spans(
        body, max_tokens, overlap_tokens
    )
    out: list[Chunk] = []
    cursor = line_start
    body_line_count = body.count("\n") + 1
    body_end = line_start + body_line_count - 1
    for piece_text, pre_overlap_lines in pieces_with_spans:
        line_end = cursor + pre_overlap_lines - 1
        # Clamp to the body's actual end so line_end never exceeds the source.
        if line_end > body_end:
            line_end = body_end
        c = Chunk(
            id=_stable_id(relative_path, cursor, piece_text),
            text=piece_text.strip(),
            file_path=relative_path,
            heading_path=heading_path,
            line_start=cursor,
            line_end=line_end,
        )
        out.append(c)
        cursor += pre_overlap_lines
        if cursor > body_end:
            cursor = body_end
    return out


def _protect_code_blocks_split(
    body: str, max_tokens: int, overlap_tokens: int
) -> list[str]:
    """先把代码块抽出占位，按段切，再回填。"""
    return [p for p, _ in _protect_code_blocks_split_with_spans(body, max_tokens, overlap_tokens)]


def _protect_code_blocks_split_with_spans(
    body: str, max_tokens: int, overlap_tokens: int
) -> list[tuple[str, int]]:
    """Like `_protect_code_blocks_split` but also returns the pre-overlap line
    count for each piece (see `_split_by_paragraph_with_spans`)."""
    code_blocks: list[str] = []

    def stash(m):
        code_blocks.append(m.group(0))
        return f"{_CODE_PH_OPEN}{len(code_blocks) - 1}{_CODE_PH_CLOSE}"

    masked = re.sub(r"```[\s\S]*?```", stash, body)
    pieces_with_spans = _split_by_paragraph_with_spans(
        masked, max_tokens, overlap_tokens
    )

    def restore(text: str) -> str:
        return _CODE_PH_RE.sub(lambda m: code_blocks[int(m.group(1))], text)

    return [(restore(p), n) for p, n in pieces_with_spans]
