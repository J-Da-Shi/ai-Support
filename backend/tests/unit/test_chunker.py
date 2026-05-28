from pathlib import Path

import tiktoken

from core.chunker import chunk_markdown_file

FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes_sample"
_ENC = tiktoken.get_encoding("cl100k_base")


def test_chunk_splits_on_h2():
    chunks = chunk_markdown_file(
        FIXTURES / "mysql" / "事务隔离.md",
        relative_path="mysql/事务隔离.md",
        max_tokens=500,
        overlap_tokens=50,
    )
    assert len(chunks) == 2
    h2s = [c.heading_path[-1] for c in chunks]
    assert h2s == ["隔离级别有哪些", "RR vs RC"]
    assert all(c.heading_path[0] == "MySQL 事务隔离" for c in chunks)


def test_chunk_preserves_code_block():
    chunks = chunk_markdown_file(
        FIXTURES / "mysql" / "事务隔离.md",
        relative_path="mysql/事务隔离.md",
        max_tokens=500,
        overlap_tokens=50,
    )
    first = chunks[0].text
    assert "```sql" in first
    assert "READ COMMITTED" in first
    # 代码块未被切断（同一个 chunk 内开 ``` 闭 ```）
    assert first.count("```") % 2 == 0


def test_chunk_id_stable():
    a = chunk_markdown_file(
        FIXTURES / "mysql" / "事务隔离.md",
        relative_path="mysql/事务隔离.md",
        max_tokens=500, overlap_tokens=50,
    )
    b = chunk_markdown_file(
        FIXTURES / "mysql" / "事务隔离.md",
        relative_path="mysql/事务隔离.md",
        max_tokens=500, overlap_tokens=50,
    )
    assert [c.id for c in a] == [c.id for c in b]


def test_line_numbers_within_file():
    chunks = chunk_markdown_file(
        FIXTURES / "k8s" / "controller-runtime.md",
        relative_path="k8s/controller-runtime.md",
        max_tokens=500, overlap_tokens=50,
    )
    for c in chunks:
        assert c.line_start >= 1
        assert c.line_end >= c.line_start


def test_long_section_is_split_with_overlap(tmp_path):
    long_md = "# T\n\n## S\n\n" + ("段落一。" * 600 + "\n\n") + ("段落二。" * 600)
    p = tmp_path / "_tmp_long.md"
    p.write_text(long_md, encoding="utf-8")
    chunks = chunk_markdown_file(p, relative_path="_tmp_long.md", max_tokens=300, overlap_tokens=50)
    assert len(chunks) >= 2
    # overlap：相邻 chunk 至少有部分共同字符
    joined = chunks[0].text[-100:]
    assert any(joined[-30:] in c.text for c in chunks[1:])


# ---------- regression tests for code-review fixes ----------


def test_line_end_never_exceeds_file_line_count(tmp_path):
    """C1 regression: line_end must not walk past EOF after overlap is added."""
    # Build a file with multiple long paragraphs that will trigger overlap-prepend
    # in `_split_by_paragraph` via `_emit_chunks`.
    paragraphs = []
    for i in range(8):
        paragraphs.append(("段落%d内容句。" % i) * 80)
    content = "# Title\n\n## Section\n\n" + "\n\n".join(paragraphs) + "\n"
    p = tmp_path / "long.md"
    p.write_text(content, encoding="utf-8")
    file_line_count = len(content.splitlines())

    chunks = chunk_markdown_file(
        p, relative_path="long.md", max_tokens=200, overlap_tokens=40
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert c.line_end <= file_line_count, (
            f"chunk line_end={c.line_end} exceeds file line count {file_line_count}"
        )
        assert c.line_start >= 1
        assert c.line_end >= c.line_start


def test_placeholder_collision_in_prose(tmp_path):
    """C2 regression: literal " CODE0 " in prose must not be replaced by code."""
    content = (
        "# Doc\n"
        "\n"
        "## Section\n"
        "\n"
        "When debugging the device, the CODE0 register holds the last error.\n"
        "Note we sometimes spell it CODE0 in the docs as well.\n"
        "\n"
        "Below is a real code block:\n"
        "\n"
        "```python\n"
        "print('hello world')\n"
        "```\n"
        "\n"
        "After the block, CODE0 is referenced again in prose.\n"
    )
    p = tmp_path / "collision.md"
    p.write_text(content, encoding="utf-8")
    chunks = chunk_markdown_file(
        p, relative_path="collision.md", max_tokens=500, overlap_tokens=50
    )
    joined = "\n".join(c.text for c in chunks)
    # The prose mentions of CODE0 must be preserved verbatim.
    assert "the CODE0 register" in joined
    assert "spell it CODE0" in joined
    assert "CODE0 is referenced again" in joined
    # The real code block must round-trip intact.
    assert "```python" in joined
    assert "print('hello world')" in joined


def test_pre_h2_prologue_is_emitted(tmp_path):
    """I1 regression: content between H1 and first H2 must not be dropped."""
    content = (
        "# Top Title\n"
        "\n"
        "This is the prologue paragraph that sits between the H1 and the first H2.\n"
        "It contains a SENTINEL_PROLOGUE marker we will look for.\n"
        "\n"
        "## First Section\n"
        "\n"
        "Section body here. SENTINEL_SECTION marker.\n"
    )
    p = tmp_path / "prologue.md"
    p.write_text(content, encoding="utf-8")
    chunks = chunk_markdown_file(
        p, relative_path="prologue.md", max_tokens=500, overlap_tokens=50
    )
    joined = "\n".join(c.text for c in chunks)
    assert "SENTINEL_PROLOGUE" in joined, "prologue content must be emitted"
    assert "SENTINEL_SECTION" in joined
    # The prologue chunk's heading_path should be just the H1, with no H2.
    prologue_chunks = [c for c in chunks if "SENTINEL_PROLOGUE" in c.text]
    assert len(prologue_chunks) >= 1
    assert prologue_chunks[0].heading_path == ["Top Title"]


def test_oversize_single_sentence_is_sliced(tmp_path):
    """I2 regression: a single sentence > max_tokens must be sliced by tokens."""
    # Build one sentence with no punctuation that is much larger than max_tokens.
    sentence = "alpha " * 1200  # no terminal punctuation -> one "sentence"
    content = "# T\n\n## S\n\n" + sentence + "\n"
    p = tmp_path / "oversize.md"
    p.write_text(content, encoding="utf-8")

    max_tokens = 200
    overlap_tokens = 30
    chunks = chunk_markdown_file(
        p,
        relative_path="oversize.md",
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    assert len(chunks) >= 2
    # Each chunk's token count must fit within max_tokens + overlap_tokens
    # (overlap is prepended onto subsequent chunks).
    tolerance = max_tokens + overlap_tokens
    for c in chunks:
        n = len(_ENC.encode(c.text))
        assert n <= tolerance, f"chunk has {n} tokens, exceeds tolerance {tolerance}"
