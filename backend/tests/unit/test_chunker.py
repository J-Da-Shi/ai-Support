from pathlib import Path
import pytest
from core.chunker import chunk_markdown_file
from core.models import Chunk

FIXTURES = Path(__file__).parent.parent / "fixtures" / "notes_sample"


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


def test_long_section_is_split_with_overlap():
    long_md = "# T\n\n## S\n\n" + ("段落一。" * 600 + "\n\n") + ("段落二。" * 600)
    p = FIXTURES.parent / "_tmp_long.md"
    p.write_text(long_md, encoding="utf-8")
    try:
        chunks = chunk_markdown_file(p, relative_path="_tmp_long.md", max_tokens=300, overlap_tokens=50)
        assert len(chunks) >= 2
        # overlap：相邻 chunk 至少有部分共同字符
        joined = chunks[0].text[-100:]
        assert any(joined[-30:] in c.text for c in chunks[1:])
    finally:
        p.unlink()
