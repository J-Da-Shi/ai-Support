# 面试实时辅助 RAG 系统 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建本地 Web 应用：候选人面试中通过语音/文字查询本地 Markdown 笔记，命中时双栏显示原文 + LLM 延展，未命中时基于个人简历用第一人称兜底生成。

**Architecture:** FastAPI 后端 + React 前端，单进程单用户。Markdown 切块嵌入入 FAISS in-memory，向量+BM25 混合检索；SSE 流式输出；LLM provider 可插拔（默认 DeepSeek）。

**Tech Stack:** Python 3.11+ / FastAPI / FAISS / rank-bm25 / jieba / markdown-it-py / OpenAI Embedding & Whisper API / DeepSeek API；React 18 + Vite + TypeScript + Tailwind + react-markdown。

**Spec:** [`docs/superpowers/specs/2026-05-28-interview-rag-design.md`](../specs/2026-05-28-interview-rag-design.md)

---

## File Structure

### 后端 `backend/`

```
backend/
├─ pyproject.toml          # 依赖与工具配置
├─ pytest.ini              # pytest 配置
├─ main.py                 # FastAPI app 装配，启动时加载索引/简历
├─ routes/
│   ├─ __init__.py
│   ├─ asr.py              # POST /api/asr
│   └─ ask.py              # GET  /api/ask (SSE)
├─ core/
│   ├─ __init__.py
│   ├─ config.py           # .env 加载，pydantic Settings
│   ├─ models.py           # Chunk、AskMode 等 dataclass/Enum
│   ├─ chunker.py          # Markdown → Chunk 列表（纯函数）
│   ├─ embedder.py         # 嵌入客户端（OpenAI），含批量 + 缓存
│   ├─ indexer.py          # 扫描笔记 → 调 chunker/embedder → FAISS 持久化
│   ├─ retriever.py        # 向量+BM25 混合检索 + 阈值判定
│   ├─ asr.py              # Whisper API 客户端
│   ├─ resume.py           # 简历加载
│   ├─ prompts.py          # 命中/未命中 prompt 模板
│   └─ llm/
│       ├─ __init__.py
│       ├─ base.py         # LLMProvider Protocol
│       ├─ deepseek.py
│       ├─ openai.py
│       └─ claude.py
└─ tests/
    ├─ unit/
    │   ├─ test_chunker.py
    │   ├─ test_retriever.py
    │   ├─ test_llm_providers.py
    │   ├─ test_asr.py
    │   └─ test_config.py
    ├─ integration/
    │   └─ test_ask_flow.py
    └─ fixtures/
        └─ notes_sample/   # 测试用小笔记
```

### 前端 `frontend/`

```
frontend/
├─ package.json
├─ tsconfig.json
├─ vite.config.ts
├─ tailwind.config.js
├─ postcss.config.js
├─ index.html
└─ src/
    ├─ main.tsx
    ├─ App.tsx
    ├─ types.ts             # Chunk、AskEvent 等 TS 类型
    ├─ hooks/
    │   ├─ useAsk.ts        # SSE 订阅 + 状态机
    │   └─ usePushToTalk.ts # MediaRecorder + 热键
    ├─ components/
    │   ├─ PushToTalkButton.tsx
    │   ├─ QueryInput.tsx
    │   ├─ RetrievalPane.tsx
    │   ├─ SummaryPane.tsx
    │   └─ Toast.tsx
    └─ index.css            # Tailwind 入口
```

### 顶层

```
RAG/
├─ .env.example
├─ .gitignore
├─ README.md
├─ notes/                   # gitignore，用户笔记
├─ resume.md                # gitignore，用户简历
├─ backend/                 # ↑
└─ frontend/                # ↑
```

---

## Task 0: 项目初始化与 git

**Files:**
- Create: `/Users/mi/Desktop/RAG/.gitignore`
- Create: `/Users/mi/Desktop/RAG/README.md`
- Create: `/Users/mi/Desktop/RAG/.env.example`

- [ ] **Step 1: 初始化 git 仓库**

```bash
cd /Users/mi/Desktop/RAG
git init
git add docs/
git commit -m "docs: add design spec and implementation plan"
```

- [ ] **Step 2: 写 `.gitignore`**

```
# Python
__pycache__/
*.pyc
.venv/
venv/
.pytest_cache/
*.egg-info/

# Node
node_modules/
dist/
.vite/

# 数据与日志
backend/data/
backend/logs/
frontend/dist/

# 用户内容
notes/
resume.md
.env
.env.local

# 编辑器
.vscode/
.idea/
.DS_Store
```

- [ ] **Step 3: 写 `.env.example`**

```
# 笔记与简历
NOTES_DIR=./notes
RESUME_PATH=./resume.md

# API Keys
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
ANTHROPIC_API_KEY=

# 模型
EMBEDDING_MODEL=text-embedding-3-small
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
ASR_PROVIDER=openai
ASR_MODEL=whisper-1

# 检索
RETRIEVAL_TOP_K=8
RERANK_TOP_K=3
RETRIEVAL_THRESHOLD=0.5
VECTOR_WEIGHT=0.7
BM25_WEIGHT=0.3
MAX_CHUNK_TOKENS=500
CHUNK_OVERLAP_TOKENS=50

# 超时
ASR_TIMEOUT_S=8
EMBED_TIMEOUT_S=5
LLM_FIRST_TOKEN_TIMEOUT_S=10
LLM_TOTAL_TIMEOUT_S=30
```

- [ ] **Step 4: 写 `README.md`（最小可用版）**

````markdown
# 面试实时辅助 RAG

候选人面试中实时查询本地 Markdown 笔记，命中显示原文 + LLM 延展，未命中基于简历兜底生成。

## 启动

```bash
# 1. 配置
cp .env.example .env
# 编辑 .env 填入 API Key

# 2. 后端
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m core.indexer --rebuild   # 首次构建索引
uvicorn main:app --reload

# 3. 前端
cd ../frontend
npm install
npm run dev
```

浏览器打开 http://localhost:5173。
````

- [ ] **Step 5: 提交**

```bash
git add .gitignore README.md .env.example
git commit -m "chore: scaffold project (gitignore, readme, env example)"
```

---

## Task 1: 后端 Python 项目骨架与配置加载

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/pytest.ini`
- Create: `backend/main.py`
- Create: `backend/core/__init__.py`
- Create: `backend/core/config.py`
- Test: `backend/tests/unit/test_config.py`

- [ ] **Step 1: 写 `backend/pyproject.toml`**

```toml
[project]
name = "interview-rag-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.32",
  "python-dotenv>=1.0",
  "pydantic>=2.9",
  "pydantic-settings>=2.5",
  "markdown-it-py>=3.0",
  "faiss-cpu>=1.9",
  "rank-bm25>=0.2.2",
  "jieba>=0.42",
  "openai>=1.50",
  "httpx>=0.27",
  "anthropic>=0.40",
  "tiktoken>=0.8",
  "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3",
  "pytest-asyncio>=0.24",
  "respx>=0.21",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["core*", "routes*"]
```

- [ ] **Step 2: 写 `backend/pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
addopts = -ra --tb=short
```

- [ ] **Step 3: 安装依赖**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

- [ ] **Step 4: 写失败测试 `tests/unit/test_config.py`**

```python
import os
from pathlib import Path
import pytest
from core.config import Settings, load_settings


def test_load_settings_from_env(tmp_path, monkeypatch):
    notes = tmp_path / "notes"
    notes.mkdir()
    resume = tmp_path / "resume.md"
    resume.write_text("hi", encoding="utf-8")
    monkeypatch.setenv("NOTES_DIR", str(notes))
    monkeypatch.setenv("RESUME_PATH", str(resume))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dk-test")
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.42")

    s = load_settings()
    assert s.notes_dir == notes
    assert s.resume_path == resume
    assert s.retrieval_threshold == 0.42
    assert s.llm_provider == "deepseek"
    assert s.embedding_model == "text-embedding-3-small"


def test_missing_api_key_for_provider_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("NOTES_DIR", str(tmp_path))
    monkeypatch.setenv("RESUME_PATH", str(tmp_path / "resume.md"))
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        load_settings().validate_for_runtime()
```

- [ ] **Step 5: 运行测试验证失败**

Run: `pytest tests/unit/test_config.py -v`
Expected: FAIL — `core.config` 不存在

- [ ] **Step 6: 写 `core/__init__.py`（空）和 `core/config.py`**

```python
# core/__init__.py
```

```python
# core/config.py
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 路径
    notes_dir: Path = Field(default=Path("./notes"))
    resume_path: Path = Field(default=Path("./resume.md"))

    # API keys
    openai_api_key: str = ""
    deepseek_api_key: str = ""
    anthropic_api_key: str = ""

    # 模型
    embedding_model: str = "text-embedding-3-small"
    llm_provider: str = "deepseek"      # deepseek | openai | claude
    llm_model: str = "deepseek-chat"
    asr_provider: str = "openai"
    asr_model: str = "whisper-1"

    # 检索
    retrieval_top_k: int = 8
    rerank_top_k: int = 3
    retrieval_threshold: float = 0.5
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    max_chunk_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # 超时
    asr_timeout_s: float = 8.0
    embed_timeout_s: float = 5.0
    llm_first_token_timeout_s: float = 10.0
    llm_total_timeout_s: float = 30.0

    def validate_for_runtime(self) -> "Settings":
        """检查所选 provider 的 API key 是否就位。"""
        provider_key = {
            "deepseek": ("DEEPSEEK_API_KEY", self.deepseek_api_key),
            "openai": ("OPENAI_API_KEY", self.openai_api_key),
            "claude": ("ANTHROPIC_API_KEY", self.anthropic_api_key),
        }
        if self.llm_provider not in provider_key:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.llm_provider}")
        name, val = provider_key[self.llm_provider]
        if not val:
            raise ValueError(f"{name} required for LLM_PROVIDER={self.llm_provider}")
        if self.asr_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for ASR_PROVIDER=openai")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for embeddings")
        return self


def load_settings() -> Settings:
    return Settings()
```

- [ ] **Step 7: 运行测试验证通过**

Run: `pytest tests/unit/test_config.py -v`
Expected: PASS（两个用例）

- [ ] **Step 8: 写 `backend/main.py`（最小可启）**

```python
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.config import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    try:
        settings.validate_for_runtime()
        log.info("Settings validated. provider=%s notes_dir=%s", settings.llm_provider, settings.notes_dir)
    except ValueError as e:
        log.warning("Settings invalid (server still up for guidance): %s", e)
    app.state.settings = settings
    yield


app = FastAPI(title="Interview RAG", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"ok": True}
```

- [ ] **Step 9: 验证启动**

```bash
cd backend
OPENAI_API_KEY=x DEEPSEEK_API_KEY=y uvicorn main:app --port 8000 &
sleep 1
curl -s http://localhost:8000/api/health
kill %1
```
Expected: `{"ok":true}`

- [ ] **Step 10: 提交**

```bash
git add backend/pyproject.toml backend/pytest.ini backend/main.py backend/core/__init__.py backend/core/config.py backend/tests/unit/test_config.py
git commit -m "feat(backend): scaffold FastAPI app with config loader"
```

---

## Task 2: 数据模型与 Markdown 切块器

**Files:**
- Create: `backend/core/models.py`
- Create: `backend/core/chunker.py`
- Test: `backend/tests/unit/test_chunker.py`
- Create: `backend/tests/fixtures/notes_sample/mysql/事务隔离.md`
- Create: `backend/tests/fixtures/notes_sample/k8s/controller-runtime.md`
- Create: `backend/tests/fixtures/notes_sample/项目/订单系统复盘.md`

- [ ] **Step 1: 写 `core/models.py`**

```python
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
```

- [ ] **Step 2: 准备 fixtures**

`backend/tests/fixtures/notes_sample/mysql/事务隔离.md`:

````markdown
# MySQL 事务隔离

## 隔离级别有哪些

四种：READ UNCOMMITTED、READ COMMITTED、REPEATABLE READ、SERIALIZABLE。
MySQL 默认 RR，Oracle 默认 RC。

```sql
SET SESSION TRANSACTION ISOLATION LEVEL READ COMMITTED;
```

## RR vs RC

RR 在事务内多次读结果一致，靠 MVCC + Next-Key Lock 防幻读。
RC 每次读取最新已提交数据，可能出现不可重复读。
````

`backend/tests/fixtures/notes_sample/k8s/controller-runtime.md`:

````markdown
# controller-runtime

## Reconcile 循环

每个 controller 监听若干资源，事件驱动调用 Reconcile(req)。
Reconcile 应当是幂等的，返回 Result{Requeue, RequeueAfter} 决定下次执行时机。
````

`backend/tests/fixtures/notes_sample/项目/订单系统复盘.md`:

````markdown
# 订单系统复盘

## 幂等性方案

我们用「业务唯一键 + 状态机」做幂等：
- 入口处用 orderNo 做唯一索引
- 状态机只允许合法跃迁（PENDING -> PAID）
- 重复请求查到已存在直接返回旧结果
````

- [ ] **Step 3: 写失败测试 `tests/unit/test_chunker.py`**

```python
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
```

- [ ] **Step 4: 运行测试验证失败**

Run: `pytest tests/unit/test_chunker.py -v`
Expected: FAIL — `core.chunker` 不存在

- [ ] **Step 5: 实现 `core/chunker.py`**

```python
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
        return f" CODE{len(code_blocks)-1} "

    masked = re.sub(r"```[\s\S]*?```", stash, body)
    pieces = _split_by_paragraph(masked, max_tokens, overlap_tokens)

    def restore(text: str) -> str:
        return re.sub(
            r" CODE(\d+) ",
            lambda m: code_blocks[int(m.group(1))],
            text,
        )

    return [restore(p) for p in pieces]
```

- [ ] **Step 6: 运行测试验证通过**

Run: `pytest tests/unit/test_chunker.py -v`
Expected: PASS（5 个用例）

- [ ] **Step 7: 提交**

```bash
git add backend/core/models.py backend/core/chunker.py backend/tests/unit/test_chunker.py backend/tests/fixtures/
git commit -m "feat(backend): markdown chunker with H2 split and code-block protection"
```

---

## Task 3: 嵌入客户端（含缓存）

**Files:**
- Create: `backend/core/embedder.py`
- Test: `backend/tests/unit/test_embedder.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_embedder.py`**

```python
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.embedder import OpenAIEmbedder


@pytest.mark.asyncio
async def test_embed_batches_and_caches(tmp_path):
    cache = tmp_path / "cache.json"
    fake_client = AsyncMock()
    fake_client.create = AsyncMock(return_value=type("R", (), {
        "data": [type("E", (), {"embedding": [0.1, 0.2, 0.3]})() for _ in range(2)]
    })())

    emb = OpenAIEmbedder(api_key="sk-x", model="m", cache_path=cache, _client=fake_client, batch_size=2)

    out1 = await emb.embed(["hello", "world"])
    assert len(out1) == 2 and out1[0] == [0.1, 0.2, 0.3]
    assert fake_client.create.await_count == 1

    # 第二次同样输入：完全命中缓存，不再调用
    out2 = await emb.embed(["hello", "world"])
    assert out2 == out1
    assert fake_client.create.await_count == 1

    # cache 落盘
    saved = json.loads(cache.read_text())
    assert len(saved) == 2


@pytest.mark.asyncio
async def test_embed_partial_cache_hit(tmp_path):
    cache = tmp_path / "cache.json"
    fake_client = AsyncMock()
    # 第一轮：embed 两个，写缓存
    fake_client.create = AsyncMock(return_value=type("R", (), {
        "data": [type("E", (), {"embedding": [1.0]})(), type("E", (), {"embedding": [2.0]})()]
    })())
    emb = OpenAIEmbedder(api_key="sk", model="m", cache_path=cache, _client=fake_client, batch_size=10)
    await emb.embed(["a", "b"])

    # 第二轮：a 命中、c 未命中 → 只对 c 发请求
    fake_client.create = AsyncMock(return_value=type("R", (), {
        "data": [type("E", (), {"embedding": [3.0]})()]
    })())
    out = await emb.embed(["a", "c"])
    assert out == [[1.0], [3.0]]
    fake_client.create.assert_awaited_once()
    assert fake_client.create.call_args.kwargs["input"] == ["c"]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/unit/test_embedder.py -v`
Expected: FAIL — `core.embedder` 不存在

- [ ] **Step 3: 实现 `core/embedder.py`**

```python
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Sequence

from openai import AsyncOpenAI


def _key(model: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(text.encode())
    return h.hexdigest()


class OpenAIEmbedder:
    def __init__(
        self,
        api_key: str,
        model: str,
        cache_path: Path,
        batch_size: int = 100,
        _client=None,
    ):
        self.model = model
        self.cache_path = cache_path
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}
        self._client = _client or AsyncOpenAI(api_key=api_key).embeddings

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        misses: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            k = _key(self.model, t)
            cached = self._cache.get(k)
            if cached is not None:
                results[i] = cached
            else:
                misses.append((i, t))

        for start in range(0, len(misses), self.batch_size):
            batch = misses[start : start + self.batch_size]
            inputs = [t for _, t in batch]
            resp = await self._client.create(model=self.model, input=inputs)
            for (orig_idx, txt), e in zip(batch, resp.data):
                vec = list(e.embedding)
                results[orig_idx] = vec
                self._cache[_key(self.model, txt)] = vec

        if misses:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")

        return [r for r in results]  # type: ignore[return-value]
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/unit/test_embedder.py -v`
Expected: PASS（2 个用例）

- [ ] **Step 5: 提交**

```bash
git add backend/core/embedder.py backend/tests/unit/test_embedder.py
git commit -m "feat(backend): OpenAI embedder with persistent disk cache"
```

---

## Task 4: 索引器（扫描 → 切块 → 嵌入 → FAISS 持久化）

**Files:**
- Create: `backend/core/indexer.py`
- Test: `backend/tests/unit/test_indexer.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_indexer.py`**

```python
from pathlib import Path
from unittest.mock import AsyncMock
import pytest

from core.indexer import build_or_update_index, IndexBundle


class FakeEmbedder:
    def __init__(self):
        self.calls = 0
    async def embed(self, texts):
        self.calls += 1
        # 1536 维 dummy 向量，前两个值用文本 hash 做区分
        out = []
        for t in texts:
            h = hash(t)
            v = [float((h >> i) & 1) for i in range(1536)]
            out.append(v)
        return out


@pytest.mark.asyncio
async def test_build_index_from_fixtures(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    data_dir = tmp_path / "data"
    embedder = FakeEmbedder()

    bundle = await build_or_update_index(
        notes_dir=notes,
        data_dir=data_dir,
        embedder=embedder,
        max_tokens=500,
        overlap_tokens=50,
    )

    assert isinstance(bundle, IndexBundle)
    assert len(bundle.chunks) >= 4    # 三个文件，至少各 1 段
    assert bundle.faiss_index.ntotal == len(bundle.chunks)
    assert (data_dir / "index.pkl").exists()
    assert (data_dir / "chunks.json").exists()
    assert (data_dir / "meta.json").exists()


@pytest.mark.asyncio
async def test_incremental_skips_unchanged(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    data_dir = tmp_path / "data"
    embedder = FakeEmbedder()

    await build_or_update_index(notes, data_dir, embedder, 500, 50)
    first_calls = embedder.calls

    # 第二次同样输入：embed 不再被调用（chunks 全部命中缓存层 / 文件未变）
    await build_or_update_index(notes, data_dir, embedder, 500, 50)
    assert embedder.calls == first_calls


@pytest.mark.asyncio
async def test_modified_file_reindexes(tmp_path):
    # 复制 fixtures 到可写目录
    src = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    notes = tmp_path / "notes"
    notes.mkdir()
    for p in src.rglob("*.md"):
        dst = notes / p.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    data_dir = tmp_path / "data"
    embedder = FakeEmbedder()
    bundle1 = await build_or_update_index(notes, data_dir, embedder, 500, 50)
    n1 = len(bundle1.chunks)

    # 修改一个文件
    target = next(notes.rglob("controller-runtime.md"))
    target.write_text(target.read_text() + "\n\n## 新章节\n\n新内容。", encoding="utf-8")

    bundle2 = await build_or_update_index(notes, data_dir, embedder, 500, 50)
    assert len(bundle2.chunks) == n1 + 1
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/unit/test_indexer.py -v`
Expected: FAIL — `core.indexer` 不存在

- [ ] **Step 3: 实现 `core/indexer.py`**

```python
import argparse
import asyncio
import hashlib
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Protocol

import faiss
import numpy as np

from core.chunker import chunk_markdown_file
from core.config import load_settings
from core.embedder import OpenAIEmbedder
from core.models import Chunk


class _EmbProto(Protocol):
    async def embed(self, texts): ...


class IndexBundle:
    def __init__(self, faiss_index, chunks: list[Chunk], embedding_model: str):
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.embedding_model = embedding_model


def _file_hash(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def _embed_text_for_chunk(c: Chunk) -> str:
    """嵌入时把面包屑拼到正文前面，提升短 chunk 的可识别性。"""
    crumb = " > ".join(c.heading_path) if c.heading_path else ""
    return f"{crumb}: {c.text}" if crumb else c.text


def _scan_notes(notes_dir: Path) -> list[Path]:
    return sorted(p for p in notes_dir.rglob("*.md") if p.is_file())


async def build_or_update_index(
    notes_dir: Path,
    data_dir: Path,
    embedder: _EmbProto,
    max_tokens: int,
    overlap_tokens: int,
) -> IndexBundle:
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "meta.json"
    chunks_path = data_dir / "chunks.json"
    index_path = data_dir / "index.pkl"

    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    files_hash: dict[str, str] = meta.get("files_hash", {})

    files = _scan_notes(notes_dir)
    current_hash = {str(p.relative_to(notes_dir)): _file_hash(p) for p in files}

    # 读旧 chunks（若有）
    old_chunks_by_file: dict[str, list[Chunk]] = {}
    if chunks_path.exists():
        try:
            data = json.loads(chunks_path.read_text(encoding="utf-8"))
            for d in data:
                c = Chunk(**d)
                old_chunks_by_file.setdefault(c.file_path, []).append(c)
        except (json.JSONDecodeError, TypeError):
            old_chunks_by_file = {}

    # 决定哪些文件需要重切
    new_chunks: list[Chunk] = []
    chunks_to_embed: list[Chunk] = []
    reuse_vectors: dict[str, list[float]] = {}

    # 加载旧向量 → reuse 池
    if index_path.exists() and meta.get("embedding_model"):
        try:
            with open(index_path, "rb") as f:
                old_payload = pickle.load(f)
            for c, vec in zip(old_payload["chunks"], old_payload["vectors"]):
                reuse_vectors[c["id"]] = vec
        except Exception:
            reuse_vectors = {}

    for p in files:
        rel = str(p.relative_to(notes_dir))
        unchanged = files_hash.get(rel) == current_hash[rel] and rel in old_chunks_by_file
        if unchanged:
            for c in old_chunks_by_file[rel]:
                new_chunks.append(c)
        else:
            for c in chunk_markdown_file(p, rel, max_tokens, overlap_tokens):
                new_chunks.append(c)
                if c.id not in reuse_vectors:
                    chunks_to_embed.append(c)

    # 嵌入未命中
    if chunks_to_embed:
        texts = [_embed_text_for_chunk(c) for c in chunks_to_embed]
        vecs = await embedder.embed(texts)
        for c, v in zip(chunks_to_embed, vecs):
            reuse_vectors[c.id] = v

    # 组 FAISS
    if not new_chunks:
        index = faiss.IndexFlatIP(1)
        bundle = IndexBundle(index, [], meta.get("embedding_model", ""))
        meta_path.write_text(json.dumps({"files_hash": current_hash, "embedding_model": "", "last_indexed_at": int(time.time())}), encoding="utf-8")
        chunks_path.write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")
        with open(index_path, "wb") as f:
            pickle.dump({"chunks": [], "vectors": []}, f)
        return bundle

    matrix = np.array([reuse_vectors[c.id] for c in new_chunks], dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    # 持久化
    with open(index_path, "wb") as f:
        pickle.dump(
            {"chunks": [asdict(c) for c in new_chunks], "vectors": matrix.tolist()},
            f,
        )
    chunks_path.write_text(json.dumps([asdict(c) for c in new_chunks], ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(
        json.dumps({
            "files_hash": current_hash,
            "embedding_model": getattr(embedder, "model", ""),
            "last_indexed_at": int(time.time()),
        }),
        encoding="utf-8",
    )
    return IndexBundle(index, new_chunks, getattr(embedder, "model", ""))


def load_index(data_dir: Path) -> IndexBundle | None:
    index_path = data_dir / "index.pkl"
    meta_path = data_dir / "meta.json"
    if not index_path.exists() or not meta_path.exists():
        return None
    try:
        with open(index_path, "rb") as f:
            payload = pickle.load(f)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        # 备份损坏文件
        backup = index_path.with_suffix(f".broken-{int(time.time())}")
        index_path.rename(backup)
        return None
    chunks = [Chunk(**c) for c in payload["chunks"]]
    if not chunks:
        return IndexBundle(faiss.IndexFlatIP(1), [], meta.get("embedding_model", ""))
    matrix = np.array(payload["vectors"], dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return IndexBundle(index, chunks, meta.get("embedding_model", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="忽略旧索引，全量重建")
    args = parser.parse_args()

    settings = load_settings()
    settings.validate_for_runtime()
    data_dir = Path(__file__).resolve().parent.parent / "data"

    if args.rebuild:
        for f in ["index.pkl", "chunks.json", "meta.json"]:
            p = data_dir / f
            if p.exists():
                p.unlink()

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        cache_path=data_dir / "embedding_cache.json",
    )

    bundle = asyncio.run(build_or_update_index(
        notes_dir=settings.notes_dir,
        data_dir=data_dir,
        embedder=embedder,
        max_tokens=settings.max_chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    ))
    print(f"Indexed {len(bundle.chunks)} chunks from {settings.notes_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/unit/test_indexer.py -v`
Expected: PASS（3 个用例）

- [ ] **Step 5: 提交**

```bash
git add backend/core/indexer.py backend/tests/unit/test_indexer.py
git commit -m "feat(backend): incremental indexer with FAISS persistence"
```

---

## Task 5: 检索器（向量 + BM25 混合）

**Files:**
- Create: `backend/core/retriever.py`
- Test: `backend/tests/unit/test_retriever.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_retriever.py`**

```python
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from core.indexer import build_or_update_index
from core.models import AskMode
from core.retriever import Retriever


class FakeEmbedder:
    model = "m"
    async def embed(self, texts):
        return [_dummy_vec(t) for t in texts]


def _dummy_vec(t: str) -> list[float]:
    rng = np.random.default_rng(abs(hash(t)) % (2**32))
    return rng.normal(size=1536).astype("float32").tolist()


@pytest.mark.asyncio
async def test_retriever_hits_known_query(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    embedder = FakeEmbedder()
    bundle = await build_or_update_index(notes, tmp_path / "data", embedder, 500, 50)

    r = Retriever(
        bundle=bundle, embedder=embedder,
        top_k=8, rerank_top_k=3, threshold=0.0,    # 阈值置 0：dummy 向量未必稳定，HIT 由 BM25 占主导
        vector_weight=0.0, bm25_weight=1.0,
    )
    res = await r.search("RR vs RC")
    assert res.mode == AskMode.HIT
    paths = [c.chunk.file_path for c in res.chunks]
    assert any("事务隔离" in p for p in paths)
    assert len(res.chunks) <= 3


@pytest.mark.asyncio
async def test_retriever_fallback_when_below_threshold(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    embedder = FakeEmbedder()
    bundle = await build_or_update_index(notes, tmp_path / "data", embedder, 500, 50)

    r = Retriever(
        bundle=bundle, embedder=embedder,
        top_k=8, rerank_top_k=3, threshold=0.99,   # 极高阈值
        vector_weight=0.7, bm25_weight=0.3,
    )
    res = await r.search("Rust 生命周期")
    assert res.mode == AskMode.FALLBACK
    assert len(res.chunks) <= 3                    # 仍返回弱相关片段
    assert res.top1_score < 0.99


@pytest.mark.asyncio
async def test_retriever_empty_when_no_chunks(tmp_path):
    embedder = FakeEmbedder()
    empty_notes = tmp_path / "empty"
    empty_notes.mkdir()
    bundle = await build_or_update_index(empty_notes, tmp_path / "data", embedder, 500, 50)

    r = Retriever(bundle=bundle, embedder=embedder, top_k=8, rerank_top_k=3, threshold=0.5, vector_weight=0.7, bm25_weight=0.3)
    res = await r.search("anything")
    assert res.mode == AskMode.EMPTY
    assert res.chunks == []
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/unit/test_retriever.py -v`
Expected: FAIL — `core.retriever` 不存在

- [ ] **Step 3: 实现 `core/retriever.py`**

```python
from dataclasses import dataclass
from typing import Protocol

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from core.models import AskMode, Chunk, ScoredChunk


class _EmbProto(Protocol):
    async def embed(self, texts): ...


@dataclass
class RetrievalResult:
    mode: AskMode
    chunks: list[ScoredChunk]
    top1_score: float


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class Retriever:
    def __init__(
        self,
        bundle,                # IndexBundle
        embedder: _EmbProto,
        top_k: int,
        rerank_top_k: int,
        threshold: float,
        vector_weight: float,
        bm25_weight: float,
    ):
        self.bundle = bundle
        self.embedder = embedder
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.threshold = threshold
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        if bundle.chunks:
            corpus = [_tokenize(c.text) for c in bundle.chunks]
            self.bm25 = BM25Okapi(corpus)
        else:
            self.bm25 = None

    async def search(self, query: str) -> RetrievalResult:
        if not self.bundle.chunks or self.bm25 is None:
            return RetrievalResult(AskMode.EMPTY, [], 0.0)

        # 向量召回
        try:
            qvec = (await self.embedder.embed([query]))[0]
        except Exception:
            qvec = None

        n = len(self.bundle.chunks)
        if qvec is not None:
            v = np.array([qvec], dtype="float32")
            # 归一化与索引内一致
            faiss_normalize(v)
            k = min(self.top_k, n)
            sims, ids = self.bundle.faiss_index.search(v, k)
            vec_pairs = list(zip(ids[0].tolist(), sims[0].tolist()))
        else:
            vec_pairs = [(i, 0.0) for i in range(min(self.top_k, n))]

        # BM25 全量打分
        bm25_scores = self.bm25.get_scores(_tokenize(query))

        # 候选集 = 向量 top_k 与 bm25 top_k 的并集
        bm25_top_idx = np.argsort(bm25_scores)[::-1][: self.top_k].tolist()
        candidate_ids = list({i for i, _ in vec_pairs} | set(bm25_top_idx))

        vec_lookup = {i: s for i, s in vec_pairs}
        cand_vec = [vec_lookup.get(i, 0.0) for i in candidate_ids]
        cand_bm = [float(bm25_scores[i]) for i in candidate_ids]
        nv = _normalize(cand_vec)
        nb = _normalize(cand_bm)
        mixed = [
            self.vector_weight * v + self.bm25_weight * b
            for v, b in zip(nv, nb)
        ]

        order = sorted(range(len(candidate_ids)), key=lambda k: mixed[k], reverse=True)
        top = order[: self.rerank_top_k]
        scored = [
            ScoredChunk(
                chunk=self.bundle.chunks[candidate_ids[k]],
                score=mixed[k],
                vector_score=cand_vec[k],
                bm25_score=cand_bm[k],
            )
            for k in top
        ]
        top1 = scored[0].score if scored else 0.0
        mode = AskMode.HIT if top1 >= self.threshold else AskMode.FALLBACK
        return RetrievalResult(mode=mode, chunks=scored, top1_score=top1)


def faiss_normalize(v: np.ndarray) -> None:
    import faiss
    faiss.normalize_L2(v)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/unit/test_retriever.py -v`
Expected: PASS（3 个用例）

- [ ] **Step 5: 提交**

```bash
git add backend/core/retriever.py backend/tests/unit/test_retriever.py
git commit -m "feat(backend): hybrid retriever (vector + BM25) with threshold mode"
```

---

## Task 6: LLM provider 抽象 + DeepSeek 实现 + Prompt 模板

**Files:**
- Create: `backend/core/llm/__init__.py`
- Create: `backend/core/llm/base.py`
- Create: `backend/core/llm/deepseek.py`
- Create: `backend/core/prompts.py`
- Test: `backend/tests/unit/test_llm_providers.py`

- [ ] **Step 1: 写 `core/llm/__init__.py` 与 `core/llm/base.py`**

```python
# core/llm/__init__.py
from core.llm.base import LLMProvider, build_provider
__all__ = ["LLMProvider", "build_provider"]
```

```python
# core/llm/base.py
from typing import AsyncIterator, Protocol


class LLMProvider(Protocol):
    async def stream(self, prompt: str) -> AsyncIterator[str]: ...


def build_provider(name: str, *, api_key: str, model: str) -> "LLMProvider":
    if name == "deepseek":
        from core.llm.deepseek import DeepSeekProvider
        return DeepSeekProvider(api_key=api_key, model=model)
    if name == "openai":
        from core.llm.openai import OpenAIProvider
        return OpenAIProvider(api_key=api_key, model=model)
    if name == "claude":
        from core.llm.claude import ClaudeProvider
        return ClaudeProvider(api_key=api_key, model=model)
    raise ValueError(f"Unknown LLM provider: {name}")
```

- [ ] **Step 2: 写 `core/prompts.py`**

```python
from core.models import ScoredChunk


def hit_prompt(query: str, chunks: list[ScoredChunk]) -> str:
    refs = "\n\n".join(
        f"[{i+1}] {' > '.join(c.chunk.heading_path)}\n{c.chunk.text}"
        for i, c in enumerate(chunks)
    )
    return f"""你是候选人本人的实时面试助手。候选人的笔记已经直接覆盖了这个问题，原文已显示在屏幕上。你的任务是基于这些原文做"延展"，让候选人答得更深。

【输出结构（严格遵守，每节 1-3 句）】
▎可主动延展：原文外可以多说的相关知识点
▎容易追问：面试官可能下一步追问的方向 + 简短应答思路
▎踩坑提醒：原文里没强调但容易翻车的点

【约束】
- 不要复述原文内容，只补充
- 第一人称口吻
- 总长度 ≤ 150 字

【面试官问题】
{query}

【已命中的笔记原文】
{refs}
"""


def fallback_prompt(query: str, resume_text: str, weak_chunks: list[ScoredChunk]) -> str:
    refs = "\n\n".join(
        f"[{i+1}] {' > '.join(c.chunk.heading_path)}\n{c.chunk.text}"
        for i, c in enumerate(weak_chunks)
    ) or "(无)"
    return f"""你是候选人本人，正在面试中。基于"我的简历"回答面试官的问题。

【强约束】
- 用第一人称："我"、"我之前"、"我一般会"
- 自然衔接到简历中的真实项目经验，不要生造未在简历中出现的项目
- 如果简历经验不直接相关，老实承认："这块我接触不多，但..."然后引到熟悉领域
- 控制在 200 字以内，口语化，面试官能听懂
- 不要说"基于简历"、"根据资料"这类暴露的措辞

【输出结构】
▎核心回答：1-2 句直接答
▎我的经验：从简历项目里挑最相关的 1 个串过来
▎可延展：1 句把话题引到我熟悉的方向

【面试官问题】
{query}

【我的简历】
{resume_text}

【笔记中的弱相关片段（仅供参考，可不用）】
{refs}
"""
```

- [ ] **Step 3: 写失败测试 `tests/unit/test_llm_providers.py`**

```python
import pytest
import respx
import httpx
from core.llm.deepseek import DeepSeekProvider


@pytest.mark.asyncio
@respx.mock
async def test_deepseek_streams_tokens():
    body = (
        b'data: {"choices":[{"delta":{"content":"\xe4\xb8\x80"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"\xe4\xba\x8c"}}]}\n\n'
        b'data: [DONE]\n\n'
    )
    respx.post("https://api.deepseek.com/chat/completions").mock(
        return_value=httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )
    provider = DeepSeekProvider(api_key="dk", model="deepseek-chat")
    out = []
    async for tok in provider.stream("hi"):
        out.append(tok)
    assert "".join(out) == "一二"


@pytest.mark.asyncio
@respx.mock
async def test_deepseek_handles_http_error():
    respx.post("https://api.deepseek.com/chat/completions").mock(
        return_value=httpx.Response(500, text="boom")
    )
    provider = DeepSeekProvider(api_key="dk", model="deepseek-chat")
    with pytest.raises(RuntimeError, match="DeepSeek"):
        async for _ in provider.stream("hi"):
            pass
```

- [ ] **Step 4: 运行测试验证失败**

Run: `pytest tests/unit/test_llm_providers.py -v`
Expected: FAIL — `core.llm.deepseek` 不存在

- [ ] **Step 5: 实现 `core/llm/deepseek.py`**

```python
import json
from typing import AsyncIterator

import httpx


class DeepSeekProvider:
    BASE_URL = "https://api.deepseek.com/chat/completions"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self.BASE_URL, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    text = await resp.aread()
                    raise RuntimeError(f"DeepSeek API error {resp.status_code}: {text!r}")
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (
                        obj.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if delta:
                        yield delta
```

- [ ] **Step 6: 运行测试验证通过**

Run: `pytest tests/unit/test_llm_providers.py -v`
Expected: PASS（2 个用例）

- [ ] **Step 7: 实现 `core/llm/openai.py` 与 `core/llm/claude.py`（占位但可用）**

```python
# core/llm/openai.py
from typing import AsyncIterator
from openai import AsyncOpenAI


class OpenAIProvider:
    def __init__(self, api_key: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
```

```python
# core/llm/claude.py
from typing import AsyncIterator
from anthropic import AsyncAnthropic


class ClaudeProvider:
    def __init__(self, api_key: str, model: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            async for text in s.text_stream:
                yield text
```

- [ ] **Step 8: 提交**

```bash
git add backend/core/llm/ backend/core/prompts.py backend/tests/unit/test_llm_providers.py
git commit -m "feat(backend): pluggable LLM providers (DeepSeek/OpenAI/Claude) + prompts"
```

---

## Task 7: ASR（Whisper API）+ 简历加载

**Files:**
- Create: `backend/core/asr.py`
- Create: `backend/core/resume.py`
- Test: `backend/tests/unit/test_asr.py`

- [ ] **Step 1: 写失败测试 `tests/unit/test_asr.py`**

```python
import io
import pytest
import respx
import httpx
from core.asr import WhisperASR


@pytest.mark.asyncio
@respx.mock
async def test_whisper_returns_text():
    respx.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "你好世界"})
    )
    asr = WhisperASR(api_key="sk", model="whisper-1")
    text = await asr.transcribe(io.BytesIO(b"\x00\x01"), filename="a.webm", mime="audio/webm")
    assert text == "你好世界"


@pytest.mark.asyncio
@respx.mock
async def test_whisper_raises_on_error():
    respx.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(503, text="busy")
    )
    asr = WhisperASR(api_key="sk", model="whisper-1")
    with pytest.raises(RuntimeError, match="Whisper"):
        await asr.transcribe(io.BytesIO(b"\x00"), filename="a.webm", mime="audio/webm")
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/unit/test_asr.py -v`
Expected: FAIL — `core.asr` 不存在

- [ ] **Step 3: 实现 `core/asr.py`**

```python
from typing import BinaryIO

import httpx


class WhisperASR:
    URL = "https://api.openai.com/v1/audio/transcriptions"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def transcribe(self, audio: BinaryIO, filename: str, mime: str, language: str = "zh") -> str:
        files = {"file": (filename, audio, mime)}
        data = {"model": self.model, "language": language}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.URL, headers=headers, files=files, data=data)
            if resp.status_code >= 400:
                raise RuntimeError(f"Whisper API error {resp.status_code}: {resp.text}")
            return resp.json()["text"]
```

- [ ] **Step 4: 实现 `core/resume.py`**

```python
import logging
from pathlib import Path

import tiktoken

log = logging.getLogger(__name__)
_enc = tiktoken.get_encoding("cl100k_base")
_MAX_TOKENS = 8000


def load_resume(path: Path) -> str:
    if not path.exists():
        log.warning("Resume file not found: %s (fallback mode will indicate empty resume)", path)
        return ""
    text = path.read_text(encoding="utf-8")
    tok = _enc.encode(text)
    if len(tok) > _MAX_TOKENS:
        log.warning("Resume too long (%d tokens), truncating to %d", len(tok), _MAX_TOKENS)
        text = _enc.decode(tok[:_MAX_TOKENS])
    return text
```

- [ ] **Step 5: 运行测试验证通过**

Run: `pytest tests/unit/test_asr.py -v`
Expected: PASS（2 个用例）

- [ ] **Step 6: 提交**

```bash
git add backend/core/asr.py backend/core/resume.py backend/tests/unit/test_asr.py
git commit -m "feat(backend): Whisper ASR client + resume loader"
```

---

## Task 8: `/api/asr` 与 `/api/ask` 路由（SSE）

**Files:**
- Modify: `backend/main.py`
- Create: `backend/routes/__init__.py`
- Create: `backend/routes/asr.py`
- Create: `backend/routes/ask.py`
- Test: `backend/tests/integration/test_ask_flow.py`

- [ ] **Step 1: 写 `routes/__init__.py`（空）**

```python
```

- [ ] **Step 2: 写 `routes/asr.py`**

```python
import asyncio
import logging
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

log = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/asr")
async def asr(request: Request, file: UploadFile = File(...)):
    settings = request.app.state.settings
    asr_client = request.app.state.asr
    try:
        audio = await file.read()
        from io import BytesIO
        text = await asyncio.wait_for(
            asr_client.transcribe(
                BytesIO(audio), filename=file.filename or "audio.webm",
                mime=file.content_type or "audio/webm",
            ),
            timeout=settings.asr_timeout_s,
        )
        return {"text": text}
    except asyncio.TimeoutError:
        log.warning("ASR timeout")
        raise HTTPException(status_code=504, detail="ASR timeout")
    except Exception as e:
        log.exception("ASR failed")
        raise HTTPException(status_code=502, detail=f"ASR failed: {e}")
```

- [ ] **Step 3: 写 `routes/ask.py`**

```python
import asyncio
import json
import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from core.models import AskMode
from core.prompts import fallback_prompt, hit_prompt

log = logging.getLogger(__name__)
router = APIRouter()


def _sse(event: str, data) -> bytes:
    payload = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


@router.get("/api/ask")
async def ask(request: Request, query: str):
    settings = request.app.state.settings
    retriever = request.app.state.retriever
    llm = request.app.state.llm
    resume_text: str = request.app.state.resume_text

    async def gen():
        t0 = time.monotonic()
        try:
            result = await retriever.search(query)
        except Exception as e:
            log.exception("retrieve failed")
            yield _sse("error", {"stage": "retrieve", "message": str(e), "recoverable": False})
            return

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        yield _sse("mode", {"mode": result.mode.value, "top1_score": result.top1_score, "elapsed_ms": elapsed_ms})
        yield _sse("chunks", [
            {
                "id": sc.chunk.id,
                "file_path": sc.chunk.file_path,
                "heading_path": sc.chunk.heading_path,
                "text": sc.chunk.text,
                "score": sc.score,
                "line_start": sc.chunk.line_start,
                "line_end": sc.chunk.line_end,
            }
            for sc in result.chunks
        ])

        if result.mode == AskMode.EMPTY:
            if not resume_text:
                yield _sse("done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)})
                return
            prompt = fallback_prompt(query, resume_text, [])
        elif result.mode == AskMode.FALLBACK:
            prompt = fallback_prompt(query, resume_text, result.chunks)
        else:
            prompt = hit_prompt(query, result.chunks)

        try:
            first_token_deadline = settings.llm_first_token_timeout_s
            total_deadline = settings.llm_total_timeout_s
            stream = llm.stream(prompt)
            first = True
            async def _next():
                return await stream.__anext__()

            stream_task_started = time.monotonic()
            while True:
                if await request.is_disconnected():
                    log.info("client disconnected, abort llm stream")
                    return
                timeout = first_token_deadline if first else max(0.1, total_deadline - (time.monotonic() - stream_task_started))
                try:
                    tok = await asyncio.wait_for(_next(), timeout=timeout)
                except asyncio.TimeoutError:
                    yield _sse("error", {"stage": "llm", "message": "timeout", "recoverable": False})
                    return
                except StopAsyncIteration:
                    break
                first = False
                yield _sse("token", tok)
        except Exception as e:
            log.exception("llm stream failed")
            yield _sse("error", {"stage": "llm", "message": str(e), "recoverable": False})
            return

        yield _sse("done", {"elapsed_ms": int((time.monotonic() - t0) * 1000)})

    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 4: 修改 `main.py` 装配应用**

```python
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.asr import WhisperASR
from core.config import load_settings
from core.embedder import OpenAIEmbedder
from core.indexer import build_or_update_index, load_index
from core.llm import build_provider
from core.resume import load_resume
from core.retriever import Retriever
from routes.asr import router as asr_router
from routes.ask import router as ask_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    settings.validate_for_runtime()
    data_dir = Path(__file__).resolve().parent / "data"

    embedder = OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        cache_path=data_dir / "embedding_cache.json",
    )
    bundle = load_index(data_dir)
    if bundle is None or not bundle.chunks:
        log.info("No usable index, building from %s", settings.notes_dir)
        bundle = await build_or_update_index(
            settings.notes_dir, data_dir, embedder,
            settings.max_chunk_tokens, settings.chunk_overlap_tokens,
        )

    retriever = Retriever(
        bundle=bundle, embedder=embedder,
        top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
        threshold=settings.retrieval_threshold,
        vector_weight=settings.vector_weight,
        bm25_weight=settings.bm25_weight,
    )
    llm = build_provider(settings.llm_provider, api_key=getattr(settings, f"{settings.llm_provider}_api_key" if settings.llm_provider != "claude" else "anthropic_api_key"), model=settings.llm_model)
    asr_client = WhisperASR(api_key=settings.openai_api_key, model=settings.asr_model)
    resume_text = load_resume(settings.resume_path)

    app.state.settings = settings
    app.state.retriever = retriever
    app.state.llm = llm
    app.state.asr = asr_client
    app.state.resume_text = resume_text
    log.info("Ready: %d chunks, provider=%s", len(bundle.chunks), settings.llm_provider)
    yield


app = FastAPI(title="Interview RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(asr_router)
app.include_router(ask_router)


@app.get("/api/health")
def health():
    return {"ok": True}
```

- [ ] **Step 5: 写集成测试 `tests/integration/test_ask_flow.py`**

```python
import asyncio
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.indexer import build_or_update_index
from core.models import AskMode
from core.retriever import Retriever


class FakeEmbedder:
    model = "m"
    async def embed(self, texts):
        return [_dv(t) for t in texts]


def _dv(t: str) -> list[float]:
    rng = np.random.default_rng(abs(hash(t)) % (2**32))
    return rng.normal(size=1536).astype("float32").tolist()


class FakeLLM:
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        for ch in "假装的回答":
            yield ch


def _build_app(notes_dir: Path, data_dir: Path, threshold: float, resume_text: str):
    from main import app
    embedder = FakeEmbedder()
    async def _setup():
        bundle = await build_or_update_index(notes_dir, data_dir, embedder, 500, 50)
        retriever = Retriever(bundle, embedder, 8, 3, threshold, 0.0, 1.0)
        app.state.settings = type("S", (), {
            "asr_timeout_s": 5,
            "llm_first_token_timeout_s": 5,
            "llm_total_timeout_s": 30,
            "retrieval_threshold": threshold,
        })()
        app.state.retriever = retriever
        app.state.llm = FakeLLM()
        app.state.resume_text = resume_text
        app.state.asr = None
    asyncio.get_event_loop().run_until_complete(_setup())
    return app


def _parse_sse(raw: str) -> list[tuple[str, str]]:
    events = []
    for block in raw.strip().split("\n\n"):
        ev = data = None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        events.append((ev, data))
    return events


def test_ask_hit_flow(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    app = _build_app(notes, tmp_path / "d", threshold=0.0, resume_text="我是测试简历")
    with TestClient(app) as c:
        with c.stream("GET", "/api/ask", params={"query": "RR vs RC"}) as r:
            text = "".join(r.iter_text())
    events = _parse_sse(text)
    types = [e for e, _ in events]
    assert "mode" in types
    assert "chunks" in types
    assert types.count("token") >= 1
    assert types[-1] == "done"

    import json
    mode_ev = next(d for e, d in events if e == "mode")
    assert json.loads(mode_ev)["mode"] == "hit"


def test_ask_fallback_flow(tmp_path):
    notes = Path(__file__).parent.parent / "fixtures" / "notes_sample"
    app = _build_app(notes, tmp_path / "d", threshold=99.0, resume_text="我做过订单系统")
    with TestClient(app) as c:
        with c.stream("GET", "/api/ask", params={"query": "Rust 生命周期"}) as r:
            text = "".join(r.iter_text())
    events = _parse_sse(text)
    import json
    mode_ev = next(d for e, d in events if e == "mode")
    assert json.loads(mode_ev)["mode"] == "fallback"
    assert any(e == "token" for e, _ in events)
```

- [ ] **Step 6: 运行集成测试验证通过**

Run: `pytest tests/integration -v`
Expected: PASS（2 个用例）

- [ ] **Step 7: 端到端冒烟（手动）**

```bash
cd backend
OPENAI_API_KEY=sk-xxx DEEPSEEK_API_KEY=dk-xxx \
  NOTES_DIR=./tests/fixtures/notes_sample \
  RESUME_PATH=./tests/fixtures/_no_resume.md \
  uvicorn main:app --port 8000 &
sleep 2
curl -N "http://localhost:8000/api/ask?query=RR%20vs%20RC"
kill %1
```
Expected: 看到 `event: mode` / `event: chunks` / `event: token` / `event: done` 的 SSE 流

- [ ] **Step 8: 提交**

```bash
git add backend/main.py backend/routes/ backend/tests/integration/
git commit -m "feat(backend): SSE /api/ask + multipart /api/asr with TestClient integration tests"
```

---

## Task 9: 前端项目骨架（Vite + React + Tailwind）

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/index.css`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/types.ts`

- [ ] **Step 1: 初始化前端工程**

```bash
cd /Users/mi/Desktop/RAG/frontend
npm create vite@latest . -- --template react-ts
# 提示 directory not empty 时选 ignore
```

- [ ] **Step 2: 安装依赖**

```bash
npm install
npm install react-markdown rehype-highlight highlight.js
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

- [ ] **Step 3: 写 `tailwind.config.js`**

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

- [ ] **Step 4: 写 `src/index.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@import "highlight.js/styles/github.css";

html, body, #root { height: 100%; }
body { @apply bg-zinc-50 text-zinc-900; }
```

- [ ] **Step 5: 改 `vite.config.ts`（前端 5173 → 后端 8000 代理）**

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
```

- [ ] **Step 6: 写 `src/types.ts`**

```ts
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
```

- [ ] **Step 7: 写最小 `App.tsx` 与 `main.tsx`（占位，后续 Task 填）**

```tsx
// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
```

```tsx
// src/App.tsx
export default function App() {
  return (
    <div className="h-full flex items-center justify-center">
      <p className="text-xl">Interview RAG · scaffold ready</p>
    </div>
  );
}
```

- [ ] **Step 8: 验证 dev server 跑得起来**

```bash
cd /Users/mi/Desktop/RAG/frontend
npm run dev &
sleep 3
curl -s http://localhost:5173 | head -n 5
kill %1
```
Expected: 返回 HTML 含 `<div id="root">`

- [ ] **Step 9: 提交**

```bash
git add frontend/
git commit -m "feat(frontend): vite + react + tailwind scaffold"
```

---

## Task 10: SSE 订阅 hook + 双栏组件

**Files:**
- Create: `frontend/src/hooks/useAsk.ts`
- Create: `frontend/src/components/RetrievalPane.tsx`
- Create: `frontend/src/components/SummaryPane.tsx`
- Create: `frontend/src/components/Toast.tsx`
- Create: `frontend/src/components/QueryInput.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: 写 `hooks/useAsk.ts`**

```ts
import { useCallback, useRef, useState } from "react";
import type { AskMode, ChunkPayload } from "../types";

export interface AskState {
  mode: AskMode | null;
  top1Score: number | null;
  chunks: ChunkPayload[];
  summary: string;
  status: "idle" | "loading" | "streaming" | "done" | "error";
  errorMessage?: string;
  elapsedMs?: number;
}

const initial: AskState = { mode: null, top1Score: null, chunks: [], summary: "", status: "idle" };

export function useAsk() {
  const [state, setState] = useState<AskState>(initial);
  const sourceRef = useRef<EventSource | null>(null);

  const abort = useCallback(() => {
    sourceRef.current?.close();
    sourceRef.current = null;
  }, []);

  const ask = useCallback((query: string) => {
    abort();
    setState({ ...initial, status: "loading" });
    const url = `/api/ask?query=${encodeURIComponent(query)}`;
    const es = new EventSource(url);
    sourceRef.current = es;

    es.addEventListener("mode", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, mode: d.mode, top1Score: d.top1_score, status: "streaming" }));
    });
    es.addEventListener("chunks", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, chunks: d }));
    });
    es.addEventListener("token", (e) => {
      const d = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, summary: s.summary + d }));
    });
    es.addEventListener("error", (e) => {
      const raw = (e as MessageEvent).data;
      let msg = "连接错误";
      if (raw) {
        try { msg = JSON.parse(raw).message ?? msg; } catch {}
      }
      setState((s) => ({ ...s, status: "error", errorMessage: msg }));
      es.close();
    });
    es.addEventListener("done", (e) => {
      const d = JSON.parse((e as MessageEvent).data || "{}");
      setState((s) => ({ ...s, status: "done", elapsedMs: d.elapsed_ms }));
      es.close();
    });
    es.onerror = () => {
      setState((s) => (s.status === "done" ? s : { ...s, status: "error", errorMessage: "SSE 中断" }));
      es.close();
    };
  }, [abort]);

  const reset = useCallback(() => {
    abort();
    setState(initial);
  }, [abort]);

  return { state, ask, abort, reset };
}
```

- [ ] **Step 2: 写 `components/RetrievalPane.tsx`**

```tsx
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import type { AskMode, ChunkPayload } from "../types";

interface Props {
  mode: AskMode | null;
  chunks: ChunkPayload[];
}

export function RetrievalPane({ mode, chunks }: Props) {
  if (!mode || chunks.length === 0) {
    return <div className="text-zinc-400 text-sm">等待查询…</div>;
  }
  const dim = mode === "fallback";
  return (
    <div className={dim ? "opacity-60" : ""}>
      {dim && (
        <div className="mb-2 text-xs text-amber-700 bg-amber-50 px-2 py-1 rounded">
          ⚠ 弱相关，仅供参考
        </div>
      )}
      {chunks.map((c) => (
        <div key={c.id} className="mb-4 bg-white rounded-lg shadow-sm p-3">
          <div className="text-xs text-zinc-500 mb-1 flex justify-between">
            <span>{c.heading_path.join(" > ")} · {c.file_path}</span>
            <span>score {c.score.toFixed(2)}</span>
          </div>
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{c.text}</ReactMarkdown>
          </div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: 写 `components/SummaryPane.tsx`**

```tsx
import type { AskMode } from "../types";

interface Props {
  mode: AskMode | null;
  summary: string;
  status: "idle" | "loading" | "streaming" | "done" | "error";
}

export function SummaryPane({ mode, summary, status }: Props) {
  const tag =
    mode === "hit" ? { text: "✓ 笔记命中", cls: "bg-emerald-100 text-emerald-800" } :
    mode === "fallback" ? { text: "🟡 笔记未直接命中，基于简历回答", cls: "bg-amber-100 text-amber-800" } :
    null;
  return (
    <div>
      {tag && (
        <div className={`text-xs px-2 py-1 rounded inline-block mb-2 ${tag.cls}`}>{tag.text}</div>
      )}
      <pre className="whitespace-pre-wrap font-sans text-sm leading-6">{summary}</pre>
      {status === "loading" && <p className="text-zinc-400 text-sm">检索中…</p>}
      {status === "streaming" && summary === "" && <p className="text-zinc-400 text-sm">生成中…</p>}
    </div>
  );
}
```

- [ ] **Step 4: 写 `components/Toast.tsx`**

```tsx
import { useEffect } from "react";

interface Props {
  message: string | null;
  onClose: () => void;
}

export function Toast({ message, onClose }: Props) {
  useEffect(() => {
    if (!message) return;
    const t = setTimeout(onClose, 3000);
    return () => clearTimeout(t);
  }, [message, onClose]);
  if (!message) return null;
  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-zinc-900 text-white text-sm px-4 py-2 rounded shadow-lg">
      {message}
    </div>
  );
}
```

- [ ] **Step 5: 写 `components/QueryInput.tsx`**

```tsx
import { useState } from "react";

interface Props {
  onSubmit: (q: string) => void;
  initialValue?: string;
}

export function QueryInput({ onSubmit, initialValue }: Props) {
  const [v, setV] = useState(initialValue ?? "");
  return (
    <form
      className="flex gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        const q = v.trim();
        if (q) onSubmit(q);
      }}
    >
      <input
        className="flex-1 px-3 py-2 rounded border border-zinc-300 focus:border-zinc-500 outline-none"
        placeholder="输入问题或松开热键…"
        value={v}
        onChange={(e) => setV(e.target.value)}
        autoFocus
      />
      <button className="px-4 py-2 rounded bg-zinc-900 text-white hover:bg-zinc-700" type="submit">
        提交
      </button>
    </form>
  );
}
```

- [ ] **Step 6: 改 `App.tsx`（不含 push-to-talk，下一 Task 加）**

```tsx
import { useEffect, useState } from "react";
import { QueryInput } from "./components/QueryInput";
import { RetrievalPane } from "./components/RetrievalPane";
import { SummaryPane } from "./components/SummaryPane";
import { Toast } from "./components/Toast";
import { useAsk } from "./hooks/useAsk";

export default function App() {
  const { state, ask, reset } = useAsk();
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    if (state.status === "error" && state.errorMessage) setToast(state.errorMessage);
  }, [state.status, state.errorMessage]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        reset();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [reset]);

  return (
    <div className="h-full flex flex-col p-4 gap-4 max-w-7xl mx-auto">
      <header className="flex items-center gap-3">
        <h1 className="text-lg font-semibold">面试实时辅助</h1>
        <span className="text-xs text-zinc-500">⌘K 清空</span>
      </header>
      <QueryInput onSubmit={ask} />
      <main className="grid grid-cols-2 gap-4 flex-1 min-h-0">
        <section className="overflow-auto bg-zinc-100 rounded-lg p-3">
          <h2 className="text-xs uppercase text-zinc-500 mb-2">原文片段</h2>
          <RetrievalPane mode={state.mode} chunks={state.chunks} />
        </section>
        <section className="overflow-auto bg-zinc-100 rounded-lg p-3">
          <h2 className="text-xs uppercase text-zinc-500 mb-2">延展 / 简历兜底</h2>
          <SummaryPane mode={state.mode} summary={state.summary} status={state.status} />
        </section>
      </main>
      <Toast message={toast} onClose={() => setToast(null)} />
    </div>
  );
}
```

- [ ] **Step 7: 手动验证（需要后端运行 + 真实 API key）**

```bash
# 终端 A
cd backend && uvicorn main:app --port 8000 --reload
# 终端 B
cd frontend && npm run dev
```
浏览器打开 http://localhost:5173，文字提交问题，观察双栏渲染。

- [ ] **Step 8: 提交**

```bash
git add frontend/src
git commit -m "feat(frontend): SSE hook + dual-pane UI (no voice yet)"
```

---

## Task 11: Push-to-Talk（MediaRecorder + 热键）

**Files:**
- Create: `frontend/src/hooks/usePushToTalk.ts`
- Create: `frontend/src/components/PushToTalkButton.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: 写 `hooks/usePushToTalk.ts`**

```ts
import { useCallback, useEffect, useRef, useState } from "react";

interface Options {
  onTranscript: (text: string) => void;
  onError: (msg: string) => void;
  hotkey?: string;     // 例如 "Space"
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
      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const fd = new FormData();
        fd.append("file", blob, "voice.webm");
        try {
          const resp = await fetch("/api/asr", { method: "POST", body: fd });
          if (!resp.ok) throw new Error(`ASR ${resp.status}`);
          const j = await resp.json();
          onTranscript(j.text || "");
        } catch (e: any) {
          onError(`语音识别失败：${e.message ?? e}`);
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
    } catch (e: any) {
      onError(`无法访问麦克风：${e.message ?? e}`);
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
      if (e.code === hotkey && !e.repeat) { e.preventDefault(); start(); }
    };
    const onUp = (e: KeyboardEvent) => {
      if (e.code === hotkey) { e.preventDefault(); stop(); }
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
```

- [ ] **Step 2: 写 `components/PushToTalkButton.tsx`**

```tsx
interface Props {
  recording: boolean;
  onMouseDown: () => void;
  onMouseUp: () => void;
}

export function PushToTalkButton({ recording, onMouseDown, onMouseUp }: Props) {
  return (
    <button
      className={`px-4 py-2 rounded text-white select-none ${recording ? "bg-red-600" : "bg-zinc-900 hover:bg-zinc-700"}`}
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
```

- [ ] **Step 3: 改 `App.tsx` 接入 push-to-talk + 自动提交**

```tsx
import { useEffect, useRef, useState } from "react";
import { PushToTalkButton } from "./components/PushToTalkButton";
import { QueryInput } from "./components/QueryInput";
import { RetrievalPane } from "./components/RetrievalPane";
import { SummaryPane } from "./components/SummaryPane";
import { Toast } from "./components/Toast";
import { useAsk } from "./hooks/useAsk";
import { usePushToTalk } from "./hooks/usePushToTalk";

export default function App() {
  const { state, ask, reset } = useAsk();
  const [toast, setToast] = useState<string | null>(null);
  const [draft, setDraft] = useState<string>("");
  const lastTranscriptRef = useRef<string>("");

  const { recording, start, stop } = usePushToTalk({
    onTranscript: (text) => {
      lastTranscriptRef.current = text;
      setDraft(text);
      if (text.trim()) ask(text.trim());
    },
    onError: (msg) => setToast(msg),
  });

  useEffect(() => {
    if (state.status === "error" && state.errorMessage) setToast(state.errorMessage);
  }, [state.status, state.errorMessage]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        reset();
        setDraft("");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [reset]);

  return (
    <div className="h-full flex flex-col p-4 gap-4 max-w-7xl mx-auto">
      <header className="flex items-center gap-3">
        <h1 className="text-lg font-semibold">面试实时辅助</h1>
        <span className="text-xs text-zinc-500">⌘K 清空 · 按住 Space 说话</span>
      </header>
      <div className="flex gap-2 items-center">
        <PushToTalkButton recording={recording} onMouseDown={start} onMouseUp={stop} />
        <div className="flex-1">
          <QueryInput key={draft} initialValue={draft} onSubmit={ask} />
        </div>
      </div>
      <main className="grid grid-cols-2 gap-4 flex-1 min-h-0">
        <section className="overflow-auto bg-zinc-100 rounded-lg p-3">
          <h2 className="text-xs uppercase text-zinc-500 mb-2">原文片段</h2>
          <RetrievalPane mode={state.mode} chunks={state.chunks} />
        </section>
        <section className="overflow-auto bg-zinc-100 rounded-lg p-3">
          <h2 className="text-xs uppercase text-zinc-500 mb-2">延展 / 简历兜底</h2>
          <SummaryPane mode={state.mode} summary={state.summary} status={state.status} />
        </section>
      </main>
      <Toast message={toast} onClose={() => setToast(null)} />
    </div>
  );
}
```

- [ ] **Step 4: 手动验证**

启动后端与前端（同 Task 10 Step 7），按住 Space 键说话，松开后查看：
- 文字应自动回填到输入框
- 自动触发查询
- 双栏正常渲染

- [ ] **Step 5: 提交**

```bash
git add frontend/src/hooks/usePushToTalk.ts frontend/src/components/PushToTalkButton.tsx frontend/src/App.tsx
git commit -m "feat(frontend): push-to-talk via MediaRecorder + Space hotkey"
```

---

## Task 12: 检索质量评测脚本（不进 CI）

**Files:**
- Create: `backend/tests/eval/__init__.py`
- Create: `backend/tests/eval/cases.yaml`
- Create: `backend/tests/eval/__main__.py`

- [ ] **Step 1: 写 `tests/eval/__init__.py`（空）**

- [ ] **Step 2: 写 `tests/eval/cases.yaml`（占位 5 条，用户后续补 30-50）**

```yaml
- query: "MySQL 事务隔离级别有哪些"
  expected_mode: hit
  expected_files_contain: ["mysql/事务隔离.md"]

- query: "RR 和 RC 的区别"
  expected_mode: hit
  expected_files_contain: ["mysql/事务隔离.md"]

- query: "controller-runtime 的 Reconcile 是什么"
  expected_mode: hit
  expected_files_contain: ["k8s/controller-runtime.md"]

- query: "你们订单系统怎么做幂等"
  expected_mode: hit
  expected_files_contain: ["项目/订单系统复盘.md"]

- query: "你对 Rust 的看法"
  expected_mode: fallback
```

- [ ] **Step 3: 写 `tests/eval/__main__.py`**

```python
import asyncio
import sys
from pathlib import Path

import yaml

from core.config import load_settings
from core.embedder import OpenAIEmbedder
from core.indexer import build_or_update_index, load_index
from core.retriever import Retriever


async def main():
    settings = load_settings()
    settings.validate_for_runtime()
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    embedder = OpenAIEmbedder(settings.openai_api_key, settings.embedding_model, data_dir / "embedding_cache.json")
    bundle = load_index(data_dir) or await build_or_update_index(
        settings.notes_dir, data_dir, embedder, settings.max_chunk_tokens, settings.chunk_overlap_tokens,
    )

    cases = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text(encoding="utf-8"))
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    summary = {t: {"mode_correct": 0, "file_hit": 0, "total": 0} for t in thresholds}

    for t in thresholds:
        retr = Retriever(bundle, embedder, settings.retrieval_top_k, settings.rerank_top_k, t,
                         settings.vector_weight, settings.bm25_weight)
        for case in cases:
            res = await retr.search(case["query"])
            summary[t]["total"] += 1
            if res.mode.value == case["expected_mode"]:
                summary[t]["mode_correct"] += 1
            if "expected_files_contain" in case:
                paths = [c.chunk.file_path for c in res.chunks]
                if any(any(want in p for p in paths) for want in case["expected_files_contain"]):
                    summary[t]["file_hit"] += 1

    print(f"{'threshold':>10}  mode-acc  file-hit")
    for t, s in summary.items():
        n = max(s["total"], 1)
        print(f"{t:>10}  {s['mode_correct']/n:.2%}    {s['file_hit']/n:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 手动验证（需要 API key 与笔记）**

```bash
cd backend
python -m tests.eval
```
Expected: 输出多阈值下 mode-acc / file-hit 表格

- [ ] **Step 5: 提交**

```bash
git add backend/tests/eval/
git commit -m "feat(backend): retrieval quality eval harness with threshold sweep"
```

---

## Task 13: 测试清单 + 跑全套测试 + 收尾

**Files:**
- Create: `docs/test-checklist.md`

- [ ] **Step 1: 写 `docs/test-checklist.md`**

```markdown
# 端到端手动测试清单

后端启动：`cd backend && uvicorn main:app --port 8000 --reload`
前端启动：`cd frontend && npm run dev`
浏览器：http://localhost:5173

- [ ] 启动后浏览器可正常访问，header 显示 "面试实时辅助"
- [ ] 文字命中：输入"RR vs RC"→ 左栏 ≤2s 出 mysql/事务隔离.md 原文，右栏 ≤3s 流完延展
- [ ] 文字未命中：输入"你对 Rust 的看法"→ 右栏黄色提示 + 第一人称简历兜底回答
- [ ] push-to-talk：按住 Space → 录音指示 → 松开 → 文字回填到查询框 → 自动发送
- [ ] 连续按两次查询：旧 SSE 流被取消，新查询正常工作
- [ ] 拔网线（关闭后端）：toast 提示 "SSE 中断"，UI 不卡死，再开后端可继续
- [ ] 笔记加文件：在 NOTES_DIR 下加新 .md，重启后端，新内容能被检索
- [ ] 简历缺失：删除 resume.md，未命中场景不崩，右栏提示无简历
- [ ] 中英文混合 query：输入"k8s controller 的 reconcile 是什么"，能命中
- [ ] 代码块在原文渲染时高亮、不被截断
- [ ] ⌘K 清空：双栏被清空，焦点回到输入框
```

- [ ] **Step 2: 跑全部单元 + 集成测试**

```bash
cd backend
pytest tests/unit tests/integration -v
```
Expected: 全部 PASS

- [ ] **Step 3: 在真实环境跑一遍 checklist 上的关键三项**

至少完成：
- 文字命中 / 文字未命中
- push-to-talk
- ⌘K 清空

- [ ] **Step 4: 提交**

```bash
git add docs/test-checklist.md
git commit -m "docs: add end-to-end manual test checklist"
```
