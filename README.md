# 面试实时辅助 RAG

候选人面试中实时查询本地 Markdown 笔记，命中显示原文 + LLM 延展，未命中基于简历兜底生成。

- 输入：文字 / 浏览器原生语音（按住 Space）
- 输出：左栏笔记原文（top-3）+ 右栏 LLM 流式延展或简历兜底
- 全程本地 Web，浏览器打开 localhost 即用

## 1. 准备工作

### 1.1 笔记格式（重要）

笔记放进 `notes/` 目录，可任意层级子目录，全是 `.md` 文件即可。

**核心原则**：让每个"问答单元 / 知识点"成为一个 H2（`##`）小节，章节用 H1（`#`）。chunker 按 H2 切片，所以 H2 的粒度直接决定召回粒度。

#### 推荐结构

```markdown
# 一、AI Agent 核心技术

## Agent Loop 是什么

正文……可包含若干段、代码块、列表，但围绕一个主题。

```python
def agent_loop(): ...
```

## 死循环怎么防

正文……

# 二、前端技术

## iframe 微前端通信怎么做

正文……

## React fiber 原理

正文……
```

- **H1 (`#`)**：章节大类，作为面包屑顶层（"二、前端技术 > React fiber 原理"）。可以有多个 H1，每个 H2 自动归入上方最近的 H1。
- **H2 (`##`)**：一个独立的问答单元 / 知识点。**这是最关键的层级** —— chunker 在这里切片。
- **不要用 H3 (`###`)** 当切点。如果你有 H3 子标题，要么提升为 H2，要么把它当作正文里的小标题（不被切）。
- 章节内可任意用 ` ```代码块``` `、列表、引用、表格 —— chunker 会保护代码块完整不被截断。

#### 反模式（会导致检索结果混乱）

```markdown
# 模拟面试稿

## 全部内容

### 自我介绍
...

### Agent Loop
...

### iframe 微前端
...
```

整篇只有一个 H2，所有内容堆在一起。chunker 切不开，召回时会把"自我介绍 + Agent Loop + iframe"混在一个 chunk 里。

如果已经写成这种结构，运行 `sed -i '' 's/^### /## /g; s/^## /# /g' notes/yourfile.md` 一键提升标题层级即可。

#### 单个 chunk 的合理大小

`MAX_CHUNK_TOKENS=500`（约 700 中文字）。实际效果：

- H2 section ≤ 500 token → 整段一个 chunk（最佳）
- H2 section > 500 token → 按段落再切，相邻 chunk 留 50 token overlap
- H2 section 含 token 超长的代码块 → 代码块整体保留，不切

每个 H2 控制在 **300-700 字** 范围内体感最好。如果某个问答展开了几百字，可以拆成两三个 H2（如"Agent Loop 是什么"、"Agent Loop 死循环防控"、"Agent Loop 上下文压缩"）。

### 1.2 简历（可选）

放在仓库根的 `resume.md`，用于"未命中笔记"时让 LLM 以第一人称基于简历回答。

- 格式：自由 Markdown，写工作经历 + 项目经验 + 技术栈即可
- 大小上限：8000 token（约 5000 中文字），超出自动截断
- PDF 简历可用 `pdftotext -layout 简历.pdf resume.md` 抽成 Markdown

不放也能跑，未命中场景会显示"未配置简历"。

### 1.3 API Key

至少需要两个：

- **嵌入** —— 用于把笔记 / 查询转向量。OpenAI 兼容端点都行（OpenAI / 阿里云 DashScope / SiliconFlow / 自建）
- **LLM** —— 用于生成延展或简历兜底回答。DeepSeek / OpenAI / Claude 都行

国内推荐组合：

| 用途 | 提供方 | 模型 | 备注 |
|---|---|---|---|
| 嵌入 | 阿里云 DashScope | `text-embedding-v4` (1024 维) | OpenAI 兼容端点，需开通付费（有免费额度）|
| LLM | DeepSeek | `deepseek-chat` | 中文好、便宜、首字快 |
| 语音 | 浏览器 Web Speech | — | Chrome / Edge 原生，零成本，零部署 |

## 2. 启动

```bash
# 1. 配置
cp .env.example .env
# 编辑 .env：填 API key、模型、base_url（见下方示例）

# 2. 后端
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e .
cd ..
python -m core.indexer --rebuild  # 首次构建索引（在仓库根运行，不是 backend/）

# 3. 启动后端
( cd /Users/mi/Desktop/RAG && \
  /Users/mi/Desktop/RAG/backend/.venv/bin/uvicorn \
    --app-dir backend main:app --port 8000 --reload )

# 4. 启动前端（新终端）
cd frontend
npm install
npm run dev
```

浏览器打开 http://localhost:5173。

> **注意**：`indexer` 与 `uvicorn` 都必须在仓库根目录运行（不是 `backend/`），这样 `Settings(env_file=".env")` 才能找到根目录的 `.env`。

### 2.1 `.env` 示例（DashScope 嵌入 + DeepSeek LLM + 浏览器 ASR）

```env
NOTES_DIR=./notes
RESUME_PATH=./resume.md

# 嵌入：阿里云 DashScope (OpenAI 兼容)
EMBEDDING_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_BATCH_SIZE=10  # 阿里 v4 上限 10；OpenAI 可 100

# LLM：DeepSeek
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat

# ASR：浏览器原生
ASR_PROVIDER=browser

# 检索（按真实使用情况校准）
RETRIEVAL_THRESHOLD=0.4   # < 0.5 算命中 hit；建议先 0.4，跑 30+ 真问题再调
VECTOR_WEIGHT=0.7
BM25_WEIGHT=0.3
```

完整字段见 `.env.example`。

## 3. 使用

### 3.1 文字查询

输入框输入问题，回车提交。

- 左栏立即显示笔记 top-3 原文片段，标注面包屑（章节 > 小节）和相似度
- 右栏流式输出 LLM 延展（命中模式）或简历兜底（未命中模式）

### 3.2 语音查询（仅 Chrome / Edge）

按住 **Space** 键说话 → 松开 → 文字自动回填到输入框 → 自动提交。

需要授权浏览器麦克风权限。如果不可用，会用文字输入。

### 3.3 快捷键

- `Space`（按住）— 语音录音
- `⌘K` / `Ctrl+K` — 清空双栏 + 取消进行中的请求

### 3.4 增量更新笔记

修改 / 新增 / 删除笔记后，重启后端即可：

```bash
# 杀掉旧进程
kill $(lsof -nP -iTCP:8000 -sTCP:LISTEN -t)
# 重启（自动检测文件变化、只重嵌入修改过的部分）
( cd /Users/mi/Desktop/RAG && \
  backend/.venv/bin/uvicorn --app-dir backend main:app --port 8000 )
```

启动时会按 sha256 比对文件，未变的笔记复用旧向量、改过的重切重嵌。

如果想强制全量重建（例如换了嵌入模型）：

```bash
python -m core.indexer --rebuild
```

## 4. 校准检索阈值（可选）

`RETRIEVAL_THRESHOLD` 决定"命中 hit"还是"未命中 fallback"的边界。默认 0.4，建议用真实问题集校准。

```bash
# 在 backend/tests/eval/cases.yaml 写 30-50 条真实面试问题：
# - query: "Agent Loop 怎么实现"
#   expected_mode: hit
#   expected_files_contain: ["笔记.md"]
# - query: "你对 Rust 的看法"
#   expected_mode: fallback

cd /Users/mi/Desktop/RAG
source backend/.venv/bin/activate
python -m tests.eval
```

输出示例：

```
 threshold  mode-acc  file-hit
       0.3   85.00%   90.00%
       0.4   92.00%   90.00%   ← 选这个
       0.5   80.00%   88.00%
       0.6   65.00%   85.00%
```

挑 mode-acc 最高的那个写进 `.env`。

## 5. 故障排查

| 现象 | 原因 / 修复 |
|---|---|
| `/api/health` 返回 `ok=false, config_error=...` | API key 没配；编辑 `.env` 后重启 |
| 索引时报 `batch size invalid, ≤ 10` | DashScope v4 上限；设 `EMBEDDING_BATCH_SIZE=10` |
| 索引时报 `Arrearage` | 阿里云账户未开通付费；去 [bailian.console.aliyun.com](https://bailian.console.aliyun.com) 开通 |
| 召回内容混乱、命中无关章节 | 笔记结构问题；检查是否所有内容堆在单个 H2 下；按 1.1 节调整 |
| 命中相似度普遍偏低 | 嵌入模型与笔记语种不匹配；中文笔记建议用 DashScope `text-embedding-v4` 或 BGE 系列 |
| 浏览器语音不识别 | 仅 Chrome / Edge 支持；Safari / Firefox 用文字 |
| `npm run dev` 起不来 | 删 `frontend/node_modules` + `package-lock.json` 重 `npm install` |

## 6. 项目结构

```
RAG/
├─ notes/              # 你的 Markdown 笔记（gitignored）
├─ resume.md           # 你的简历（gitignored）
├─ .env                # API key 与配置（gitignored）
├─ backend/
│   ├─ main.py
│   ├─ core/
│   │   ├─ chunker.py    # Markdown 切片（H1+H2 边界 + 代码块保护）
│   │   ├─ indexer.py    # FAISS 索引（增量构建 + 原子写入）
│   │   ├─ retriever.py  # 向量+BM25 混合检索
│   │   ├─ embedder.py   # OpenAI 兼容嵌入客户端
│   │   ├─ resume.py     # 简历加载（8000 token 截断）
│   │   ├─ llm/          # DeepSeek / OpenAI / Claude provider
│   │   └─ prompts.py    # 命中延展 + 简历兜底 prompt
│   ├─ routes/
│   │   ├─ ask.py        # GET /api/ask (SSE)
│   │   └─ asr.py        # POST /api/asr (Whisper, 仅 ASR_PROVIDER=openai 时用)
│   ├─ tests/
│   │   ├─ unit/         # pytest 单元
│   │   ├─ integration/  # FastAPI TestClient
│   │   └─ eval/         # 检索质量评测（手动跑）
│   └─ data/             # 索引产物（gitignored）
└─ frontend/
    └─ src/
        ├─ App.tsx
        ├─ hooks/
        │   ├─ useAsk.ts          # SSE 订阅
        │   └─ usePushToTalk.ts   # Web Speech API
        └─ components/
            ├─ RetrievalPane.tsx  # 左栏原文
            ├─ SummaryPane.tsx    # 右栏延展 / 兜底
            ├─ QueryInput.tsx
            ├─ PushToTalkButton.tsx
            └─ Toast.tsx
```

## 7. 设计文档

- 完整 spec：`docs/superpowers/specs/2026-05-28-interview-rag-design.md`
- 实施计划：`docs/superpowers/plans/2026-05-28-interview-rag.md`
- 手动测试清单：`docs/test-checklist.md`
