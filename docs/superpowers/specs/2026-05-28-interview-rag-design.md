# 面试实时辅助 RAG 系统 — 设计文档

- 日期：2026-05-28
- 状态：待用户审阅

## 1. 项目目标

为面试候选人提供实时辅助工具：候选人在面试中听到问题后，通过语音或文字快速查询本地笔记，秒级展示对应内容。命中笔记时直接展示原文 + LLM 延展；未命中时基于个人简历用第一人称兜底生成。

### 成功标准

- 命中模式：从松开热键/提交查询到左栏出现原文 ≤ 2s
- 未命中模式：右栏首字 ≤ 2s，整段流式完成 ≤ 5s
- 单用户、本地运行，浏览器 localhost 访问
- 失败安静、有兜底，不打断面试节奏

### 范围外（明确不做）

- 多用户、云端部署
- 实时文件监听（笔记不会面试中改）
- 性能压测、CI/CD 流水线
- 错误上报到外部服务

## 2. 整体架构

```
┌──── React 前端 (Vite, dev:5173 / prod 静态托管) ────┐
│  PushToTalkButton  ──► POST /api/asr ─► 文本回填    │
│  QueryInput        ──► GET  /api/ask  (SSE)         │
│  双栏布局：左原文 / 右 LLM 延展或简历兜底           │
└─────────────────────────┬───────────────────────────┘
                          │
┌──── FastAPI 后端 (Python 3.11+, uvicorn) ───────────┐
│  routes/                                             │
│   ├─ POST /api/asr   音频 multipart  → text          │
│   └─ GET  /api/ask   query → SSE 事件流              │
│                                                      │
│  core/                                               │
│   ├─ indexer.py    Markdown 切块 → 嵌入 → FAISS     │
│   ├─ retriever.py  向量召回 + BM25 重排              │
│   ├─ llm.py        provider 抽象，流式 yield         │
│   ├─ asr.py        Whisper API 客户端                │
│   ├─ resume.py     简历加载                          │
│   └─ config.py     .env 加载                         │
│                                                      │
│  data/                                               │
│   ├─ index.pkl     FAISS IndexFlatIP                 │
│   ├─ chunks.json   chunk 列表（与 FAISS 同序）       │
│   └─ meta.json     索引元数据                        │
└─────────────────────────┬───────────────────────────┘
                          │
              OpenAI Embeddings / Whisper API
              LLM API (默认 DeepSeek，可切 Claude/OpenAI)
```

### 关键决策

- **单进程 FastAPI**：单用户场景，不引入 Celery/Redis
- **FAISS in-memory**：< 5MB 笔记，启动加载到内存毫秒级查询
- **SSE 流式**：单向流，比 WebSocket 简单（FastAPI `StreamingResponse` + 浏览器 `EventSource` 原生）
- **单一端点 `/api/ask` 自适应模式**：前端只订阅一个 SSE，无需先调检索再决定要不要调流式
- **LLM provider 可插拔**：默认 DeepSeek（中文好、便宜、首字快），`.env` 一行切换

## 3. 索引与切块

### 笔记目录约定

```
notes/                      ← .env 中 NOTES_DIR
  ├─ mysql/
  │   └─ 事务隔离.md
  ├─ k8s/
  │   └─ controller-runtime.md
  └─ 项目/
      └─ 订单系统复盘.md
```

启动时递归扫描所有 `.md`，子目录任意层。

### 切块策略

1. 用 `markdown-it-py` 解析 Markdown AST
2. 按二级标题 `##` 切大段
3. 大段超过 `MAX_CHUNK_TOKENS`（默认 500）时按段落切；段落超长按句切
4. 代码块（` ``` ` 围起来）整体保留，不被切开
5. 相邻 chunk 留 50 token overlap

不用 LangChain 的 RecursiveCharacterTextSplitter：对中文 + 代码块处理不如手写可控。

### Chunk 数据结构

```python
@dataclass
class Chunk:
    id: str                  # hash(file_path + offset)
    text: str                # chunk 正文
    file_path: str           # 相对 notes/ 的路径
    heading_path: list[str]  # ["MySQL", "事务隔离", "RR vs RC"]
    line_start: int
    line_end: int
```

`heading_path` 在嵌入时拼到 chunk 前面（"MySQL > 事务隔离 > RR vs RC: <chunk text>"），让短 chunk 也带上下文。

### 嵌入

- 模型：`text-embedding-3-small`（OpenAI），1536 维
- 批量调用：每批 100 chunk
- 缓存：`data/embedding_cache.json`，key = sha256(text)，重新索引时同样的 chunk 不重新调 API

预估：< 5MB 笔记 ≈ 500 chunk，全量嵌入 ≈ $0.005、< 5s。

### 索引文件

```
data/
  index.pkl       FAISS IndexFlatIP（精确内积）
  chunks.json     chunk 列表，与 FAISS 同序
  meta.json       { last_indexed_at, files_hash, embedding_model }
```

用 IndexFlatIP 不用 HNSW：500 个向量精确查询 < 1ms，HNSW 没意义。

### 增量索引

启动时对比 `meta.json.files_hash` 与当前文件树：
- 新文件：嵌入并加入
- 修改的文件（mtime 变了）：删旧 chunk、嵌入新 chunk
- 删除的文件：移除对应 chunk

CLI：`python -m core.indexer --rebuild` 强制全量重建。

## 4. 检索与回答

### 流程

```
用户 query
   │
   ▼
向量召回 top-8 + BM25 重排 → top-3
score = 0.7 * vec_sim + 0.3 * bm25_norm
   │
   ▼
top-1 score ≥ THRESHOLD(0.5)?
   │
   ├─ 是 → 命中：左栏返回 top-3 原文 + 右栏 LLM 流式延展
   │
   └─ 否 → 未命中：右栏基于简历 + query + 弱参考 chunks 流式生成第一人称回答
                   左栏折叠展示弱相关片段（灰显）
```

### BM25 实现

`rank_bm25` 库 + jieba 分词 + 停用词过滤，索引时建好 `BM25Okapi` 实例存到 pickle，查询时直接打分。

### 阈值

- `RETRIEVAL_THRESHOLD=0.5` 起点，`.env` 可调
- 响应附带 `top1_score` 与 `mode`，便于校准
- 实施后用真实问题集（30-50 条）评测调整

### 简历加载

- 启动时读 `RESUME_PATH`（默认 `resume.md`），整篇加载到内存
- 限制：≤ 8000 token（约 5000 中文字），超长截断并 warn
- 不进 FAISS 索引（不污染笔记检索）

### 接口契约：`GET /api/ask`

```
GET /api/ask?query=...
Content-Type: text/event-stream

# 命中场景
event: mode
data: {"mode": "hit", "top1_score": 0.78, "elapsed_ms": 187}

event: chunks                       # 左栏立即渲染
data: [{"id":"...","file_path":"...","heading_path":[...],"text":"...","score":0.78,"line_start":12,"line_end":47}, ...]

event: token                        # 右栏流式延展
data: "▎"
event: token
data: "可"
...

event: done
data: {"elapsed_ms": 2950}

# 未命中场景
event: mode
data: {"mode": "fallback", "top1_score": 0.31}

event: chunks                       # 弱相关，左栏折叠/灰显
data: [...]

event: token                        # 右栏：第一人称简历兜底
data: "我"
...
event: done
```

### Prompt：命中模式（延展）

```
你是候选人本人的实时面试助手。候选人的笔记已经直接覆盖了这个问题，
原文已显示在屏幕上。你的任务是基于这些原文做"延展"，让候选人答得更深。

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
{top3_chunks}
```

### Prompt：未命中模式（简历兜底）

```
你是候选人本人，正在面试中。基于"我的简历"回答面试官的问题。

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
{resume_full_text}

【笔记中的弱相关片段（仅供参考，可不用）】
{top3_chunks_with_low_score}
```

### LLM provider 抽象

```python
# core/llm.py
class LLMProvider(Protocol):
    async def stream(self, prompt: str) -> AsyncIterator[str]: ...

class DeepSeekProvider:    # 默认
class OpenAIProvider:      # gpt-4o-mini
class ClaudeProvider:      # claude-haiku-4-5
```

`.env`：`LLM_PROVIDER=deepseek`。

## 5. ASR

- 提供方：OpenAI Whisper API
- 触发：push-to-talk 热键（前端 MediaRecorder 录 webm/opus）
- 路径：松开热键 → POST `/api/asr` multipart 音频 → 后端转发 Whisper → 返回文本 → 前端回填到查询框 → 自动提交

为什么不用浏览器 Web Speech API：候选人讲技术术语（"事务隔离级别 RR"、"controller-runtime"）频繁，Whisper 中文专业术语识别明显更稳。

## 6. 前端

### 技术栈

- Vite + React 18 + TypeScript
- 状态管理：`useState`（不引 Redux/Zustand）
- 样式：Tailwind
- Markdown 渲染：`react-markdown` + 代码高亮（`rehype-highlight`）
- 流式：`EventSource`（浏览器原生）

### 组件树

```
<App>
  <PushToTalkButton />        # 录音 + 调 /api/asr
  <QueryInput />              # 文字 + 回填，提交触发 EventSource
  <RetrievalPane />           # 左栏：top-3 原文（命中）或弱相关折叠（未命中）
  <SummaryPane />             # 右栏：LLM 流式延展或简历兜底
  <Toast />                   # 错误提示，3s 自动消失
</App>
```

### 模式渲染

```
mode=hit:
  左栏：原文 top-3，每个 chunk 顶部显示 heading_path 面包屑 + score
  右栏：✓ 笔记命中 标签 + 流式延展（▎可延展｜▎追问｜▎踩坑）

mode=fallback:
  左栏：⚠ 弱相关，可折叠
  右栏：🟡 笔记未直接命中，基于简历回答 + 流式第一人称回答
```

### 交互细节

- ⌘K 清空两栏 + 取消进行中请求
- 连续提交：前端主动 close 旧 EventSource；后端 detect 客户端断开后立即取消上游 LLM 流（节省额度）
- 错误提示用底部 toast，禁用 modal/alert

### 部署

- Dev：Vite 5173，FastAPI 8000，前端 `proxy` 到 8000
- Prod（自用）：FastAPI 直接托管 `frontend/dist`

## 7. 错误处理与降级

| 失败点 | 降级策略 |
|---|---|
| Whisper API 超时/失败 | toast "语音识别失败，请打字"；自动聚焦文字输入框 |
| 嵌入 API 失败 | 降级纯 BM25 检索；toast "向量服务异常，仅关键词匹配" |
| LLM 首字超时（>10s） | 中止流，右栏 "⚠ 总结生成失败"；左栏不受影响 |
| LLM 流中断 | SSE `event: error`；右栏保留已收到内容 + "（生成中断）" |
| 检索 0 命中 + 简历未配置 | `mode: "empty"`，前端显式提示 + 配置入口 |
| 笔记目录不存在/为空 | 启动不挂；Web 显示引导 |
| 索引文件损坏 | 自动全量重建；备份损坏文件到 `data/index.pkl.broken-<ts>` |
| 同时多个查询 | 前端 close 旧流；后端 detect 断开后取消上游 |
| API Key 未配置 | 启动校验；Web 显示配置引导，不直接 500 |

### 超时配置（`.env`）

```
ASR_TIMEOUT_S=8
EMBED_TIMEOUT_S=5
LLM_FIRST_TOKEN_TIMEOUT_S=10
LLM_TOTAL_TIMEOUT_S=30
```

一律用 `asyncio.wait_for`。

### 重试策略

| 调用 | 重试 |
|---|---|
| Whisper | 不重试（让用户改打字） |
| Embedding（query 时） | 1 次，间隔 200ms；失败降级 BM25 |
| Embedding（索引时） | 3 次指数退避；失败 chunk 跳过并记录 |
| LLM stream | 不重试（流式重试体验差） |

### SSE 错误事件契约

```
event: error
data: {
  "stage": "asr" | "embed" | "retrieve" | "llm",
  "message": "可读错误（脱敏，不暴露 API key）",
  "recoverable": true | false
}
```

### 日志

- `logging` 模块，按 stage 打 INFO/WARN/ERROR
- 每次请求一条结构化日志：`{request_id, query, mode, top1_score, elapsed_ms_per_stage, errors}`
- `logs/app.log` 按天 rotate，保留 7 天
- 错误同步 stderr

## 8. 测试策略

### 单元测试（pytest）

| 模块 | 测试重点 |
|---|---|
| `indexer.py` | Markdown 切块边界、代码块完整性、超长段落、overlap、line_start/line_end 准确 |
| `retriever.py` | 混合分数计算（mock 嵌入）、阈值判定、top-k 排序 |
| `llm.py` | provider 接口契约（mock HTTP）、流式 yield、超时取消 |
| `asr.py` | 错误处理（mock 4xx/5xx/超时） |
| `config.py` | .env 加载、缺失字段、路径解析 |

嵌入和 LLM 通过抽象接口 mock，单元测试不调真实 API。

### 集成测试

用真实笔记小样本 + mock LLM，跑 FastAPI TestClient：

- `test_hit_mode_returns_chunks_immediately`：mode=hit、top1_score≥阈值、chunks 含目标文件
- `test_fallback_mode_uses_resume`：mode=fallback、prompt 含简历、流式 token 输出
- `test_embedding_failure_falls_back_to_bm25`：mock 嵌入异常、走 BM25 路径
- `test_sse_event_order_and_contract`：mode → chunks → token* → done 顺序与字段

### 检索质量评测

`tests/eval/cases.yaml` 真实面试问题集（30-50 条）：

```yaml
- query: "MySQL 事务隔离级别有哪些"
  expected_mode: hit
  expected_files_contain: ["mysql/事务隔离.md"]

- query: "你对 Rust 的看法"
  expected_mode: fallback
```

跑法：`python -m tests.eval` 输出 hit/fallback 召回率、文件命中率、阈值建议。不在 CI 跑，是校准工具。

### 端到端手动 checklist

`docs/test-checklist.md` 包含：

- 启动后浏览器可访问
- 文字命中：左栏 ≤2s 出原文，右栏 ≤3s 流完
- 文字未命中：右栏黄色提示 + 第一人称回答
- push-to-talk：录音 → 文字回填 → 自动发送
- 连续查询：旧流被取消
- 拔网线：toast 提示，UI 不卡死
- 增量索引：加文件、重启、能查到
- 简历缺失：显式提示，不崩
- 中英文混合 query
- 代码块原文渲染高亮、不截断

### 测试命令

```bash
pytest tests/unit          # <5s
pytest tests/integration   # <30s
python -m tests.eval       # 检索质量评测
```

## 9. 技术栈与依赖

### 后端 Python

```
fastapi
uvicorn[standard]
python-dotenv
markdown-it-py            # Markdown AST
faiss-cpu                 # 向量检索
rank-bm25                 # BM25
jieba                     # 中文分词
openai                    # 嵌入 + Whisper + LLM 之一
httpx                     # 通用 HTTP（DeepSeek/Claude）
pydantic                  # schema
pytest                    # 测试
pytest-asyncio
```

### 前端

```
react react-dom
typescript
vite
tailwindcss
react-markdown rehype-highlight
```

## 10. 配置与目录结构

### 项目结构

```
RAG/
├─ backend/
│   ├─ main.py                FastAPI 入口
│   ├─ routes/
│   │   ├─ asr.py
│   │   └─ ask.py
│   ├─ core/
│   │   ├─ indexer.py
│   │   ├─ retriever.py
│   │   ├─ llm/
│   │   │   ├─ base.py
│   │   │   ├─ deepseek.py
│   │   │   ├─ openai.py
│   │   │   └─ claude.py
│   │   ├─ asr.py
│   │   ├─ resume.py
│   │   └─ config.py
│   ├─ data/                  # gitignore
│   ├─ logs/                  # gitignore
│   └─ tests/
├─ frontend/
│   ├─ src/
│   │   ├─ App.tsx
│   │   ├─ components/
│   │   └─ hooks/
│   ├─ index.html
│   ├─ vite.config.ts
│   └─ tailwind.config.js
├─ notes/                     # 用户笔记，gitignore
├─ resume.md                  # 用户简历，gitignore
├─ docs/
│   ├─ superpowers/specs/
│   └─ test-checklist.md
├─ .env.example
├─ .gitignore
└─ README.md
```

### `.env.example`

```
# 笔记与简历
NOTES_DIR=./notes
RESUME_PATH=./resume.md

# API Keys（按所选 provider 填）
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
ANTHROPIC_API_KEY=

# 模型配置
EMBEDDING_MODEL=text-embedding-3-small
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
ASR_PROVIDER=openai
ASR_MODEL=whisper-1

# 检索参数
RETRIEVAL_TOP_K=8
RERANK_TOP_K=3
RETRIEVAL_THRESHOLD=0.5
VECTOR_WEIGHT=0.7
BM25_WEIGHT=0.3
MAX_CHUNK_TOKENS=500
CHUNK_OVERLAP_TOKENS=50

# 超时（秒）
ASR_TIMEOUT_S=8
EMBED_TIMEOUT_S=5
LLM_FIRST_TOKEN_TIMEOUT_S=10
LLM_TOTAL_TIMEOUT_S=30
```

## 11. 时延预算

| 模式 | 阶段 | 时间点 |
|---|---|---|
| 命中 | 左栏出现（原文） | ~1.5s |
| 命中 | 右栏首字（延展） | ~2.0s |
| 命中 | 右栏完成 | ~3-4s |
| 未命中 | 右栏首字（兜底） | ~1.8s |
| 未命中 | 右栏完成 | ~3-5s |

ASR（5s 音频）= ~1.0s；嵌入 query = ~0.2s；FAISS+BM25 = ~0.05s；LLM 首字（DeepSeek）= ~0.7s。

## 12. 不做的事（YAGNI 汇总）

- 多用户、云端部署、CI/CD
- 实时文件监听
- 性能压测、错误上报
- 前端 UI 单元测试
- LangChain/LlamaIndex 框架
- 向量量化、HNSW
- 请求队列
- LLM 流式重试
