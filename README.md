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
