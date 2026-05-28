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
