# 相关产品官网

- [OpenEvidence](https://www.openevidence.com) —— 面向医生的证据检索型 AI 助手，本项目对标的产品原型
- [UpToDate](https://www.uptodate.com) —— Wolters Kluwer 旗下老牌临床决策支持工具，本项目"专用 vs 通用"对比讨论中的另一个专用工具样本

# 建议阅读

- [OpenEvidence，超过半数的美国医生都在用的AI，它凭什么？](https://mp.weixin.qq.com/s/wVTcKNsMU_cMdYp7Ee0WnQ)
- [花699美元买的专业医疗AI，输给了通用大模型](https://mp.weixin.qq.com/s/yIo4skD4g_pA5GlRhMH5WA)
- Vishwanath, K. et al. General-purpose large language models outperform specialized clinical AI tools on medical benchmarks. Nature Medicine (2026). [https://doi.org/10.1038/s41591-026-04431-5](https://doi.org/10.1038/s41591-026-04431-5)
- [NEJM AI Grand Rounds, Episode 42: The OpenEvidence Episode — Dr. Travis Zack on the Future of Clinical Evidence](https://ai-podcast.nejm.org/e/the-openevidence-episode-dr-travis-zack-on-the-future-of-clinical-evidence/)。Travis Zack，OpenEvidence 首席医疗官。首播 2026 年 5 月 20 日，Massachusetts Medical Society 出品。

# 部分学习资源

**数据获取**

- [PubMed E-utilities 官方文档](https://www.ncbi.nlm.nih.gov/books/NBK25501/) —— NCBI 官方 API 使用手册，涵盖 esearch/efetch 等接口用法，是文献数据在线检索的权威参考
- [Europe PMC RESTful API](https://europepmc.org/RestfulWebService) —— 免费公开的文献检索接口，覆盖超 3300 万篇文献，支持在线查询与开放获取内容批量下载
- [ClinicalTrials.gov Data API v2](https://clinicaltrials.gov/data-api/api) —— NIH 官方临床试验数据库的现代 REST 接口，无需鉴权，支持按条件检索和批量拉取试验记录
- [OpenAI Life Sciences Research Plugin](https://github.com/openai/plugins/tree/main/plugins/life-science-research) —— OpenAI 开源的生命科学科研 Skill 插件包，可装进 Codex/Claude Code，一次性串联 PubMed、基因组学等 50+ 数据源完成检索与综合
- [技能研究室 - 麦伴科研 MaltSci.com](https://www.maltsci.com/skills) 2000+ 科研相关skills

**文档处理与检索**

- [pymupdf4llm](https://github.com/pymupdf/pymupdf4llm) —— 面向 RAG 场景的 PDF 解析库，一行代码把 PDF 转成结构化 Markdown，支持按页切块并保留表格/版式信息
- [Chroma 官方文档](https://docs.trychroma.com) —— 开源向量数据库，`pip install chromadb` 后几行代码即可跑通embedding 存储与语义检索，是 RAG 原型开发的常用起点

**Prompt 设计与效果评估**

- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) —— RAG 概念的原始论文，了解摘要即可
- [RAG Prompt Engineering Guide (SurePrompts, 2026)](https://sureprompts.com/blog/rag-prompt-engineering-guide) —— 系统/检索/生成三层 Prompt 架构与引用格式设计的实践指南
- [Ragas](https://github.com/explodinggradients/ragas) —— 开源 RAG 评测框架，可衡量回答的 faithfulness（忠实度）与引用准确率
- [TruLens](https://github.com/truera/trulens) —— 另一款常用的开源 LLM/RAG 应用评测与可观测性工具

