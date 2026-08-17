# Rerank 精排流程

## 四步流水线

```
混合检索结果 (topk × 3 候选, ~15条)
  │
  ▼
① 文献级聚合
  同一 PMID 的多个 chunk 合并为 1 条
  15个chunk → 8篇文献
  │
  ▼
② LLM 逐篇打分 (0-10)
  标题 + 前500字符 → DeepSeek 批量评分
  0-2: 无关  3-5: 弱相关  6-8: 直接相关  9-10: 精准
  │
  ▼
③ 证据等级加权
  Meta-Analysis ×1.3  Systematic Review ×1.2
  RCT ×1.1           Guideline ×1.15
  其他 ×1.0
  final_score = rerank_score × weight
  │
  ▼
④ 按 final_score 降序 → 取 topk (5条)
```

## LLM 打分机制

```
一次API调用，批量评分全部候选文献：
  
  Prompt: "Score each document 0-10 on how well 
           it directly answers the question."
  
  输入: 8篇文献 (标题+摘要前500字符)
  输出: JSON {"scores": [9, 7, 3, 6, 2, 8, 4, 5]}
  
  temperature=0.0  保证打分一致性
```

## 容错

| 异常 | 兜底 |
|------|------|
| API Key缺失 | 全部赋 5.0 中性分 |
| LLM 调用失败 | 全部赋 5.0 中性分 |
| JSON 解析失败 | 打印原始输出，全部赋 5.0 |
| 候选数 ≤ topk | 跳过Rerank，直接返回 |
