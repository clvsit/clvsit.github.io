---
title: Long Context 调研
date: 2024-06-10 22:43:57
tags:
---

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/LLM/LongContext/Long%20Context.png)

- [Transformer 架构中的位置编码](https://clvsit.github.io/Transformer-%E6%9E%B6%E6%9E%84%E4%B8%AD%E7%9A%84%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81/)

# 推理优化

## kv cache 压缩
随着模型规模的增大，推理需要的时间越来越多。kv cache 作为推理加速的关键技术，通过缓存之前的解码步骤中计算出的 `key_states` 和 `value_states` 来避免在后续解码过程中重复计算，从而减少解码时间（起到空间换时间的作用）。

但是，随着序列长度增大，需要缓存的 kv cache 呈线性增长，占用大量显存。针对这一问题，之前的工作设计策略是对 kv cache 进行压缩。

[论文阅读：《Sequence can Secretly Tell You What to Discard》，减少推理阶段的 kv cache](https://clvsit.github.io/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%EF%BC%9A%E3%80%8ASequence-can-Secretly-Tell-You-What-to-Discard%E3%80%8B%EF%BC%8C%E5%87%8F%E5%B0%91%E6%8E%A8%E7%90%86%E9%98%B6%E6%AE%B5%E7%9A%84-kv-cache/)
- **论文地址**：https://arxiv.org/abs/2404.15949
- **发表日期**：2024-04-24
- **简要介绍**：研究发现在 LLaMA2 系列模型上：（i）相邻 token 的 query 向量之间的相似度非常高，（ii）当前 query 的注意力计算可以完全依赖于一小部分前面 query 的注意力信息。基于这些观察结果，作者提出了一种无需重新训练的 KV 缓存驱逐策略 CORM，通过重复使用最近的 query 注意力信息来显著减少显存占用。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Sequence%20can%20Secretly%20Tell%20You%20What%20to%20Discard/Figure%202.png)

> 该论文与 StreamingLLM 中提出的 attention sink、Label are anchor 等论文以及 prompt 压缩的可行性都能在理论方面进行印证。但该论文提出的 CORM 方法也存在一个缺陷，是否存在某个 key 与未来一段时间内的 query 的相似度都很低，在这之后才出现较高的相似度。而此时，如果将该 key 认为是不重要而进行丢弃的话，未来的 query 可能获取不到这部分信息。

[2.5%KV缓存保持大模型90%性能，大模型金字塔式信息汇聚模式探秘｜开源 - 量子位的文章 - 知乎](https://zhuanlan.zhihu.com/p/703313505)
- **论文地址**：https://arxiv.org/abs/2406.02069
- **发表日期**：2024-06-04
- **GitHub 仓库**：https://github.com/Zefan-Cai/PyramidKV
- **简要介绍**：对不同 Transformer 层采用不同的 kv cache 压缩设置，底层尽量都保留，高层仅保留关键 token 的 kv cache。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/PyramidKV%20Dynamic%20KV%20Cache%20Compression%20based%20on%20Pyramidal%20Information%20Funneling/Figure%203.png)
