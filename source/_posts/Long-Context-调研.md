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

对不同 Transformer 层采用不同的 kv cache 压缩设置，底层尽量都保留，高层仅保留关键 token 的 kv cache：[2.5%KV缓存保持大模型90%性能，大模型金字塔式信息汇聚模式探秘｜开源 - 量子位的文章 - 知乎](https://zhuanlan.zhihu.com/p/703313505)
