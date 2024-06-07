---
title: ConvRAG：通过细粒度检索增强和自我检查提升大模型对话式问答能力
date: 2024-04-10 15:23:56
cover: https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Boosting%20Conversational%20Question%20Answering%20with%20Fine-Grained%20Retrieval%20Augmentation%20and%20Self-Check/Figure%201.png
tags:
- RAG
- 检索增强
category:
- RAG
- 查询检索
- 检索环节
---

- 论文地址：[https://arxiv.org/pdf/2403.18243.pdf](https://arxiv.org/pdf/2403.18243.pdf)
- 文章链接：[https://mp.weixin.qq.com/s/InjLKF8lepX6hfi6W-oeMQ](https://mp.weixin.qq.com/s/InjLKF8lepX6hfi6W-oeMQ)

ConvRAG 是一种对话式问答方法，通过细粒度检索增强和自我检查机制提升 LLM 在对话环境中的问题理解和信息获取能力。

文章介绍了一种名为 ConvRAG 的新型对话式问答方法，旨在增强 LLMs 的对话问答能力。ConvRAG 通过结合细粒度检索增强和自我检查机制，解决了以往 RAG 方法在单轮问答中的局限性，并将其成功适应于复杂的对话环境（问题与之前的上下文相互依赖）。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Boosting%20Conversational%20Question%20Answering%20with%20Fine-Grained%20Retrieval%20Augmentation%20and%20Self-Check/Figure%201.png)

ConvRAG 的核心在于三个关键组件的协同工作：

- **对话式问题精炼器**：通过问题重构和关键词提取，使问题意图更加明确，以便更好地理解与上下文相关联的问题。
- **细粒度检索器**：利用问题重构和关键词从网络中检索最相关的信息，以支持响应生成。检索过程包括文档级检索、段落级召回和段落级重排，以确保获取到最有用的信息片段。
- **基于自我检查的响应生成器**：在生成响应之前，先对检索到的信息进行自我检查，以确保使用的是有用的信息，从而提高响应的准确性。
