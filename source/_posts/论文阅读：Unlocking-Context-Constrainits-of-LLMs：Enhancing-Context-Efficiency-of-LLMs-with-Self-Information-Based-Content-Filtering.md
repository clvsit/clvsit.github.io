---
title: >-
  论文阅读：Unlocking Context Constrainits of LLMs：Enhancing Context Efficiency of
  LLMs with Self-Information-Based Content Filtering
date: 2024-03-03 17:16:00
cover: https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Unlocking%20Context%20Constrainits%20of%20LLMs%20Enhancing%20Context%20Efficiency%20of%20LLMs%20with%20Self-Information-Based%20Content%20Filtering/Figure%202.png
mathjax: true
tags:
- 论文阅读
- prompt 工程
- prompt 压缩
category:
- prompt 工程
- prompt 压缩
---

- **论文地址**：https://arxiv.org/abs/2304.12102
- **GitHub 仓库**：https://github.com/liyucheng09/Selective_Context
- **HuggingFace Demo**：https://huggingface.co/spaces/liyucheng/selective_context

大型语言模型（LLMs）在各种任务中表现出色，因而受到广泛关注。然而，其固定的上下文长度在处理长文档或保持长时间对话时带来了挑战。本文**提出了一种名为“选择性上下文”（Selective Context）的方法，利用自信息过滤掉信息量较少的内容，从而提高固定上下文长度的效率**。作者在不同数据源（包括学术论文、新闻报道和对话记录）的摘要和问题解答任务中演示了该方法的有效性。

# 2. 自信息

自信息（Self-information），又称为信息量，是信息论中的一个基本概念，它量化了一个事件所传递的信息量。在语言建模的上下文中，这里的事件是生成的一个步骤（即一个 token）。它被定义为 token 的负对数可能性：

$$
I(x) = - log_2 P(x_t | x_0, x_1, \ldots, x_{t-1}) \tag{1}
$$

其中，I(x) 表示 token x 的自信息，P(x) 表示其输出概率。

在信息理论中，自信息衡量的是与事件相关的不确定程度；罕见事件传递的信息较多，因此自信息较高，而常见事件传递的信息较少，自信息较低。在语言建模中，自信息可用于评估词汇单位（如单词、短语或句子）的信息量，以了解哪些信息更有可能是新的或对理解上下文更重要。

自信息通常不直接用于 NLP。相反，熵（entropy）和复杂度（perplexity）等密切相关的术语被广泛用于语言模型的优化和评估。

$$
H(S) = \frac{1}{N} \sum_t I(x_t) \tag{2}
$$

$$
PPL(S) = 2^{H(S)} \tag{3}
$$

其中，句子 $S = (x_0, \ldots, x_n)$ 的熵 H(S) 是句子中词语的平均自信息量，句子的困惑度 PPL(S) 可以用熵来计算。与该方法特别相关的自信息属性是可加性。

$$
\begin{aligned} I\left(x_{0}, x_{1}\right) & =-\log _{2} P\left(x_{0}, x_{1}\right) \\ & =-\log _{2} P\left(x_{0}\right) P\left(x_{1} \mid x_{0}\right) \\ & =-\log _{2} P\left(x_{0}\right)-\log _{2} P\left(x_{1} \mid x_{0}\right) \\ & =I\left(x_{0}\right) I\left(x_{1}\right)\end{aligned}
$$

这意味着我们只需将一个词汇单元中的短语的自信息相加，就能计算出该词汇单元的自信息。

# 方法

在本节中，将详细介绍提出的“选择性上下文”方法，该方法通过过滤掉信息量较少的内容来优化 LLM 中上下文长度的使用。其主要思路是计算给定上下文中词汇单位（如句子、短语或 token）的自信息，并利用它来评估其信息量。首先计算上下文中每个 token 的自信息，然后根据短语或句子等词汇单位合并 token 及其自信息。整个方法包括以下步骤：

## 计算自信息

给定上下文 $C = x_0, x_1, \ldots, x_n$，其中 $x_i$ 表示 token，使用语言模型 M 计算每个 token $x_t$ 的自信息：

$$
I(x_i) = - log_2 P(x_i | x_0, x_1, \ldots, x_{i - 1}) \tag{8}
$$

语言模型是 causal 语言模型，如 GPT-2、OPT 和 LLaMA。

## 合并到词汇单位

如果直接在 token 级别对选择性上下文进行内容过滤，可能会导致上下文非常不连贯。因此，除了 token 级别的过滤，作者还在短语和句子层面进行过滤，将过滤中的基本单位称为词汇单位（可以是一个 token、一个短语或一个句子）。

为了对短语和句子进行选择性上下文处理，应该将短语及其自信息合并为词汇单元。对于每个词汇单位 $u = (x_t, \ldots, x_{t+\alpha})$，可以根据自信息的可加性，将各个短语的自信息相加来计算其自信息：

$$
I(u) = \sum_{i = t}^{\alpha} I(x_i) \tag{9}
$$

使用句子 tokenizer 来获取句子级别的词汇单位。接着使用 spacy 将 token 合并为名词短语。作者没有合并动词短语，因为这可能会产生超长的短语。

## 选择性保留信息上下文

计算出每个词汇单元的自我信息后，就可以评估它们的信息量了。作者建议使用基于百分位数的过滤方法来自适应地选择信息量最大的内容，而不是使用固定的阈值或保留固定数量的前 k 个词汇单元。

首先，根据词汇单位的自信息值降序排列。然后，计算所有词汇单位中自信息值的 p-th 百分位数。

$$
I_p = np.percentile([I(u_0), \ldots, I(u_k)], p) \tag{10}
$$

接下来，有选择性地保留自信息值大于或等于 p-th 百分位数的词汇单元，从而构建一个过滤后上下文 C′：

$$
C' = U_i | I(U_i) \geq I_p, 1 \leq i \leq n \tag{11}
$$

基于百分位数的过滤是一种更灵活的方法，可以根据给定上下文中自信息值的分布情况保留信息量最大的内容。在图 2 中，以短语为例，将 p 设为 50，即过滤掉一半的短语。在这种情况下，经过选择性上下文处理后的上下文只剩下 57.2% 的 token，节省了 42.7% 的上下文长度。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Unlocking%20Context%20Constrainits%20of%20LLMs%20Enhancing%20Context%20Efficiency%20of%20LLMs%20with%20Self-Information-Based%20Content%20Filtering/Figure%202.png)

> 图 2. 基于自信息的内容过滤器可视化。该段来自一篇最新论文。

# 总结

在本文中，作者引入了**“选择性上下文”（Selective Context）技术，以最大限度地发挥 LLM 中固定上下文长度的效用。通过过滤掉信息量较少的内容，为 LLM 提供了一种更紧凑、更高效的上下文表示法，同时又不影响它们在各种任务中的性能**，从而证明了该方法的有效性。作者在 arxiv 论文、BBC 新闻报道和对话记录上进行的广泛评估表明，选择性上下文可以显著提高 LLM 的效率，使它们能够更有效地处理长文档和扩展对话。
