---
title: >-
  论文阅读：RCoT Detecting and Rectifying Factual Inconsistency in Reasoning by
  Reversing Chain-of-Thought
date: 2024-05-17 21:37:04
cover: https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/RCoT%EF%BC%9ADetecting%20and%20Rectifying%20Factual%20Inconsistency%20in%20Reasoning%20by%20Reversing%20Chain-of-Thought/Figure%204.png
mathjax: true
tags:
- prompt 工程
category:
- prompt 工程
- CoT 家族
---

大语言模型（LLMs）通过结合逐步思维链（step-by-step CoT）提示，在算术推理任务中取得了可喜的成绩。然而，**大语言模型在推理过程中保持事实一致性方面面临挑战，在特定问题上表现出条件忽略、问题曲解和条件幻觉的倾向**。现有方法使用粗粒度反馈（如答案是否正确）来提高事实一致性。在这项工作中，作者提出了 **RCOT（Reverseing Chain-of-Thought），一种通过自动检测和纠正 LLM 生成的解决方案中的事实不一致性来提高 LLM 推理能力的新方法**。为了检测事实不一致，RCOT 首先要求 LLM 根据生成的解决方案重建问题。然后，对原始问题和重构问题进行细粒度比较，以发现原始解决方案中的事实不一致之处。为了纠正解决方案，RCoT 将检测到的事实不一致转化为细粒度反馈，以指导 LLM 修订解决方案。

实验结果表明，在七个算术数据集上，RCoT 比标准 CoT 有持续的改进。此外，作者还发现，人工编写的细粒度反馈可以显著提高 LLM 的推理能力（例如，ChatGPT 在 GSM8K 上的准确率达到 94.6%），从而鼓励社区进一步探索细粒度反馈生成方法。

# RCoT 方法
作者提出了用于检测和纠正 CoT 中事实不一致（即条件幻觉、忽略和问题误解）的 RCoT，以提高 LLM 的推理能力。具体来说，给定一个复杂的推理问题 Q 和由 LLM 生成的原始解决方案 c。

首先要求 LLM 检测事实不一致：
- **问题重构**：根据生成的解决方案 c 重构问题 Q。
- **细粒度比较**：对原始问题 Q 和重构后的问题$\hat{Q}$进行细粒度比较，以检测条件幻觉、忽略和问题误解。

然后，利用检测到的事实不一致来修正 LLM：
- **细粒度反馈和修订**：细粒度比较揭示了原始解决方案中的事实不一致之处。检测到的事实不一致会形成细粒度反馈，以指导 LLM 据此修改其解决方案。

图 4 展示了作者建议的整体示意图，附录 A.3 则展示了 RCoT 的示例。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/RCoT%EF%BC%9ADetecting%20and%20Rectifying%20Factual%20Inconsistency%20in%20Reasoning%20by%20Reversing%20Chain-of-Thought/Figure%204.png)

> 图 4：RCoT 框架。(1) **重构**：要求 LLM 根据原始解法重建问题，并提供指导和演示示例。(2) **分解**：将原始问题和重构问题分解为细粒度条件列表。(3) **比较**：比较两个子条件表和问题表，核实是否存在幻觉、遗漏和误解。(4) **修订**：将所有与事实不符的地方汇集成细粒度反馈，指导 LLM 修订解决方案。

## 步骤 1：问题重构
直观地说，如果生成的算术问题分布解在逻辑和事实上都是正确和完整的，那么人类就更有可能推断出问题的原貌。同样，作者要求 LLM 根据自己的解法 c 重构问题，得到 $\hat{Q}$，以验证它是否真正理解问题。作者手动编写指令和上下文示例作为重构提示。

作者发现，通过比较重构后的问题 $\hat{Q}$ 和原始问题Q（第 3.2 节），可以有效地揭示条件幻觉（例如，LLM 使用了问题 Q 中没有提及的条件）、条件忽略（例如，LLM 忽略了问题 Q 中的一些重要条件）和问题误解（例如，LLM 误解了问题 Q）等事实不一致的情况，分别如附录 A.1 中的图 9、12 和 17 所示（读者可自行阅读论文）。prompt 模板见图 5。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/RCoT%EF%BC%9ADetecting%20and%20Rectifying%20Factual%20Inconsistency%20in%20Reasoning%20by%20Reversing%20Chain-of-Thought/Figure%205.png)

> 图 5：RCoT 中使用的所有 prompts。

## 步骤 2：精细比较
为了检测条件幻觉和忽略，以及重构问题 $\hat{Q}$ 的解 c 中的问题误解，一种 naive 的方法是要求 LLM 直接比较 Q 和 $\hat{Q}$。然而，这种比较通常无法产生高质量的检测结果（图 6），这并不奇怪，因为 Q 和 $\hat{Q}$ 包含丰富的信息，粗粒度比较不可避免地会忽略一些重要信息，从而导致次优结果。因此，作者采用细粒度的逐步比较来提高检测质量。所有 prompt 模板如图 5 所示。具体过程如下：

- **问题（Problem）分解**：Q 和 $\hat{Q}$ 都是非结构化文本，很难有条理地进行比较。为了解决这个问题，要求 LLM 将问题分解成一个条件列表 $L_Q = [L_Q^1, \ldots, L_Q^m], L_{\hat{Q}} = [L_{\hat{Q}}^1, \ldots, L_{\hat{Q}}^n]$。结构化条件列表将用于细粒度比较。
- **条件比较**：为了找出 Q 和 $\hat{Q}$ 之间的差异，首先要检查它们的条件列表 $L_Q$ 和 $L_{\hat{Q}}$ 是否相同。具体来说，LLM 需要回答每个 $L_Q^i$ 是否都能从 $L_{\hat{Q}}$ 中推断出来。如果不能从 $L_{\hat{Q}}$ 中推断出 $L_Q^i$，那么 $L_Q^i$ 要么 (1) 在解决方案中被忽略了，要么 (2) 被 LLM 幻觉成了另一个条件。同理，如果 $L_{\hat{Q}}^j$ 不能从 $L_Q$ 中推断出来，那么 $L_{\hat{Q}}^j$ 就是幻觉。显然，我们总共需要进行 nm 次比较。
- **问题（Question）比较**：LLM 有时也会误解问题（图 2）。因此，作者也要求 LLM 比较 Q 和 $\hat{Q}$ 中的问题。如果 LLM 发现两个问题不同，那么 LLM 就会在其解决方案中误解问题。这种比较只需进行一次，因为在大多数情况下，每个问题（question）只有一个问题（problem）。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/RCoT%EF%BC%9ADetecting%20and%20Rectifying%20Factual%20Inconsistency%20in%20Reasoning%20by%20Reversing%20Chain-of-Thought/Figure%206.png)

> 图 6：这是一个显示粗粒度比较失败的例子。红色：原始问题与重建问题之间的不一致情况。

经过这些比较，我们可以发现幻觉条件、被忽视的条件和被误解的问题。然后，利用这些信息制定精细反馈，指导 LLM 修改其解决方案。

## 步骤 3：精细反馈和修订
如果我们通过细粒度比较没有发现任何与事实不符之处，我们就认为原始解决方案是正确的。相反，如果检测到任何与事实不符之处，我们会制定细粒度反馈来指导 LLM 修改其解决方案。具体来说，细粒度反馈将首先说明解决方案不正确，然后列出检测到的事实不一致之处，最后要求 LLM 修改其解决方案。图 5 显示了我们用来制定反馈的模板。我们将修改后的解决方案的答案作为最终的评估输出。

# 实验部分
## RCoT 有利于算术推理
表 1 显示了 RCoT 在七个算术数据集上的结果。在 zero-shot CoT 中，RCoT 方法始终优于标准 CoT 和双重检查方法。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/RCoT%EF%BC%9ADetecting%20and%20Rectifying%20Factual%20Inconsistency%20in%20Reasoning%20by%20Reversing%20Chain-of-Thought/Table%201.png)

> 表 1：七个算术推理数据集的平均准确率和标准偏差。粗体表示最佳结果。绿色：与标准 CoT 和 Active-Prompting 相比，分别在 zero-shot 和 few-shot 设置下的性能提升。“*”表示使用手动-CoT 的 LLM。“-”表示 Active-Prompting 的源代码不支持该数据集。

- 在推理步骤复杂、更具挑战性的任务中，RCoT 方法能让 LLM 受益更多。例如，AQuA 包含多种问题；Date 需要多跳推理和常识性日期知识。因此，这些任务对 LLMs 来说很难完成（七项任务中准确率最低的分别为 51.3% 和 66.7%）。实验表明，RCoT 方法可以帮助 LLM 在 AQuA 和 Date 上分别提高 4.1% 和 5.0%，是所有七项任务中提高幅度最大的。
- 对于较简单的任务也依然有效。例如，RCoT 使 SVAMP 数据集的性能提高了 2.8%，该数据集包含的问题通常只需要一步计算。在 SVAMP 上的性能提升低于 AQuA 和 Date，这可能是因为 RCoT 方法善于发现复杂推理中的事实不一致，而 SVAMP 只需要简单的一步推理。

在 few-shot 设置下，上述结论仍然成立。虽然选择最不确定的问题作为 LLM 的演示示例有助于推理，但 RCoT 仍能提高准确性。值得注意的是，双重检查方法在 few-shot 设置中的性能大幅下降。在 AQuA 和 GSM8K 数据集上，它的性能分别下降了 27.0% 和 4.0%，这表明 few-shot 示例可能会增加将正确解修改为错误解的风险。

## 精细反馈对解决方案的修订至关重要
该方法的成功来自于细粒度反馈，它能指出详细的事实不一致之处（条件幻觉和忽略，以及问题曲解）。在本节中，作者将证明粗粒度反馈会导致更差的成绩，从而证明细粒度反馈的必要性。作者用两种粗粒度反馈来取代细粒度反馈：

- 无理由反馈：不告诉 LLM 通过 RCoT 检测到的事实不一致，只给出高级别的判断。因此，如果 RCoT 没有检测到与事实不符的情况，就将原始解决方案作为最终输出进行评估。否则，会提示“Your answer is wrong. You should double-check your answer”，以指导 LLM 修改答案。
- 无判断+理由（即双重检查）： 进一步去除提示中的高级判断。因此，无论 RCoT 的检测结果如何，都会使用“You should double-check your answer”来指导 LLM 修改答案。

表 2 显示了 SVAMP（简单）、GSM8K（中等）和 AQuA（困难）数据集的结果。可以看到，当去除检测到的事实不一致而只保留高级判断时，性能会持续下降，这表明细粒度反馈的有效性。此外，还可以发现，进一步去除判断会使性能比标准 CoT 更差。这并不奇怪，因为 LLM 可能会错误地将正确的解修改为错误的解。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/RCoT%EF%BC%9ADetecting%20and%20Rectifying%20Factual%20Inconsistency%20in%20Reasoning%20by%20Reversing%20Chain-of-Thought/Table%202.png)

> 表 2：使用细粒度反馈和粗粒度反馈的 RCoT 性能。w/o reasons：从原来的细粒度反馈中删除对具体错误的解释。prompt 变为“Your answer is wrong. You should double-check your answer.”。w/o judgment + reasons：进一步删除高级判断，prompt 变为“You should double-check your answer.”。红色：与 RCoT 方法相比，性能下降。

为了进一步展示细粒度反馈的威力，作者进行了人工评估。具体地说，通过生成的解决方案，亲自撰写关于事实不一致的细粒度反馈。令人惊讶的是，在 GSM8K 数据集上，LLM 的准确率达到了 94.6%，而如果从反馈中删除对事实不一致的解释（即与表 2 中“w/o reasons”相同的设置），准确率只能达到 86.3%。附录 A.2 显示了人工撰写和 RCoT 生成的反馈示例。这一结果显示了与表 2 相同的观察结果，并揭示了细粒度反馈的强大威力。由于 RCoT 与人类相比仍有差距（准确率差距为 12.6%），作者鼓励社区进一步探索细粒度反馈的生成。

## 精细比较带来精细反馈
为了获得细粒度反馈，在 RCoT 中要对条件和问题进行细粒度比较。更简单的方法是让 LLM 通过比较原始问题和重构问题直接生成细粒度反馈。表 3 显示，粗粒度比较会导致准确率大幅下降（甚至比标准 CoT 更差），这表明**粗粒度无法生成高质量的反馈**（图 6）。因此，问题分解和细粒度比较至关重要。

作者还表明，条件比较和问题比较都很重要。去掉其中任何一项都会导致性能下降。这是因为 LLM 可能会产生幻觉/忽略条件以及误解问题。
# 总结
本文提出的 RCoT 是一种能够让 LLM 自动检测和纠正事实不一致的方法，以提高 LLM 的推理能力。**RCoT 通过对重构问题和原始问题进行细粒度比较来检测事实不一致，然后通过细粒度反馈要求 LLMs 纠正不一致**。在七个算术推理数据集上的实验结果证明了 RCoT 的有效性。实验还显示，在人工编写的细粒度反馈的帮助下，LLM 的推理能力得到了令人鼓舞的提高，这鼓励了社区进一步探索细粒度反馈的生成。原则上，RCoT 可以应用于其他需要 CoT 解决方案的任务。

## 限制
RCoT 无法检测到所有可能的推理错误。例如，RCoT 很难发现计算错误。不过，RCoT 可以与其他 prompt 技术相结合，例如 Program-of-Thought，这是一种通过将推理与计算分离来减少计算错误的方法。此外，利用 RCoT 生成的反馈修改解决方案与人工反馈之间仍存在很大差距，这促使作者进一步探索如何生成更高质量的细粒度反馈。RCoT 需要与 LLM 进行多次对话（例如论文中的 ChatGPT），因此可能会因 API 调用的低带宽而降低推理速度。不过，本地部署的模型可能会缓解这一问题。
# 实际使用
看图 4 可知，作者是在 OpenAI 的模型上使用的 RCoT 方法，然而在开源模型，例如使用 ollama 部署的 qwen:7b 4bit 模型来说，中间环节都无法得到正确的结果，冗长的步骤反而会引入更多的错误。但该方法提到的问题分解还是有作用的，例如这样的数学题：

- **问题**：一个新程序在第一个月有 60 次下载。第二个月的下载量是第一个月的三倍，但第三个月的下载量减少了 30%。该程序在这三个月中总共有多少次下载？
- **回答**：第二个月，程序的下载次数增加到 3 * 60 = 180，在第三个月中，程序的下载次数为 180 * (1 - 30 / 100) = 126。在这三个月中，程序的总下载次数为 60 + 180 + 126 = 366\n#### 366。

```Python
rcot_decomposition_prompt = """
{{question}}
请列出问题的条件。不要列出与计算无关的条件，但要列出所有必要条件。不要回答问题。
格式应为
条件：
这是您的条件输出，每行一个条件。
"""

resp = ollama.generate(model="qwen:7b", prompt=rcot_decomposition_prompt.replace("{{question}}", question))
print(resp["response"])
"""
条件：
1. 第一个月的下载次数：60次
2. 第二个月下载量是第一个月的三倍：60 * 3 = 180次
3. 第三个月下载量减少了 30%：180 * (1 - 0.3)) = 180 * 0.7 = 126次

总下载次数：60 + 180 + 126 = 446次
"""
```

qwen:7b 4bit 版本无法准确遵循指令，非要去执行 {{question}} 内的指令。但调用在线版本的通义千问可以得到期待的结果。

```text
条件：
- 第一个月下载量：60次
- 第二个月下载量：第一个月的三倍
- 第三个月下载量：第二个月的70%（因为减少了30%）
```

我们可以让模型输出中间结果，然后判断这些中间结果是否正确，如果存在不正确的条件，则将其过滤掉，避免其影响最终的决策。
