---
title: >-
  论文阅读：Self-Evolved Diverse Data Sampling for Efficient Instruction Tuning
  数据子集挑选方法
date: 2024-05-12 23:31:34
cover: https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Self-Evolved%20Diverse%20Data%20Sampling%20for%20Efficient%20Instruction%20Tuning/Figure%201.png
mathjax: true
tags:
- 论文阅读
- 数据子集挑选
category:
- LLM
- 数据增强
- 数据子集挑选
---

论文链接：https://arxiv.org/abs/2202.06417

GitHub 仓库：https://github.com/OFA-Sys/DiverseEvol

提高大型语言模型（LLM）的指令遵循能力主要需要大量的指令调整数据集。然而，这些数据集的庞大数量带来了相当大的计算负担和标注成本。为了研究**一种标注效率高的指令调整方法，使模型本身能够主动采样同样有效甚至更有效的子集，作者引入了一种自进化机制 DIVERSEEVOL**。在这一过程中，**模型会反复增强其训练子集，以完善自身性能，而无需人类或更高级 LLM 的干预**。**该数据采样技术的关键在于提高所选子集的多样性，因为模型会根据其当前的嵌入空间选择与任何现有数据点最不同的新数据点**。三个数据集和基准的广泛实验证明了 DIVERSEEVOL 的有效性。与在全部数据上进行微调相比，在不到 8% 的原始数据集上训练的模型保持或提高了性能。作者还提供了经验证据来分析指令数据多样性的重要性，以及迭代方案相对于一次性采样的重要性。

# 方法：DIVERSEEVOL

## 迭代指令数据选择
目标是将指令数据挖掘正规化，使其成为一个迭代过程，按照一定的策略从庞大的源指令数据集中逐步提取指令。给定一个指令响应对集合，表示为 $Z = \{(x_i, y_i)\}_{i \in N}$ ，其中每个 (xi, yi) 代表一个特定的指令响应对，定义 $N = \{1, \ldots, n\}$ 为初始源代码指令数据集的大小。

迭代过程围绕两个数据容器展开：截至迭代步骤 t 的训练数据池 $P_t$ 和未选择数据点容器 $Q_t$。在每次迭代 t 中，选择函数（即策略）A 决定将哪些数据点 $S = \{s_j\}_{j \in K}, \ K = \{1, \ldots , k\}$，被整合到下一步的训练数据池 $P_{t+1}$中。扩大后的模型库将作为下一次模型迭代 $M_{t+1}$ 的训练集。

从随机数据池 $P_0$ 开始训练初始模型 $M_0$，之后的每一步都利用模型$M_t$、当前训练池$P_t$和综合数据集 Z 为函数 A 提供信息，然后函数 A 输出新的数据点$S_t$，将其添加到下一次迭代$P_{t+1}$的训练池中，如下所示：$S_t = A(Z, P_t, M_t); P_{t + 1} = P_t \cup S_t$。因此，每次迭代包括两个操作：

1. 根据之前训练好的模型$M_t$，提取新的数据点$S_t$并入$P_{t+1}$。
2. 使用更新后的数据池$P_{t+1}$训练后续的聊天模型$M_{t+1}$。

这种方法的有效性取决于选择函数 A，它决定了每次训练迭代的额外k 个数据点。随着P 数量的增加，尤其是多样性的增加（目前主流的数据子集选择算法都会采用 k-center sampling 算法），由此产生的聊天模型会不断完善其能力。

## 选择算法：K-Center-Sampling
DIVERSEEVOL 的核心是基于 K-Center-Sampling 的选择函数 A，详见图 1。所选子集必须能恰当地代表更广泛的数据集，以确保在缩小的子集上训练出来的模型能与在完整数据集上训练出来的模型相媲美。因此，函数 A 致力于积累一个高度多样化的源数据集子集。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Self-Evolved%20Diverse%20Data%20Sampling%20for%20Efficient%20Instruction%20Tuning/Figure%201.png)

> 图 1：迭代 DIVERSEEVOL 概述：从初始训练数据集 P0 和源数据集的剩余数据 Q0 开始，训练一个聊天模型 M0，并将所有数据点投射到其嵌入空间 EM B0 中。在此嵌入空间中利用基于 K-Center 的选择，从 Q0 中选择一组新的数据点 S0 并添加到下一个训练数据池 P1 中，以对下一个聊天模型 M1 进行调试。此过程重复 T 步，仅根据模型本身产生逐步增强的训练数据池，然后用于改进功能更完善的模型。

对于一组给定的训练数据点 $P_t$，函数 A 可识别出新的数据点 $S_t$，这些数据点与 $P_t$ 结合后，可提供源数据集的代表性样本。这就需要选择尽可能不同于任何现有数据点的新添加数据。与现有数据点之间的“差异”是由候选数据点（即 $Q_t$ 中尚未选定的数据点）与 $P_t$ 中任何现有训练数据的最近距离来量化的，换句话说，就是与其最近相邻数据点 $P_t$ 的距离：

**目标**：从候选数据池中选择 k 个数据点，使其与各自最近的现有训练数据点的距离最大化。

$$
max \sum_{i \leq i \leq k} min_{j \in P_t} \Delta(s_i, P_j) \tag{1}
$$

函数旨在将每个已知数据点指定为整个训练池中的唯一中心。因此，它寻求 $S_t$ 中的每个新数据点与 $P_t$ 中的任何现有训练数据点之间的最小距离最大化：

$$
arg \ max_{i \in Q_t} \ min_{j \in P_t} \Delta(s_i, P_j) \tag{2}
$$

由于样本间的距离（用 ∆ 表示）是根据对所有 token 位置进行平均池化后 $M_t$ 的输出隐藏状态计算得出的，这为现有数据提供了一个更合适的嵌入空间，因此当前训练模型 $M_t$ 产生的嵌入为我们的选择提供了指导。因此，根据模型当前的理解，添加到训练集的数据点确保对现有数据集进行最佳补充。这种迭代过程有利于模型的演化，因为它可以吸收之前迭代的经验来完善其性能。

### Python 实现
K-Center Sampling 问题是一个 NP 难问题，但有很多 heuristic 算法可以用来近似求解。下面是一个简单的 Python 实现，使用了简单随机初始化和 K-Means 算法的迭代过程来寻找近似解。

```python
import random
 
def k_center_points(points, k):
    # 随机初始化k个中心点
    centers = random.sample(points, k)
    last_centers = None
 
    # 迭代更新中心点直到稳定
    while centers != last_centers:
        clusters = {center: [] for center in centers}

        for point in points:
            closest_center = min(centers, key=lambda c: sum((c - point) ** 2))
            clusters[closest_center].append(point)
        
        last_centers = centers
        centers = {center: sum(cluster, []) // len(cluster) for center, cluster in clusters.items()}
    
    return centers
 
# 示例使用
points = [(1, 2), (3, 4), (5, 6), (7, 8), (2, 3), (4, 5), (6, 7), (8, 9)]
k = 2
center_points = k_center_points(points, k)
print(center_points)
  ```
 
上述代码首先随机选择 K 个点作为初始中心，然后通过迭代将每个点分配到最近的中心类别，并更新中心点位置。当中心点不再变化时，停止迭代。这个算法可以作为 K-Center Sampling 问题的一个快速近似解。

# 总结与限制
引入了 DIVERSEEVOL，这是一种用于高效调整 LLM 指令的自进化方法。DIVERSEEVOL 依靠迭代方案，利用 K-Center 策略从大量指令数据中选择不同的子集，从而逐步完善自身，而无需寻求任何外部监督。经验结果表明，该方法只用了不到原始数据大小的 8%，就能达到或超过强大的基准性能。未来的工作可以在更大的指令数据集上利用该方法，以获得可能更加精细的结果。在 DIVERSEEVOL 所奠定的基础上，更先进的多样化采样算法也有望进一步提高模型性能。

**限制**：DIVERSEEVOL 中的 K-Center-Sampling 方法需要计算数据点高维嵌入之间的距离。如果源数据集的规模进一步增大，这种计算可能会对 GPU 内存造成相当大的消耗。此外，评估结果在很大程度上依赖于 GPT4-judge。尽管试图通过将查询温度设置为 0 来获得更确定的结果，并通过在交替位置上进行两次模型响应查询来解决位置偏差问题，但评估过程仍可能受到 GPT4 模型内部固有偏差的影响。

# 附录 A：GPT-4 Judge 模板
```text
[Question]
{instruction}

[The Start of Assistant 1's Answer]
{answer-of-chatbot1}
[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer-of-chatbot2}
[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
```