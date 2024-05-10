---
title: >-
  论文阅读：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
  Checkpoints
date: 2024-03-28 16:33:37
mathjax: true
cover: https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints/Figure%202.png
tags:
- 论文阅读
category:
- 模型架构
- 注意力机制
---

论文链接：https://arxiv.org/abs/2305.13245

自回归解码器推理是 Transformer 模型的一个严重瓶颈，虽然现在的大模型（LLM）会通过 kv cache 存储历史 kv 信息，避免每个解码步骤重复计算，但这需要额外的内存空间，并且加大了内存带宽的开销（需要将 kv cache 从显存中加载到 GPU 的 SP）。因此，在 decoder-only 模型的推理过程中，pre-fill 阶段是 compute-bound，瓶颈在于 GPU 的计算，需要将输入的 prompt 计算 kv 信息；decode 阶段则是 memory-bound，瓶颈在于加载 kv cache，并且随着 context 的增加，解码的速度越慢。

因此，就有研究人员考虑是否可以不再是一个 query 头对应一个 key 和 value 头的 multi-head attention？于是提出了 MQA，多个 query 对应一个 key 和 value 头。我们知道 self-attention 中的主要参数权重是 Q、K、V 和 O 这四个矩阵，其中 K 和 V 的头从原先的 32（Llama3-8B）减少至 1 后，可以减少训练所需的参数量以及推理时加载的 kv cache，这对于模型训练亦或是推理都能节省大量的时间与成本。

但直接将 key 和 value 的头从 32 减少至 1，不可避免会损失信息（类似于降维），从而影响模型的性能。本篇论文在附录 A 的训练稳定性中也讲述了 MQA 预训练时的不稳定：

> 作者发现，MQA 会导致训练不稳定，尤其是与长输入任务相结合时。作者从头开始用 MQA 训练了多个 T5-Large 模型。在每种情况下，预训练都会出现频繁的损失峰值，最终模型在长输入任务中会立即出现偏差。因此，对于不稳定任务上的 multi-query 模型，报告的是多次运行的平均性能。不过，经过 up-training 的 GQA 模型似乎比较稳定。

我们能否将 MHA（一对一）和 MQA（all 对 1）取个折中呢？既能不损失过多的信息，同时还能减少 key 和 value 的头数？一个朴素的想法是多（query）对一（key 和 value），这就是本篇论文提出的 GQA，多组查询注意力。

# 分组查询注意力（GQA）
GQA 将 query 头分为 G 组，每组共享一个 key 和 value 头。GQA-G 指的是有 G 个组的分组 query。

- GQA-1 只有一个组，因此也只有 1 个 key和 value 头，相当于 MQA；
- GQA-H 的组数等于 query 头数，相当于 MHA。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints/Figure%202.png)

> 图 2：GQA 方法概述。MHA 有 H 个 query、key 和 value 头。MQA 在所有 query 头中共享单个 key 和 value 头。GQA 则是为每组 query 头共享单个 key 和 value 头，介于 MHA 和 MQA 之间。

图 2 显示了分组查询注意力和多头/多查询注意力的比较。从 MHA 到 MQA，可以将 H 个 kv 头减少到单个 kv 头，从而将 kv cache 的大小以及需要加载的数据量减少 H 倍，但模型训练的稳定性以及效果会有所下降。GQA 的 kv 分组会让模型的质量比 MQA 高，速度比 MHA 快，这是一种有利的权衡。

较大的模型通常会按比例增加头的数量，MQA 对此不敏感（无论 query 头怎么增加，key 和 value 头永远是 1）代表了对内存带宽和容量的更大削减。GQA 可以在模型规模增大时，保持相同比例的带宽和容量缩减，这能为大模型提供特别好的权衡（PS：Llama2 只有 70B 模型是 GQA，而 Llama3 已经全部使用 GQA）。

# Uptraining
从 MHA 模型转换到 MQA 模型分为两个步骤：首先是转换检查点，其次是额外的预训练，以使模型适应新的结构。图 1 显示了将 MHA 检查点转换为 MQA 检查点的过程。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints/Figure%201.png)

> 图 1：从 multi-head 转换为 multi-query 注意力的概述。来自所有 head 的 key 和 value 投影矩阵被平均池化到一个 head。

在将 multi-head checkpoint 转换为 MQA checkpoint 时，将 multi-head kv 头的投影矩阵平均池化成单个头的投影矩阵，作者发现这比选择单个 kv 头（从已有的 multi-head 中选择）或从头开始随机初始化新的 kv 头效果更好。然后，在相同的预训练配方上，对转换后的 checkpoint 进行原始训练步骤的一小部分 α 的预训练。通过持续预训练（continual pretrain）来让模型在原始的一小部分预训练数据集上适应新的结构。context 扩展也是相同的做法，我们往往会基于已开源的预训练模型，例如 llama、mistral 等 base model 上做持续预训练，通过 NTK-aware RoPE 等手段调整模型支持的最大上下文长度。为了让模型更好地适应调整后的 context length，我们会继续预训练或者 SFT 来让模型适应。当你扩展新的知识时，最好是用实践来验证下理论，模型同样也是如此。

上文讲述了从 MHA 转换为 MQA，同理可得，在将 MHA checkpoint 转换为 GQA checkpoint 时，通过平均池化该组中的所有 kv 头来构建每个组的 kv 头。

# 实验部分

## 主要结果
图 3 显示了 MHA T5-Large 和 T5-XXL 模型以及上训练比例为 α = 0.05 的 MQA 和 GQA-8 XXL 模型在所有数据集上的平均性能与平均推理时间的函数关系。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints/Figure%203.png)

> 图 3：经过 up-training 的 MQA 和 MHA 相比具有更高的质量和更快的速度，与 MHA-Large 相比具有更好的折衷效果，而 GQA 则具有更好的性能，速度提升与 MQA-XXL 相似，质量与 MHA-XXL 相当。

作者发现，与 MHA 模型相比，较大的 up-training MQA 模型提供了有利的权衡，与 MHA-Large 相比，它的质量更高，推理速度更快。此外，GQA 还能显著提高质量，性能接近 MHA-XXL，速度接近 MQA。表 1 包含了所有数据集的全部结果。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints/Table%201.png)

> 表 1：在摘要数据集 CNN/Daily Mail、arXiv、PubMed、MediaSum 和 MultiNews、翻译数据集 WMT 以及答题数据集 TriviaQA 上，使用 MHA 的 T5 Large 模型和 XXL 模型，以及使用 MQA 和 GQA 的 5% 的 up-training 的 T5-XXL 模型的推理时间和平均验证集性能比较。

## checkpoint 转换实验
![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints/Figure%204.png)

> 图 4：在α = 0.05 的情况下，不同 checkpoint 转换方法对 T5-Large uptrain 到 MQA 的性能比较。Mean mean-pools 分配 key 和value 头；First 选择第一个 head；Random 初始化所有 heads。

图 4 比较了不同 checkpoint 转换方法的性能。mean pooling 似乎效果最好，其次是选择单个 head，最后是随机初始化。直观地说，结果是根据预训练模型信息的保留程度排序的。

# 总结
语言模型的推理成本很高，这主要是由于加载 key 和 value 所带来的内存带宽开销。MQA 降低了这种开销，但代价是降低了模型的容量和质量。作者建议将 MHA 模型转换为 MQA 模型，只需原来预训练计算量的一小部分。此外，还引入了 **GQA，它是 MQA 和 MHA 的插值，能以与 MQA 相当的速度达到接近 MHA 的质量**。

# QA 相关
论文中提到 GQA 可以加速推理，那么到底加速了哪一部分呢？
- **训练阶段**：GQA 可以减少 self-attention 块的参数量，减少训练的参数量，从而加快训练。例如，MHA 的 self-attention 块需要训练$W_Q$、$W_K$、$W_V$ 和 $W_O$，这 4 个权重矩阵的形状为 [h, h]，因此总参数量为 $4h^2 + 4h$。使用 GQA 后，K 和 V 的注意力头数会减少，例如为原先的 h/8，那么总参数量就变为 $\frac{9}{4}h^2 + \frac{9}{4}h$。

    > **猜测**：Llama2-70B 中采用 GQA，我理解是为了减少参数量，加快训练速度，同时 GQA 并不会减少太多的性能，尤其对于 70B 模型来说，这点性能损失与模型规模相比完全可以不用考虑，训练速度更为重要。
- **推理阶段**：$W_K$ 和 $W_V$ 的参数量较少，但为了和 Q 有相同的形状来进行矩阵乘法，会调用 `repeat_kv` 函数拷贝至与 Q 相同的形状，也就是说 $QK^T$ 的计算与 MHA 没有变化。区别就是在计算 `key_states` 和 `value_states` 时的维度较少以及新增了拷贝的操作。
	```Python
	query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
	key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
	value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
	
	past_key_value = getattr(self, "past_key_value", past_key_value)
	cos, sin = self.rotary_emb(value_states, position_ids)
	query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
	
	key_states = repeat_kv(key_states, self.num_key_value_groups)
	value_states = repeat_kv(value_states, self.num_key_value_groups)
	```
	
	GQA 真正起到推理加速的作用是：
	- **减少了从显存（HBM 或 SRAM）中读取的 kv cache 数据量**：缓解 memory-bound（计算单元处理的速度快于内存单元，可以理解为传送带的速度比较慢，而车间加工的速度非常快，车间工人们忙完手中的活后，需要等传输带的原材料到才能继续工作），减少计算单元等待的时间，提高了计算利用率。
    - **减少了参数量以及 kv cache 占用的显存**：空出来的显存用来增加 LLM serving 的 batch size，从而增加服务的总吞吐量。

当 context 较短时，例如输入长度 + 输出长度 = 3500 + 500 = 4k，且模型规模较小，例如 Llama2-7B，当 batch size = 1，float16 数据格式存储，其 kv cache 总显存占用 $4bshl = 4 \times 1 \times 4000 \times 4096 \times 32 = 2,097,152,000 \approx 1.95GB$。A100 HBM 的读取速度有 1.5 TB/s，即使 Mistral-7B 使用 GQA 减少了 kv cache 的数据量和显存占用，但这部分节省的操作耗时，对于总耗时来说几乎不会有太大的影响。

![在这里插入图片描述](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/GQA%20Training%20Generalized%20Multi-Query%20Transformer%20Models%20from%20Multi-Head%20Checkpoints/benchmark_latency.png)
上图使用 vLLM 自带的 `benchmark_latency.py` 脚本测试。