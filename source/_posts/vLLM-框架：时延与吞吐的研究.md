---
title: vLLM 框架：时延与吞吐的研究
date: 2024-01-30 18:40:00
tags:
- vLLM 框架
- 模型部署
category:
- LLM
- 模型部署
---

> 注意事项：该实验是在 2024-01-05 日所作，当时所使用的 vLLM 框架版本是 v0.2.7。

读取 fastapi 部署的模型服务的日志，可以看到统计信息，如下所示：

```Bash
INFO 01-05 01:46:00 llm_engine.py:624] Avg prompt throughput: 416.3 tokens/s, Avg generation throughput: 123.8 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 5.6%, CPU KV cache usage: 0.0%
```

- **Avg prompt throughput**：平均 prompt 吞吐，即 prefill 阶段的吞吐处理能力。
- **Avg generation throughput**：平均生成吞吐，即 generate 阶段的吞吐处理能力。

查看所有的日志信息，发现线上模型服务的 KV cache usage 都相当小，甚至没有 20%，这是不是说明还能再往里面添加更多的 reqs？但根据先前的经验，增加 req 会导致单条 req 的时延增加，这是否可以证明时延与吞吐两者不可兼得？

使用官方提供的 `benchmarks/benchmark_serving.py` 脚本，执行以下所示的命令：

- 启动 fastapi 模型服务：

```Bash
python -m vllm.entrypoints.api_server --model <your_model> --swap-space 16 --disable-log-requests
```
- 启动测评脚本：

```Bash
python benchmarks/benchmark_serving.py --backend vllm --tokenizer <your_model> --dataset <target_dataset> --num-prompts 8
```

**注意事项**：启动 fastapi 模型服务中的 <your_model> 和启动测评脚本中的 <your_model> 是相同的模型，<target_dataset> 使用的是线上 1000 条真实的对话数据。也可以设置 `--num-prompts` 参数来设置总的请求数。

在测评过程中，我们发现 Runing reqs 最大不会超过 256，GPU KV cache usage 最多占用 40%，换言之，显存仍然有非常多的空余，可以容纳更多的请求。于是在代码中进行检索，发现可以设置 `EngineArgs` 的 `max_num_seqs` 属性来调整请求队列的最大数量。

下面，我们将测试在不同 `max_num_seqs` 设置下，KV cache usage、总耗时、吞吐量以及平均时延这四项指标的结果。

|max_num_seqs|KV cache usage|Total time (s)|Throughput (req/s)|Avg latency (s)|
|-|-|-|-|-|
|64|10.4%|99.68|10.03|51.59|
|128|21.3%|93.03|10.75|49.80|
|256（默认）|40.0%|88.05|11.36|52.70|
|512|68.7%|87.71|11.40|60.40|


**评测结果**：

- `max_num_seqs` 增大可以提升服务的吞吐量，并且降低总耗时，但平均时延也会随之增加。
- 当 `max_num_seqs` 较小时（例如 64），跑完所有的测试数据需要更多的队列调度，反而会导致平均时延增加。

下面，我们让调用端和服务端的请求数保持一致，即调用端的并发量与 `max_num_seqs` 相等。

|max_num_seqs|Total time (s)|Throughput (req/s)|Avg latency (s)|
|-|-|-|-|
|64|100.58|9.94|6.01|
|128|91.89|10.88|10.76|
|256（默认）|87.19|11.47|19.81|
|512|86.79|11.52|37.35|


GPU 的处理能力与显存存在较大的差距，即使我们添加足够多的请求，将显存“塞满”，也无法以显著提升吞吐，反而会降低单条请求的时延。好比我们有一个很大的仓库，但搬货的工人人数有限，搬货的速度与仓库的空间无法匹配，很多货物仅仅堆积在仓库内部，并不能被及时地搬运。因此，A100 80G 并不适合用来部署 13B 及以下的模型，会存在显存的浪费。

# Q&A

### 如何根据线上业务来设置合适的 batch size？
根据系统的可用性要求，我们要尽可能保证服务的请求不要超时，最好是能达到 99.9999% 的可用率。那么，我们就需要根据最糟糕情况下，各 batch size 的时延来找出允许的最大 batch size。

例如，线上业务的最大 prompt 长度为 3500，输出长度（`max_new_tokens`）最大为 500，那么我们需要对 `benchmark_serving.py` 脚本进行修改，自行构造长度为 4000 但不同（相同的话，会触发 vLLM PagedAttention 的 copy-of-token 机制）内容的输入 prompt。每次设置不同的 batch size 进行测试，测试线上业务极限场景下的时延与吞吐，找出均衡点。

|||||||
|-|-|-|-|-|-|
|**模型名称**|**最大并发**|**平均时延 (s)**|**最大时延 (s)**|**吞吐 (r/s)**|**kv cache usage (%)**|
|**输入 token**：3500  **输出 token**：490 - 500||||||
|mistral-7B|4|8.86|9.52|0.45||
||8|11.35|12.96|0.70||
||12|13.72|17.13|0.82||
||16|16.23|20.17|0.98||
|MoE 4x7B|2|20.91|21.82|0.10|4.7|
||4|23.18|25.61|0.17|9.4|
||8|27.85|33.29|0.29|18.7|
||12|32.40|41.08|0.34|28.0|

上图是在单张 A100 80G 显卡测试 mistral-7B 和 MoE 4x7B 在输入 token 3500 和输出 490 - 500 tokens 场景下的时延、吞吐以及 GPU kv cache usage。

假设最大允许时延在 20s 以内，可以看到 MoE 4x7B 的显存还有很多的浪费。
