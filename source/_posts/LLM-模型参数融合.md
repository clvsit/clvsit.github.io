---
title: LLM 模型参数融合
date: 2024-04-05 23:53:10
tags:
category:
- LLM
- 模型参数融合
---


模型参数融合通常指的是**在训练过程中或训练完成后将不同模型的参数以某种方式结合起来，以期望得到更好的性能**。这种融合可以在不同的层面上进行，例如在神经网络的不同层之间，或者是在完全不同的模型之间。模型参数融合的目的是结合不同模型的优点，减少过拟合的风险，并提高模型的泛化能力。在实际应用中，这通常需要大量的实验来找到最佳的融合策略。

> 本篇文章只介绍训练完成后的不同模型的参数融合，不涉及训练过程的模型参数融合。

# 思路来源

去年年底曾与 chatglm 的算法团队有过交流，一个令人印象深刻的论述是**大模型的参数空间**非常稀疏，当时 glm-130B 模型刚开源，用 3 张 RTX3090 以 int4 方式部署，推理的效果虽然相较 chatgpt 甚远，但比起 T5 也好得多，经过业务数据微调后即可投入到实际的生产业务，而非传统 NLP 的规则、模板 + 模型的生产方式。

按照 LIMA 的观点，模型的知识是在预训练期间注入，后续的 SFT 和 RLHF 是“约束”模型的输出，使其符合人类的喜好与习惯，或者学会一种表达方式。那么，错误的对齐会让模型的能力遭到破坏，训练数据的分布越是“离谱”，训练时的 Loss 越大，对模型的底层（离输入较近的层）影响也越大，容易出现模型塌陷，即“训崩了”。LoRA 类方法相比 full fine-tuning 的一个优点就是其稳定性，**维持住模型的基本盘就能够取得不错的结果**。

综上所述，大模型参数空间非常稀疏 + 浅层表征假说（SFT 影响浅层以及 LoRA 冻结骨干网络参数，通过微调 adapter 的方式来适配各种风格），**不同任务的微调或许仅仅只是修改了庞大参数空间的一隅，但这些任务数据之间高度的独立同分布**，它们各自在各自的参数空间内“各司其职、互不干扰”，就像九头蛇一样，共享同一个身体，通过不同任务的微调，使其长出一个新的头。

> 模型量化，例如 int4、int8 如果对于模型输出的效果影响不大，也可以理解为参数的略微变化并不会对结果影响太大。

## 方法优点

1. **无需训练**，只需要将现有的基于相同基底的模型进行融合即可。
2. 针对单独一个领域训练“偏科”的模型要比训练通用模型要容易得多，不需要考虑数据集内部各类型数据的配比情况，也不需要考虑数据顺序和采样，训练的过程也容易得多，甚至过拟合也未尝不可。
3. “查漏补缺”，哪里不行补哪里。

## 方法缺点

不一定有用（滑稽.jpg）。

## DARE

阿里提出了一种名为 DARE 的方法，用来将具备不同能力的多个模型融合成拥有全部能力的单个模型。

https://zhuanlan.zhihu.com/p/668152236

**论文地址**：[https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2311.03099](https://link.zhihu.com/?target=https://arxiv.org/abs/2311.03099)

作者发现基于编码器或解码器的语言模型可以通过**吸收同源模型的参数来获得新的能力，而无需重新训练**。通常，LMs 的新能力可以通过 SFT 实现，这反映在微调后模型参数与预训练参数（即 delta 参数）之间的差距上。作者提出 DARE（Drop And REscale）方法，将大部分的 delta 参数设置为 0，这并不会影响 SFT LM 的能力，并且越大的模型的可以 drop 更多的参数。基于这一观察结果，使用 DARE 进一步稀疏多个 SFT 同源模型的 delta 参数，然后通过参数平均将它们合并为一个模型。

### 使用方式

#### 官方代码

https://github.com/yule-BUAA/MergeLM/tree/main

> **注意实现**：在调用 `merge_llms_instruct_math_code.py` 脚本执行模型参数融合时，无需进行融合后模型指标的评测（这需要下载评测数据集，评测速度很慢，完全没有必要），可以将 `get_merge_performance()` 函数内的 `test_alpaca_eval()` 等评测函数相关的代码区域全部注释（见下图）。

  ![](https://secure2.wostatic.cn/static/YHX2r7Y4cPwuK7K2Nc2Ca/image.png?auth_key=1712332372-5FzefMFTrqqjcWNovXsbyC-0-ecdfb141d8ae1346fae7681ee6f0cdb9)

  最后，将函数尾部的删除模型的代码注释。

  ![](https://fuhgh5u28j.feishu.cn/space/api/box/stream/download/asynccode/?code=YzdhYTA2M2IwMzA2MTQ2ZjQ1N2Q1MzYyYTg2MjBkMzJfYTZXeWxmVndYM0xHMFZFMG1rZmE5R3VCMVVUTEtvbHFfVG9rZW46WEFjRmJwOTA1b05XRmd4TTRVUmNOQzRKbmJiXzE3MDU0NzUyMTU6MTcwNTQ3ODgxNV9WNA)

### 实验结果

|||||||||||
|-|-|-|-|-|-|-|-|-|-|
|模型名称|mt_bench 总分|writing|roleplay|reasoning|math|coding|extraction|stem|humanities|
|GPT-3.5-turbo|7.944|9.200|8.400|5.650|6.300|6.900|8.850|8.700|9.550|
|融合刘琳训练的 ep_04 模型||||||||||
|ep_04_merge_v1|7.834|9.275|8.175|6.500|6.025|5.850|8.000|9.000|9.850|
|ep_04|7.257|8.553|7.737|6.250|4.300|5.750|7.342|8.425|9.700|
|基础实验：在 mistral-7B 模型上验证参数融合是否有效||||||||||
|mistral-7b-dare-merge-v1|6.991|7.900|8.150|5.900|5.125|4.675|6.900|7.875|9.400|
|mistral-7b-instruct-v0.1|6.772|8.250|7.450|6.150|4.150|4.300|6.600|7.675|9.600|
|mistral-7b-math|4.456|3.725|4.900|4.100|5.050|2.350|4.350|5.275|5.900|


首先，是在 mistral-7B 模型上进行模型参数融合实验，验证 DARE 方法是否有效，将基于 mistral-7B 微调的 mistral-7b-instruct-v0.1 和数学能力较强的 mistral-7b-math 进行融合。观察 mt_bench 各项指标，融合后的模型的 math 能力得到了极大的提升，说明 DARE 方法切实有效。

因此，将目前最强的自研模型 ep_04（刘琳训练）和当前最强的基于 mistral-7b 微调的数学模型进行融合，最后将 mt_bench 从 7.257 提升至 7.834，超过 openchat-3.5。

## mergekit

### 使用方式

#### 官方代码

https://github.com/cg123/mergekit



使用 mergekit 融合多个模型，作为 MoE。

```YAML
base_model: mlabonne/Marcoro14-7B-slerp
experts:
  - source_model: openchat/openchat-3.5-1210
    positive_prompts:
    - "chat"
    - "assistant"
    - "tell me"
    - "explain"
  - source_model: beowolx/CodeNinja-1.0-OpenChat-7B
    positive_prompts:
    - "code"
    - "python"
    - "javascript"
    - "programming"
    - "algorithm"
  - source_model: maywell/PiVoT-0.1-Starling-LM-RP
    positive_prompts:
    - "storywriting"
    - "write"
    - "scene"
    - "story"
    - "character"
  - source_model: WizardLM/WizardMath-7B-V1.1
    positive_prompts:
    - "reason"
    - "math"
    - "mathematics"
    - "solve"
    - "count"
```

https://huggingface.co/mlabonne/Beyonder-4x7B-v2
