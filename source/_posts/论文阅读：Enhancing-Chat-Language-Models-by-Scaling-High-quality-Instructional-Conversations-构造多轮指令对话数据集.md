---
title: >-
  论文阅读：Enhancing Chat Language Models by Scaling High-quality Instructional
  Conversations 构造多轮指令对话数据集
date: 2023-09-03 23:09:59
cover: https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Enhancing%20Chat%20Language%20Models%20by%20Scaling%20High-quality%20Instructional%20Conversations/Table%207.png
tags:
- 论文阅读
- 数据构造
category:
- LLM
- 数据增强
- 数据构造
---

数据集地址：https://huggingface.co/datasets/stingning/ultrachat
GitHub 仓库：https://github.com/thunlp/UltraChat

对指令数据进行微调已被广泛认为是实施 ChatGPT 等聊天语言模型的有效方法。本文旨在通过构建一个系统设计的、多样化的、信息丰富的大规模指令对话数据集 UltraChat，来提高开源模型性能。UltraChat 包含 150 万条高质量的多轮对话，涵盖广泛的主题和指令。作者在 UltraChat 的基础上，对 LLaMA 模型进行了微调，从而创建了强大的会话模型 UltraLLaMA。评估结果表明，UltraLLaMA 始终优于其他开源模型，包括之前公认的最先进开源模型 Vicuna。

# 方法介绍

与其他倾向于使用特定任务（如问题解答、改写和总结）来构建数据的数据集不同，该方法以一个三方框架为基础，旨在捕捉人类与人工智能助手可能进行的广泛交互。作者认为，人类用户与人工智能助手之间的任何互动都可视为获取信息。
- **信息获取**：第一部分“关于世界的问题”侧重于查询世界上的现有信息。这是人机交互的一个重要方面，因为用户通常依赖于人工智能助手来快速准确地回答他们的问题。通过包含广泛的主题，数据集满足了用户对信息的不同需求，确保人工智能助手能够提供相关和全面的回复。
- **创建条件信息**：第二部分是“创作与写作”，涉及在人类输入条件下创造新信息。这一过程反映了人工智能与用户一起参与创造性任务的能力，利用其丰富的知识和模式识别能力生成原创内容。
- **信息改造**：第三部分“对现有材料的协助”涉及对现有信息的修改。这是人机交互的一个重要方面，因为它可以让人工智能助手主动参与用户的输入，通过改写、续写、总结或推理等各种方式对其进行转换。

如上所述，UltraChat 由三个不同的部分组成，每个部分都面临着独特的挑战。首要原则是使数据尽可能多样化。确保数据多样性的核心是确保开场白和用户回复风格的多样性。
- 开场白直接决定了对话的主题。开场白应高度多样化，包含人类用户可能要求聊天模型执行的任何任务。
- 用户决定对话的情节，输出应根据当前主题量身定制，语言风格和要求应多样化。

## 第一部分：关于世界的问题

主要关注现实世界中存在的概念、对象和实体。收集数据的方法涉及两个视角：
- 以主题和概念为中心：
  1. 最初，要求 ChatGPT 生成 30 个综合话题，这些话题涵盖了日常生活的各个方面。
  2. 随后，深入研究每个话题，生成 30 到 50 个子话题或相关概念。
  3. 最后，为每个子话题或概念生成 10 个不同的问题，并要求 ChatGPT 在每个原始问题的基础上再生成 10 个问题。

- 以现实世界中的实体为中心：这些对象来自维基数据实体。考虑到这些实体在维基百科文章中出现的频率，对它们进行了进一步的细化，尤其是重点关注出现频率最高的 10,000 个实体。对于每个实体，创建 5 个元问题，然后是 10 个更具体的问题和 20 个扩展问题。扩展问题旨在与原始问题保持一定的相似性，同时探索不同的对象或主题。为了创建对话，过滤并抽取了大约 50 万个问题作为开场白。在构建每段对话的过程中，都会为用户模型提供精心制作的提示，明确要求模型根据正在进行的对话历史背景做出简洁而有意义的回应。

## 第二部分：创造和生成

为了创建对话，作者过滤并抽取了大约 50 万个问题作为开场白。在构建每段对话的过程中，都会为用户模型提供精心制作的 prompt，明确要求模型根据正在进行的对话历史背景做出简洁而有意义的回应。这些指令是对话生成的开场白。在整个生成过程中，用户 prompt 会不断强化对话的主要目的，即生成和完善一篇文章。这样做的目的是确保用户模型的行为与预期目的保持一致。

## 第三部分：为现有材料提供协助

与现有文本材料相关的各种任务，如改写、翻译、摘要和问题解答等。首先从 C4 语料库中收集文本片段。C4 语料库中的每篇文章都与源 URL 相关联。为确保文本内容和风格的多样性，采用了上一节中列出的 20 种材料类型，并为每种类型手动设置了关键词。此外，还通过匹配关键词和相应的 URL 对语料库中的文本进行分类。总共从 C4 语料库中收集了 10,000 篇文本，并针对每篇文本提示 ChatGPT 生成五条不同的指令。为了将文本片段与特定指令结合起来，作者使用了人工设计的模板，如表 4 所示。最终，500,000 个片段的组合将作为生成对话的开场白。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Enhancing%20Chat%20Language%20Models%20by%20Scaling%20High-quality%20Instructional%20Conversations/Table%204.png)

> 表 4：手工设计的模板，用于连接现有材料和生成的指令。

## 用户模拟和强化

保持用户模型的理想行为是成功实现自动对话生成的关键。据观察，当用户模型只获得当前对话历史记录时，它往往会扮演人工智能助手的角色。这种“角色互换”的情况会严重影响多轮对话的连贯性。为了解决这个问题，除了展示对话历史之外，作者还加入了一些 prompt，明确指示模型采用不同的用户性格。在第二部分中，采用了一个 prompt 来提醒模型对话的主要目的，从而促进对话更加自然流畅。数据生成过程完成后，会进一步过滤，以确保整体数据质量。为了增强用户回答的真实性，特别排除了“Thank you”、“Thanks”和“You're welcome”等过于礼貌的语句。

## 数据分析

作者对 UltraChat 和其他几个教学数据集进行了统计分析，如表 5 所示。UltraChat 在规模上非常突出，是最大的公开可用数据集之一。此外，它还显示了最高的平均回合数和最长的每个数据实例的平均长度。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Enhancing%20Chat%20Language%20Models%20by%20Scaling%20High-quality%20Instructional%20Conversations/Table%205.png)

> 表 5：现有教学数据集的统计数据。词法多样性是通过平均每个语篇的 MTLD 分数（McCarthy 和 Jarvis，2010 年）计算出来的。从每个数据集中随机抽取 10000 个样本，用于测量主题多样性和一致性。话题多样性是通过使用 OpenAI embedding API 平均每对数据之间的余弦距离来测量的。连贯性由 ChatGPT 以 1-10 分来评分。

为了评估多样性，作者测量了词汇多样性和主题多样性。在词汇多样性方面，UltraChat 优于之前的数据集。然而，在话题多样性方面，UltraChat 与 GPT4ALL 相比略有不足，但仍大大超过了其他数据集。这可能是由于每个数据实例中的 token 数量相对较多，从而规范了每个对话的数据嵌入（GPT4ALL 数据集在单轮中）。为了确保多轮对话的一致性，作者还进行了一致性评估。结果表明，大多数数据集都表现出了相对较高的一致性。值得注意的是，UltraChat 和 Baize 数据的一致性排名最高。

## 基于 UltraChat 数据集训练 UltraLLaMA

作者在 UltraChat 数据集上对 LLaMA-13B 模型进行了训练，从而开发出了 LLaMA-13B 的增强变体 UltraLLaMA。为了提高模型对对话上下文的理解能力，将每段对话分解成更小的序列，限制它们的最大长度为 2048 个 tokens。在训练过程中，只计算模型回答的 loss。这种方法确保了模型能够从对话的早期部分获取相关信息，从而能够更全面地理解正在进行的对话。通过结合之前的上下文，UltraLLaMA 能够生成更符合上下文、更连贯的回答。使用标准的交叉熵损失对模型进行微调。该模型使用 128 个 A100 GPU 进行训练，总 batch_size 为 512。

# 结果评估

评估聊天模型生成的回复质量是一项重大挑战，尤其是考虑到不同环境下可能存在的不稳定性。传统的基准一直被用于评估目的；但是，当代的方法是利用 ChatGPT 和 GPT-4 等高级模型。在作者的初步实验中，与人工评估相比，这种做法已被证明能产生更可靠的结果。

**验证集**：这套评估包括 Vicuna 基准以及由 GPT-4 生成的另外 300 个问题和说明。这些问题/指令涵盖了广泛的主题，包括常识、世界知识、专业知识（特别是物理和生物）、数学、反应生成和写作任务。此外，问题集的每个部分还根据不同的难度进一步分类。表 6 列出了评估问题集的一些示例。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Enhancing%20Chat%20Language%20Models%20by%20Scaling%20High-quality%20Instructional%20Conversations/Figure%206.png)

> 表 6：作者创建的验证集的一些示例。

**基线模型**：作者选用 Alpaca、Vicuna-13B、Koala-13B、Dolly-V2 和 OpenAssistant-12B 作为基线模型。

## 输出比较

使用 ChatGPT 将模型输出与每个基线模型在每个问题上的输出进行比较。具体来说，分别从两个模型中输入问题和一对独立的答案，并让 ChatGPT 对每个答案进行 1 到 10 分的评分，并提供给定分数的理由。评估 prompt 旨在优先考虑正确性，而不是信息量等其他因素。

**回答的顺序会对评估结果产生重大影响。为了解决这个问题，随机确定了每个问题的回答顺序**。最后，统计了每个基线模型的胜/平/负次数，结果如图 2 所示。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Enhancing%20Chat%20Language%20Models%20by%20Scaling%20High-quality%20Instructional%20Conversations/Figure%202.png)

> 图 2：UltraLLaMA 与其他基线在策划评估集上的输出比较。评估由 ChatGPT 完成。

可以看出，与评估集中的所有开源模型相比，UltraLLaMA 都表现出了卓越的性能，胜率高达 85%，令人印象深刻。值得注意的是，UltraLLaMA 的胜率还比 Vicuna 高出 13%。

## 独立评分

考虑到成对比较的不稳定性，作者还通过 ChatGPT 进行了独立评分，根据回答质量从 1 到 10 分不等。表 7 展示了 UltraLLaMA 与基准模型之间的评分比较。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Enhancing%20Chat%20Language%20Models%20by%20Scaling%20High-quality%20Instructional%20Conversations/Table%207.png)

> 表 7：每个模型在经过策划的评估集上的总体得分和分段得分。得分介于 1 到 10 之间，报告的是平均得分。粗体表示最佳得分，下划线表示次佳得分。

值得注意的是，与所有开源模型相比，UltraLLaMA 在总分上表现出了显著的优势。此外，UltraLLaMA 几乎在评估集的每个部分都取得了最高的性能，展示了其非凡的能力。

这种细分还有助于深入了解每种模型在特定类型的问题和指令上的表现。一般来说，**所有模型在与常识性知识和对世界的一般理解有关的较简单问题上的表现都较好**。然而，事实证明，**涉及推理和创造性写作的较复杂任务对大多数模型来说都具有挑战性**。有趣的是，尽管 Alpaca 只有 70 亿个参数，但在与常识和世界知识相关的问题上，它的表现却比大型模型要好。然而，在要求更高的任务上，它就落后了。此外，值得注意的是，基于 Pythia（Biderman 等人，2023 年）的 Dolly 和 OpenAssistant 与基于 LLaMA 的类似甚至更小的模型相比，显示出更差的性能。这一观察结果凸显了**底层骨干语言模型的重要性**。

## 系统 prompt 的影响

使用系统 prompt 来提示 LLM 的角色和输出风格是一种常见的做法。在作者的评估中，发现**系统 prompt 对生成输出的风格有很大影响**。具体来说，当系统提示模型提供“有帮助的详细”回复时，模型往往会生成更多相关细节，从而提高回复的信息量。虽然这些 prompt 可能不会直接影响确定性问题的准确性，但它们确实会影响额外信息的提供，从而进一步提高答复的整体质量。

![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Enhancing%20Chat%20Language%20Models%20by%20Scaling%20High-quality%20Instructional%20Conversations/Table%209.png)

> 表 9：有系统 prompt 和无系统 prompt 的 UltraLLaMA 比较。

为了说明这种影响，可以参考表 9，在该表中，两个输出都是正确的，但在系统 prompt 指导下的模型得到的回答信息量更大。

> PS：2023 年 2 月份的一篇论文《Guidling Large Language Models via Directional Stimulus Prompting》提出了一个名为“定向刺激 prompt”（Directional Stimulus Prompting）的新型 prompt 框架，先训练一个经过微调、强化学习后的小型模型，然后使用该小型模型根据用户的查询来生成对应的刺激（文本），将其添加到 prompt 中来引导黑盒大语言模型朝着所需的输出方向前进。我理解更改系统 prompt 和情感刺激的作用是类似的，引导模型输出向着自己想要的方向发展。

# 总结

本篇工作介绍了 **UltraChat，这是一种结构化设计的多轮指令对话数据**，对聊天语言模型进行了有效的微调，显著提升了模型性能。UltraChat 的设计理念和方法为未来聊天模型的发展提供了新的方向，证明了高质量指令对话数据集在提升模型性能方面的重要性。
