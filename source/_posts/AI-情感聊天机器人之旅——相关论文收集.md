---
title: AI-情感聊天机器人之旅——相关论文收集
date: 2024-06-20 21:15:26
tags:
- 工作内容
category:
- 业务相关
- 闲聊场景
---

# 开放域闲聊场景

[Prompted LLMs as Chatbot Modules for Long Open-domain Conversation](https://arxiv.org/abs/2305.04533)
- **发布日期**：2023-05-01
- **简要介绍**：作者提出了 MPC（模块化提示聊天机器人），这是一种无需微调即可创建高质量对话代理的新方法，可以成为长期开放域聊天机器人的有效解决方案。该方法利用预训练好的大型语言模型（LLM）作为单独的模块，通过使用 few-shot、思维链（CoT）和外部记忆等技术来实现长期一致性和灵活性。

    ![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Prompted%20LLMs%20as%20Chatbot%20Modules%20for%20Long%20Open-domain%20Conversation/Figure%201.png)

    MPC 本质上是一种 RAG 或者说 Agent，在输入和输出的中间添加了更多思考和记忆的环节，将 LLM 从“人”的角色进一步拆分为“大脑”和“嘴巴”。这种明确的分工的确能够提升最终的效果，但同样会遇到 RAG、Agent 成本较高的问题，以及引入更多中间环节造成的误差累积。为什么成本较高？为了确保中间环节结果的正确性，往往也会接一个 LLM 去做判断，或者训练专门的小模型，这些都需要资源，并且对整个推理过程的时延造成一定的影响。在业务上是否真得要这么做，还需要进一步衡量效果和成本的 tradeoff。


[RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models](https://arxiv.org/abs/2310.00746)
- **发布日期**：2023-10
- **简要介绍**：介绍 RoleLLM，一个用于对 LLM 的角色扮演能力进行基准测试、诱导和增强的框架，包括四个阶段：(1) 100 个角色的角色档案构建；(2) 基于上下文的指令生成（Context-Instruct），用于角色特定知识的提取；(3) 使用 GPT 的角色提示（RoleGPT），用于说话风格的模仿；(4) 角色条件指令调整（Role-Conditioned Instruction Tuning，RoCIT），用于微调开源模型和角色定制。通过 Context-Instruct 和 RoleGPT，作者创建了 RoleBench，这是第一个系统化、精细化的角色扮演基准数据集。

    ![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/RoleLLM%20Benchmarking%20Eliciting%20and%20Enhancing%20Role-Playing%20Abilities%20of%20Large%20Language%20Models/Figure_2.png)


[Think Before You Speak: Cultivating Communication Skills of Large Language Models via Inner Monologue](https://arxiv.org/abs/2311.07445)
- **发布日期**：2023-11-13
- **简要介绍**：提出了一个简单而有效的策略来提高 LLM 的拟人化和主动性，为 LLM 添加了五种交流技能，使其成为拟人化的聊天机器人，而不是信息搜索工具。同时，通过 prompt 工程和上下文学习为 LLM 添加了内心独白，让 LLM 更好地理解和使用交流技能。

    ![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Think%20Before%20You%20Speak%20Cultivating%20Communication%20Skills%20of%20Large%20Language%20Models%20via%20Inner%20Monologue/Figure%202.png)


[Blending Is All You Need: Cheaper, Better Alternative to Trillion-Parameters LLM](https://arxiv.org/abs/2401.02994)
- **发布日期**：2024-01-05
- **简要介绍**：介绍了“混合”（Blended），这是一种通过随机选择不同系统（模型）的回复来组合多个聊天人工智能的简单方法。经验证据表明，当特定的较小模型被协同混合时，它们的性能有可能超过或赶上更大的同类模型，同时还能保持较小系统的推理成本。

    ![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Blending%20Is%20All%20You%20Need%20Cheaper%20Better%20Alternative%20to%20Trillion-Parameters%20LLM/Algorithm%201.png)

    在对话过程中，Blended 每次都会随机（均匀）选择产生当前响应的聊天模型（例如，有 A、B 和 C 三个聊天模型，随机从中挑选一个模型来生成响应）。论文中也提到“特定聊天模型生成的回复是以之前选择的聊天模型生成的所有回复为条件的。这意味着不同的聊天模型会对当前回复的输出产生隐性影响。因此，当前的回复融合了各个聊天人工智能的优势，它们相互协作，创造出了更吸引人的整体对话”。由于这篇论文是 chai，并且在他们自家的产品上得到了验证，因此我们也尝试了该方案。但在使用过程中，如果 A、B 和 C 这三个模型的差距较大时，用户所看到的回答风格差距也较大，就好像角色是“精神分裂”的。
    
    如果成本足够的话，可以考虑异步同时调用这三个模型，然后在后处理环节中调用一致性方法或者 reward model 去评估各响应的结果，挑选出最适合的响应。或者根据对话轮数来选择聊天模型，例如前 10 轮调用 A 模型；10 轮到 50 轮调用 B 模型；50 轮以后调用 C 模型。


[LLM-Blender：Ensembling Large Language Models with Pairwise Ranking and Generative Fusion](https://arxiv.org/abs/2306.02561)
- **发布日期**：2023-06-05
- **简要介绍**：提出了一个名为 LLM-BLENDER 的集合框架，该框架由两个模块组成：Pair-Ranker 模块和 Gen-Fuser 模块。Pair-Ranker 采用一种专门的成对比较方法来区分候选输出之间的细微差别。Gen-Fuser 的目标是合并排名靠前的候选输出，通过利用它们的优势和减少它们的劣势来生成改进的输出。

    ![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Blending%20Is%20All%20You%20Need%20Cheaper%20Better%20Alternative%20to%20Trillion-Parameters%20LLM/Figure%202.png)

    这篇工作比 chai 那篇论文要早，从方法上来说（我个人理解）也更加靠谱一些，但 Gen-Fuser 要怎么做是个比较棘手的问题。并且整体的流程过长（即便是异步调用的方式，整体的时长取决于最后一个输出的耗时，并且不同模型的输出有长有短），在实际的使用过程中要不可避免地要增加时延以及降低服务的总体吞吐。


[Aligning to Thousands of Preferences via System Message Generalization](https://arxiv.org/abs/2203.15556)
- **发布日期**：2024-05-28
- **GitHub 仓库**：https://github.com/kaistAI/Janus
- **简要介绍**：介绍了一种新颖的方法，可使 LLM 与不同的用户偏好保持一致，而无需针对每个人的偏好进行持续的再训练。该方法利用独特的 system message，引导 LLM 根据特定、细微的用户偏好做出响应。

    ![](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/paper/Aligning%20to%20Thousands%20of%20Preferences%20via%20System%20Message%20Generalization/Figure%202.png)

    核心还是在构造数据上，通过在 system message 中整合各方面（Style、Background knowledge、Informativeness、Harmlessness）的偏好，交由 GPT-4 等强 LLM 去生成对应的输出。该方法实际上和 alpaca 根据种子数据去生成指令数据集，以及情感刺激之类的 prompt 工程类似，无非是将用户偏好预置在 system message 内，让 GPT-4 等强 LLM 来生成响应，作为训练数据集，本质上仍然是一种 RLAIF。

