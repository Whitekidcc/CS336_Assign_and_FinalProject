# Assignment 1 任务索引（Spring 2026）

依据仓库随附的 `cs336_assignment1_basics.pdf` Version 26.0.3。下表保留 PDF 的 **Problem 标识**；它不是连续数字题号。章节按 PDF 原有的 1～7 节排列，不能把功能分类误认成 PDF 的 Part 1～6。表中“源码”指学生日后自行编写的位置：仓库目前没有这些算法模块，因而不虚构文件名。“适配/测试”只列实际存在的文件；“无单测”表示当前仓库没有对应的专门自动测试，仍须按 PDF 完成文字、脚本或实验交付物。

推荐顺序遵循依赖关系：先弄清 Unicode 与 BPE，再读模型基本模块、优化与数据接口，最后训练、生成和实验。每一节的表格从上到下即该节建议顺序；少量书面题可与相邻实现题并行。作业实现、算法设计、实验运行和书面答案均由学生完成。

## 第 1 节 Assignment Overview（作业总览）

先阅读 PDF 对 AI 工具、从零实现、交付物、数据集和库使用范围的规定。这里没有独立 Problem。官方规定 `cs336_basics/` 是学生实现区域，`tests/adapters.py` 是测试胶水层，`tests/test_*.py` 不得修改。

## 第 2 节 Byte-Pair Encoding (BPE) Tokenizer

本节前置知识：Unicode code point、UTF-8 字节、Pre-tokenization（预分词）、BPE merge、特殊词元与流式编码。`cs336_basics/pretokenization_example.py` 仅是文件分块辅助示例。

| PDF Problem | 中文任务及学生工作 | 涉及源码 | 适配/测试 | 建议次序 |
| --- | --- | --- | --- | --- |
| `unicode1`（2.1） | 理解 Unicode：观察 `chr(0)`、字符串表示与打印效果，并书面解释。 | 无需新增作业源码 | 无单测 | 1 |
| `unicode2`（2.2） | 比较 UTF-8、UTF-16、UTF-32；分析 PDF 所给错误解码示例及无效字节序列，提交文字回答。 | 无需新增作业源码 | 无单测 | 2 |
| `train_bpe`（2.5） | 学生自行编写 byte-level BPE 训练功能；输入文本路径、词表大小与特殊词元，输出 `dict[int, bytes]` 词表及按创建顺序排列的 merge 列表。词表上限包含初始字节、合并结果和特殊词元；训练时特殊词元是硬边界，不参与 merge 统计。 | `cs336_basics/`（学生自定文件）；可阅读 `cs336_basics/pretokenization_example.py` | `tests/adapters.py:run_train_bpe`；`tests/test_train_bpe.py` | 3 |
| `train_bpe_tinystories`（2.5） | 在 TinyStories 上训练最大 10,000 词表且包含 `<\|endoftext\|>`，保存词表和 merge；记录耗时、内存、最长词元并分析瓶颈。 | 学生自定训练脚本 | 当前无专门单测 | 5（先完成 `train_bpe`） |
| `train_bpe_expts_owt`（2.5） | 在 OpenWebText 上训练最大 32,000 词表、保存结果，并与 TinyStories 词元特点比较。 | 学生自定训练脚本 | 当前无专门单测 | 6 |
| `tokenizer`（2.6） | 学生自行实现 Tokenizer：从词表和 merge 构造、从文件加载、`encode`、内存友好的 `encode_iterable`、`decode`，并支持自定义特殊词元。无效 UTF-8 解码须按 PDF 替换为 U+FFFD。 | `cs336_basics/`（学生自定文件） | `tests/adapters.py:get_tokenizer`；`tests/test_tokenizer.py` | 4 |
| `tokenizer_experiments`（2.7） | 对两种语料抽样比较压缩率、估计吞吐量，并将训练/开发集编码为后续训练可用的 Token ID 序列。 | 学生自定实验与数据处理文件 | 当前无专门单测 | 7 |

## 第 3 节 Transformer Language Model Architecture

本节前置知识：`torch.Tensor` 形状与广播、`torch.nn.Module`、矩阵乘法、残差连接、归一化、注意力与 RoPE。先读接口和测试，再由学生决定源码文件组织方式。若 PDF 的建议接口与实际 `tests/adapters.py` 形参表述有出入，以当前测试调用方式核对，但不得改测试。

| PDF Problem | 中文任务及学生工作 | 涉及源码 | 适配/测试 | 建议次序 |
| --- | --- | --- | --- | --- |
| `linear`（3.3） | 从零实现无 bias 的 Linear 模块及其初始化；保留权重形状要求。 | `cs336_basics/`（学生自定文件） | `run_linear`；`tests/test_model.py::test_linear` | 1 |
| `embedding`（3.3） | 从零实现 Embedding 模块，通过 Token ID 查询词元向量。 | 同上 | `run_embedding`；`test_embedding` | 2 |
| `rmsnorm`（3.4） | 实现 RMSNorm 模块，关注数值稳定性与输入输出 dtype。 | 同上 | `run_rmsnorm`；`test_rmsnorm` | 3 |
| `positionwise_feedforward`（3.4） | 实现 SiLU 与 SwiGLU 组成的逐位置前馈网络；PDF 指定 `d_ff` 约为 `8/3 × d_model` 且为 64 的倍数。 | 同上 | `run_silu`、`run_swiglu`；`test_silu_matches_pytorch`、`test_swiglu` | 4 |
| `rope`（3.4） | 实现 Rotary Position Embedding（旋转位置编码，RoPE），支持任意前导批次维与提供的 token 位置。 | 同上 | `run_rope`；`test_rope` | 5 |
| `softmax`（3.4） | 实现沿指定维度归一化的 Softmax，满足 PDF 数值稳定性要求。 | 同上 | `run_softmax`；`tests/test_nn_utils.py::test_softmax_matches_pytorch` | 6 |
| `scaled_dot_product_attention`（3.4） | 实现支持可选布尔 mask、不同 query/key 长度及额外批次维的 Scaled Dot-Product Attention（缩放点积注意力）。 | 同上 | `run_scaled_dot_product_attention`；`test_scaled_dot_product_attention`、`test_4d_scaled_dot_product_attention` | 7 |
| `multihead_self_attention`（3.4） | 实现带 Causal Mask（因果掩码）的 Multi-Head Self-Attention（多头自注意力）；结合第 3.4 节的位置编码要求。 | 同上 | `run_multihead_self_attention`、`run_multihead_self_attention_with_rope`；对应两个 `test_...` | 8 |
| `transformer_block`（3.5） | 组装使用 RoPE 的 Pre-Norm Transformer Block（预归一化 Transformer 块）。 | 同上 | `run_transformer_block`；`test_transformer_block` | 9 |
| `transformer_lm`（3.5） | 组装完整 Transformer LM，输出每个位置的未归一化预测分数；支持短于 context length 的输入。 | 同上 | `run_transformer_lm`；`test_transformer_lm`、`test_transformer_lm_truncated_input` | 10 |
| `transformer_accounting`（3.5） | 按 PDF 指定模型配置计算参数、内存和前向 FLOPs，并比较不同规模模型；提交分析。 | 无需新增作业源码 | 无单测 | 11 |

## 第 4 节 Training a Transformer LM

本节前置知识：logits、Cross-Entropy（交叉熵）、Autograd、优化器状态、学习率日程和梯度范数。

| PDF Problem | 中文任务及学生工作 | 涉及源码 | 适配/测试 | 建议次序 |
| --- | --- | --- | --- | --- |
| `cross_entropy`（4.1） | 实现对 batch 求平均的 Cross-Entropy Loss（交叉熵损失），处理大幅度 logits 的数值稳定性。 | `cs336_basics/`（学生自定文件） | `run_cross_entropy`；`tests/test_nn_utils.py::test_cross_entropy` | 1 |
| `learning_rate_tuning`（4.2） | 在 PDF 给出的 SGD 玩具例子中试不同学习率，书面记录损失变化。 | 无需新增作业源码 | 无单测 | 2 |
| `adamw`（4.3） | 自行实现继承 `torch.optim.Optimizer` 的 AdamW Optimizer。 | `cs336_basics/`（学生自定文件） | `get_adamw_cls`；`tests/test_optimizer.py::test_adamw` | 3 |
| `adamw_accounting`（4.3） | 按 PDF 假设分析 AdamW 训练时参数、激活、梯度和优化器状态的内存与计算成本。 | 无需新增作业源码 | 无单测 | 4 |
| `learning_rate_schedule`（4.4） | 实现线性 warmup 加 cosine decay 的学习率计划及末段行为。 | `cs336_basics/`（学生自定文件） | `run_get_lr_cosine_schedule`；`test_get_lr_cosine_schedule` | 5 |
| `gradient_clipping`（4.5） | 实现对全部参数梯度的整体 ℓ2 范数裁剪，并原位更新梯度。 | `cs336_basics/`（学生自定文件） | `run_gradient_clipping`；`tests/test_nn_utils.py::test_gradient_clipping` | 6 |

## 第 5 节 Training loop

本节前置知识：next-token 训练样本、NumPy memory mapping、`state_dict` 与恢复训练所需状态。

| PDF Problem | 中文任务及学生工作 | 涉及源码 | 适配/测试 | 建议次序 |
| --- | --- | --- | --- | --- |
| `data_loading`（5.1） | 从一维 Token ID 序列采样输入与右移一位的目标；两个张量均为 `(batch_size, context_length)` 且放在指定 device。 | `cs336_basics/`（学生自定文件） | `run_get_batch`；`tests/test_data.py::test_get_batch` | 1 |
| `checkpointing`（5.2） | 保存、读取模型状态、优化器状态和迭代号，以便继续训练。 | `cs336_basics/`（学生自定文件） | `run_save_checkpoint`、`run_load_checkpoint`；`tests/test_serialization.py::test_checkpointing` | 2 |
| `training_together`（5.3） | 学生自行编写可配置训练脚本：高效读取训练/验证数据、记录指标、保存 Checkpoint（训练检查点）。 | 学生自定训练脚本 | 当前无专门单测 | 3 |

## 第 6 节 Generating text

| PDF Problem | 中文任务及学生工作 | 涉及源码 | 适配/测试 | 前置与次序 |
| --- | --- | --- | --- | --- |
| `decoding`（6） | 实现基于语言模型的文本生成；PDF 建议支持 prompt、最大生成长度、temperature 和 Top-p（核采样），遇 `<\|endoftext\|>` 停止。 | 学生自定生成文件 | 当前无专门单测 | 完成 Tokenizer、Transformer LM 与训练后 |

## 第 7 节 Experiments

本节要求训练、记录、绘制曲线、生成和讨论结果；表格只索引任务，不提供实验结论。PDF 中 B200 时长是题目标注的资源量；其中 leaderboard 的正式运行上限为单张 B200 的 45 分钟。

| PDF Problem | 中文任务及学生工作 | 涉及源码 | 适配/测试 | 建议次序 |
| --- | --- | --- | --- | --- |
| `experiment_log`（7.1） | 建立按梯度步数与墙钟时间记录损失的实验日志及曲线资料。 | 学生自定日志/训练文件 | 无专门单测 | 1 |
| `learning_rate`（7.2） | 在 TinyStories 上扫描学习率，提交学习曲线、搜索策略、验证损失与发散分析。 | 学生自定训练/实验文件 | 无专门单测 | 2 |
| `batch_size_experiment`（7.2） | 比较从 1 到显存上限的多种 batch size（包括 64、128 等），必要时重调学习率，并报告曲线与发现。 | 学生自定实验文件 | 无专门单测 | 3 |
| `generate`（7.2） | 使用训练好的 TinyStories Checkpoint 生成文本，报告至少 256 个 token（或至首个特殊结束词元）及流畅度分析。 | 学生自定生成文件 | 无专门单测 | 4 |
| `layer_norm_ablation`（7.3） | 移除 RMSNorm 并训练，比较原学习率与较低学习率的稳定性和曲线。 | 学生自定模型/实验文件 | 无专门单测 | 5 |
| `pre_norm_ablation`（7.3） | 改为 Post-Norm（后归一化）并与 Pre-Norm 曲线比较。 | 学生自定模型/实验文件 | 无专门单测 | 6 |
| `no_pos_emb`（7.3） | 比较移除位置信息的 NoPE 与 RoPE 的训练曲线。 | 学生自定模型/实验文件 | 无专门单测 | 7 |
| `swiglu_ablation`（7.3） | 在参数量大致匹配下比较 SwiGLU 与无门控的 SiLU 前馈网络，提交曲线与讨论。 | 学生自定模型/实验文件 | 无专门单测 | 8 |
| `main_experiment`（7.4） | 用 TinyStories 实验相同模型架构和训练迭代数在 OpenWebText 上训练，比较损失并提供生成结果。 | 学生自定训练/生成文件 | 无专门单测 | 9 |
| `leaderboard`（7.5） | 按 PDF 规则改进模型与超参数；仅使用提供的 OpenWebText 训练数据，在单张 B200 的 45 分钟限制内报告验证损失、时间曲线和方法。 | 学生自定实验文件 | 无专门单测 | 10 |

## 与仓库对应的阅读定位

- 所有目前存在的 `cs336_basics/` 源码只有 `__init__.py` 与 `pretokenization_example.py`；未来算法模块的文件名由学生决定，不要把本索引的中文标题当作必须创建的路径。
- `tests/adapters.py` 中各接口目前抛出 `NotImplementedError`。PDF 要求学生最终将自己的实现接入它们，但这份中文本地化没有填写任何适配器或 TODO。
- 具体测试输入、输出、Tensor Shape、边界条件与单测命令见[测试文件阅读指南](测试文件阅读指南.md)；适配器参数和返回值说明见[适配接口阅读指南](适配接口阅读指南.md)。
