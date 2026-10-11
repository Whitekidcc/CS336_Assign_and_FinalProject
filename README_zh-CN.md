# CS336 Spring 2026 Assignment 1：Basics 中文阅读导航

本导航对应仓库随附的 `cs336_assignment1_basics.pdf`（Version 26.0.3，Spring 2026）。仓库原 `README.md` 标题仍写 Spring 2025，使用时请以 PDF 中的题号、约束和交付要求为准。原英文 README 保留，方便与官方更新对照。

## 项目简介

本作业要求学生从零构建并训练 Transformer Language Model（Transformer 语言模型）：训练 Byte-Pair Encoding（字节对编码，BPE）Tokenizer，实现 Transformer、Cross-Entropy（交叉熵）、AdamW Optimizer（AdamW 优化器）与训练循环，随后在 TinyStories 和 OpenWebText 上训练、生成与评估。`cs336_basics/` 目前几乎是空的；实现工作由学生本人完成。本仓库的中文资料只解释已有要求和接口。

## 目录结构与职责

以下是当前工作目录的文件树；`.idea/` 和 `.DS_Store` 是本地编辑器或系统文件，不属于作业交付接口。

```text
assignment1-basics-main/
├── .DS_Store
├── .gitignore                         Git 忽略规则
├── .idea/                             本地 IDE 配置
│   ├── assignment1-basics-main.iml
│   ├── inspectionProfiles/
│   │   ├── Project_Default.xml
│   │   └── profiles_settings.xml
│   ├── modules.xml
│   └── workspace.xml
├── AGENTS.md                          AI 使用与教学规范
├── CHANGELOG.md                       官方版本变更记录
├── CLAUDE.md                          AI 使用规范副本
├── LICENSE                            开源许可证
├── README.md                          官方英文安装与数据说明
├── README_zh-CN.md                    本中文导航
├── cs336_assignment1_basics.pdf       官方作业说明，Version 26.0.3
├── make_submission.sh                 官方提交打包脚本
├── pyproject.toml                     Python 依赖、构建、pytest、Ruff 配置
├── uv.lock                            锁定依赖版本
├── cs336_basics/                      学生编写作业实现的位置
│   ├── __init__.py                    包版本信息
│   └── pretokenization_example.py    预分词分块辅助示例
├── docs/zh-CN/                        中文阅读资料
│   ├── Assignment1任务索引.md
│   ├── 测试文件阅读指南.md
│   └── 适配接口阅读指南.md
└── tests/                             官方测试及适配层；本次本地化未修改
    ├── __init__.py
    ├── adapters.py                    学生将自己实现接入测试的接口（当前均未实现）
    ├── common.py                      测试辅助与 GPT-2 字节映射
    ├── conftest.py                    pytest fixture 与快照比较工具
    ├── test_data.py                   语言模型批次采样测试
    ├── test_model.py                  Transformer 各组件与整体模型测试
    ├── test_nn_utils.py               Softmax、交叉熵、梯度裁剪测试
    ├── test_optimizer.py              AdamW、学习率计划测试
    ├── test_serialization.py          Checkpoint 保存与恢复测试
    ├── test_tokenizer.py              Tokenizer 编码、解码、内存测试
    ├── test_train_bpe.py              BPE 训练结果与速度测试
    ├── _snapshots/                    官方数值与结构快照
    │   ├── test_4d_scaled_dot_product_attention.npz
    │   ├── test_adamw.npz
    │   ├── test_embedding.npz
    │   ├── test_linear.npz
    │   ├── test_multihead_self_attention.npz
    │   ├── test_multihead_self_attention_with_rope.npz
    │   ├── test_positionwise_feedforward.npz
    │   ├── test_rmsnorm.npz
    │   ├── test_rope.npz
    │   ├── test_scaled_dot_product_attention.npz
    │   ├── test_swiglu.npz
    │   ├── test_train_bpe_special_tokens.pkl
    │   ├── test_transformer_block.npz
    │   ├── test_transformer_lm.npz
    │   └── test_transformer_lm_truncated_input.npz
    └── fixtures/                      官方测试语料与参考权重
        ├── address.txt
        ├── corpus.en
        ├── german.txt
        ├── gpt2_merges.txt
        ├── gpt2_vocab.json
        ├── special_token_double_newlines_non_whitespace.txt
        ├── special_token_trailing_newlines.txt
        ├── tinystories_sample.txt
        ├── tinystories_sample_5M.txt
        ├── train-bpe-reference-merges.txt
        ├── train-bpe-reference-vocab.json
        └── ts_tests/
            ├── model.pt
            └── model_config.json
```

`tests/adapters.py` 属于测试适配接口：PDF 说明学生最终要把自己的实现接到这里，但适配器应只是调用代码，不应放入实质算法。本次本地化未改动它。`tests/test_*.py`、`tests/common.py`、`tests/conftest.py`、fixture、快照都是官方测试资产；不要修改测试内容或快照。`cs336_basics/pretokenization_example.py` 是预分词并行分块的辅助示例，不是完整 BPE 训练实现。`pyproject.toml` 与 `uv.lock` 管理依赖；`make_submission.sh` 负责运行测试和打包，脚本中的压缩包名称仍含 Spring 2025，切勿据此判断 PDF 年份。

## 推荐阅读顺序

1. 先读 `AGENTS.md`、本导航和 PDF 第 1 节，确认允许使用的库、交付物和 AI 政策。
2. 读 PDF 第 2 节，再看 `cs336_basics/pretokenization_example.py`、`tests/test_train_bpe.py`、`tests/test_tokenizer.py` 与相关适配器。先区分 Unicode 字符、UTF-8 字节、预词元和 Token ID。
3. 读 PDF 第 3 节，按 Linear、Embedding、RMSNorm、SiLU/SwiGLU、RoPE、Softmax、Scaled Dot-Product Attention、Multi-Head Self-Attention、Transformer Block、Transformer LM 的依赖顺序阅读 `tests/test_model.py`。
4. 读 PDF 第 4 节和 `tests/test_nn_utils.py`、`tests/test_optimizer.py`，明确损失、优化器、学习率计划与梯度裁剪的接口。
5. 读 PDF 第 5 节和 `tests/test_data.py`、`tests/test_serialization.py`，再阅读训练循环要求。
6. 最后读 PDF 第 6～7 节的生成、实验与评估要求；按 [任务索引](docs/zh-CN/Assignment1任务索引.md) 对照交付物。

## 环境、测试与数据

官方 README 推荐安装 [`uv`](https://github.com/astral-sh/uv#installation)；也可用 `pip install uv` 或 `brew install uv`。[`uv` 项目依赖说明](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)解释环境管理。运行仓库中的 Python 文件时，原命令是：

```sh
uv run <python_file_path>
```

运行全部官方单元测试的原命令是：

```sh
uv run pytest
```

起始状态下，测试会因适配器的 `NotImplementedError` 而失败；这只表示作业尚未实现。PDF 还给出 `uv run pytest tests/test_train_bpe.py`、`uv run pytest tests/test_tokenizer.py` 和 `uv run pytest -k test_linear` 等定向命令。单个测试的运行方式见[测试文件阅读指南](docs/zh-CN/测试文件阅读指南.md)。本次本地化没有运行作业实验，也没有处理未通过的测试。

官方 README 的数据下载命令如下，目标是 TinyStories 的训练集、验证集，以及 OpenWebText 的训练集、验证集样本。下载属于后续学生实验步骤；阅读仓库不需要先执行。

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

如果发现作业说明或代码问题，官方 README 邀请通过 GitHub issue 或 pull request 反馈；完整要求仍以随附 PDF 为准。

## 阅读接口时的共同注意事项

- **Tensor Shape（张量形状）**：保持接口中 `...`、`batch_size`、`sequence_length`、`d_model`、`d_k`、`vocab_size` 的原始含义；`...` 表示任意前导维。不要把元素个数相同误认为形状相同。
- **dtype 与 device**：Token ID 通常是整数张量，模型权重与激活是浮点张量，mask 是布尔张量。对照测试 fixture 和接口要求检查所在设备；`run_get_batch` 的测试会检查无效设备参数是否真正生效。
- **输入输出**：适配器的类型注解和 Docstring 是公开接口说明；数值快照只验证现有测试覆盖的行为。前向输出与训练标签的维度、位置要逐项核对。
- **从零实现限制**：PDF 第 1 节要求不能直接使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的现成定义，例外是 `torch.nn.Parameter`、`torch.nn` 容器类（如 `Module`、`ModuleList`、`Sequential`）和 `torch.optim.Optimizer` 基类；其他 PyTorch 定义可用。PDF 的 `linear` Problem 又明确指定用 `torch.nn.init.trunc_normal_` 初始化权重，阅读时应保留这一具体题目要求。测试使用现成模块作参照，不代表学生实现也可调用相应高级封装。
- **文本与特殊词元**：区分 Unicode 字符、UTF-8 字节、BPE merge 顺序、预分词边界和 `<|endoftext|>`。有关特殊词元的要求请以 PDF 第 2 节为准；适配器中有一处描述与 PDF 不一致，见[适配接口阅读指南](docs/zh-CN/适配接口阅读指南.md)。

术语首次出现时采用中文解释加英文名称；类名、API、函数名、Tensor Shape 与数学符号保留原样。
