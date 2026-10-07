# [Rethinking Gradient Approximation in Quantization: A Zeroth-Order Expectation Perspective](https://openreview.net/forum?id=0ZZ3tyBGyc)

[English](README.md) | [中文](README_zh.md)

我们提出 **ZOE-Grad（Zeroth-Order Expectation Gradient）**：一种从零阶期望视角出发、面向 LLM 量化的边界感知代理梯度框架。ZOE-Grad 建立了扰动分布与边界依赖代理梯度之间的构造性双向映射，将 STE 刻画为固定幅度 Rademacher 扰动所诱导的特殊情形，并为原始量化目标给出了有界误差的收敛保证。

本仓库提供论文中下游微调实验的实现，覆盖 OPT、LLaMA-2 和 Qwen3。当前统一脚本支持 **STE、HTGE、Uniform 和 Normal** 四种代理梯度以及八个下游任务。

## 方法概览

<p align="center">
  <img src="assets/method_overview.svg" alt="Method Overview" width="100%">
</p>

前向传播保留 hard rounding，仅在反向传播时使用所选代理梯度。模型权重保持冻结，下游监督信号用于更新量化尺度修正参数，因此代理梯度本身不会带来额外的推理时计算开销。

## 环境安装

```bash
conda create -n ZOE python=3.10.19 -y
conda activate ZOE
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## 所需本地文件

开始实验前，创建以下目录，并放入所选模型所需的文件：

- `pre_quantized_models/`：保存逐层 OmniQuant 参数 checkpoint。量化过程中，`scripts/run.sh` 会在下游训练前加载所选模型的 `*-w4a16.pth` 文件。
- `act_scales/` 和 `act_shifts/`：保存激活统计文件。默认脚本为 OPT-1.3B 和 OPT-6.7B 启用 Learnable Equivalent Transformation（LET），并加载对应 `.pt` 文件；LLaMA-2-7B、LLaMA-2-13B 和 Qwen3-8B 默认关闭 LET。

OmniQuant 参数 checkpoint 和激活统计文件可从 [OmniQuant](https://github.com/OpenGVLab/OmniQuant) 获取；其中未提供的模型文件需要先自行训练得到。

## 使用方法

从仓库根目录运行以下命令：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run.sh opt-1.3b SST2 STE
CUDA_VISIBLE_DEVICES=0 scripts/run.sh llama2-7b SQuAD Normal
CUDA_VISIBLE_DEVICES=0 scripts/run.sh qwen3-8b WSC HTGE
```

支持的模型别名为 `opt-1.3b`、`opt-6.7b`、`llama2-7b`、`llama2-13b` 和 `qwen3-8b`。每个模型均可与以下任务及代理梯度组合：

- **分类任务：** SST2、RTE、CB、BoolQ、WSC、WIC、MultiRC
- **问答任务：** SQuAD
- **代理梯度：** STE、HTGE、Uniform、Normal

分类任务根据模型对候选答案给出的概率进行评分；SQuAD 在训练时使用答案 token 的 teacher-forced loss，在评测时对生成答案计算 F1。默认量化配置为 W4A16，所有任务均使用 WikiText2 数据进行量化校准。

`scripts/run.sh` 支持通过环境变量覆盖常用配置，也可在三个必需参数后追加 `train_main.py` 的其他选项：

```bash
CUDA_VISIBLE_DEVICES=0 STEPS=256 LR=1e-6 NUM_TRAIN=64 scripts/run.sh qwen3-8b RTE Uniform --no_eval true
DRY_RUN=1 scripts/run.sh opt-6.7b SQuAD Normal
```

第一条命令在 RTE 上使用 Uniform 代理梯度训练 Qwen3-8B。`scripts/run.sh` 前的环境变量赋值仅对本次调用生效：

- `CUDA_VISIBLE_DEVICES=0`：使进程可使用 GPU 0。
- `STEPS=256`：将最大训练步数设为 256，而非默认的 5,000。
- `LR=1e-6`：将学习率设为 0.000001。
- `NUM_TRAIN=64`：选择 64 条训练样本。

末尾的 `--no_eval true` 会传给 `train_main.py`，从而跳过训练结束后的评测。

第二条命令选择 OPT-6.7B、SQuAD 和 Normal 代理梯度。设置 `DRY_RUN=1` 时，脚本只打印生成的 `python train_main.py ...` 命令，不加载模型或开始训练，但仍会检查所需的 OmniQuant 参数 checkpoint 是否存在。

支持的环境变量包括 `STEPS`、`LR`、`BATCH_SIZE`、`WBITS`、`ABITS`、`RESUME_PATH`、`NUM_TRAIN`、`NUM_EVAL`、`NUM_DEV`、`DELTA_OVERRIDE`、`T`、`OUTPUT_DIR` 和 `HF_HOME`。

若只希望查看所有支持的模型、任务和代理梯度组合而不执行训练：

```bash
DRY_RUN=1 scripts/run_matrix.sh
```

若要按顺序运行全部组合：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_matrix.sh
```

## 实验结果

SST-2、RTE、CB、BoolQ、WSC、WiC 和 MultiRC 报告准确率（%）；SQuAD 报告 F1。

### W4A16 权重量化

**Table 1(a)。** **LLaMA-2-7B** 在 4-bit 权重量化下不同代理梯度在下游任务上的性能比较。

| 方法 | SST-2 | RTE | CB | BoolQ | WSC | WiC | MultiRC | SQuAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-Shot | 58.03 | 62.09 | 33.93 | 66.10 | 36.54 | 50.16 | 42.40 | 58.71 |
| Zero-Shot-Q | 50.92 | 48.38 | 50.00 | 39.90 | 53.85 | 50.00 | 58.80 | 41.80 |
| STE | 93.46 | 58.12 | 57.14 | 60.70 | 63.46 | 51.84 | 59.40 | 47.33 |
| HTGE | **95.18** | 66.06 | **73.21** | 67.80 | **73.08** | **68.81** | 59.60 | 61.73 |
| Uniform | 94.72 | **69.70** | 69.64 | 62.30 | 64.42 | 54.67 | **62.60** | **66.69** |
| Normal | 94.15 | 67.54 | **73.21** | **69.60** | 65.38 | 53.76 | 60.70 | 66.15 |

**Table 1(b)。** **OPT-6.7B** 在 4-bit 权重量化下不同代理梯度在下游任务上的性能比较。

| 方法 | SST-2 | RTE | CB | BoolQ | WSC | WiC | MultiRC | SQuAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-Shot | 61.24 | 54.87 | 53.57 | 57.30 | 37.50 | 51.25 | 41.70 | 40.37 |
| Zero-Shot-Q | 56.88 | 52.71 | 35.71 | 45.10 | 36.54 | 53.76 | 41.40 | 29.70 |
| STE | 93.46 | 62.09 | 69.64 | 60.10 | 60.52 | 55.46 | 58.40 | 41.55 |
| HTGE | 94.72 | **70.76** | 76.79 | 61.30 | **64.42** | 57.83 | 60.30 | **54.79** |
| Uniform | **94.95** | 67.15 | 78.57 | **73.80** | **64.42** | **61.29** | **60.80** | 54.40 |
| Normal | 94.72 | **70.76** | **85.71** | 72.80 | 63.46 | 59.25 | 60.40 | 54.38 |

**Table 1(c)。** **Qwen3-8B** 在 4-bit 权重量化下不同代理梯度在下游任务上的性能比较。

| 方法 | SST-2 | RTE | CB | BoolQ | WSC | WiC | MultiRC | SQuAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-Shot | 55.50 | 84.48 | 66.07 | 80.20 | 64.42 | 63.01 | 85.50 | 34.50 |
| Zero-Shot-Q | 57.57 | 83.03 | 44.64 | 78.20 | 62.50 | 59.25 | 82.90 | 30.18 |
| STE | 87.96 | 83.39 | 73.21 | 81.80 | 72.12 | 70.38 | 83.40 | 56.88 |
| HTGE | 89.91 | 86.64 | 82.14 | **85.60** | 75.00 | 72.57 | 84.60 | 65.18 |
| Uniform | **92.89** | 88.81 | **89.29** | 85.40 | **76.92** | **73.98** | 87.20 | **70.29** |
| Normal | 91.17 | **90.25** | 82.14 | 85.10 | 72.12 | 71.00 | **88.60** | 64.21 |

**Table 8。** **LLaMA-2-13B** 在 4-bit 权重量化下不同代理梯度在下游任务上的性能比较。

| 方法 | SST-2 | RTE | CB | BoolQ | WSC | WiC | MultiRC | SQuAD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-Shot | 61.12 | 50.90 | 48.21 | 73.20 | 39.42 | 50.47 | 46.10 | 64.52 |
| Zero-Shot-Q | 58.03 | 45.13 | 53.57 | 70.90 | 45.19 | 50.31 | 46.10 | 55.62 |
| STE | 89.45 | 59.21 | 71.43 | 73.60 | 63.46 | 63.17 | 70.40 | 53.55 |
| HTGE | **91.74** | 77.98 | 73.21 | 82.10 | 71.15 | **68.81** | **81.60** | **63.88** |
| Uniform | 90.14 | 77.26 | **82.14** | **83.20** | 70.19 | 68.34 | 79.10 | 60.60 |
| Normal | 90.83 | **80.87** | 78.57 | 81.80 | **72.12** | 68.50 | 80.10 | 61.61 |

### W4A4KV4 量化与旋转

**Table 3。** Qwen2.5-0.5B 在不使用旋转时的 W4A4KV4 结果。

| 方法 | R-1 | R-2 | R-L | R-LSum | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| STE | 28.86 | 9.15 | 22.52 | 22.51 | 20.76 |
| Uniform | 30.33 | 10.18 | 23.72 | 23.73 | **21.99** |
| Normal | 30.32 | 10.00 | 23.56 | 23.57 | 21.86 |

**Table 4。** Qwen2.5-0.5B 在使用旋转时的 W4A4KV4 结果。

| 方法 | R-1 | R-2 | R-L | R-LSum | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| RoSTE | 29.50 | 9.60 | 22.74 | 22.75 | 21.15 |
| RoUniform | 31.87 | 11.30 | 24.98 | 24.98 | **23.28** |
| RoNormal | 31.27 | 10.59 | 23.87 | 23.88 | 22.40 |

## 致谢

量化初始化与模型集成基于 [OmniQuant](https://github.com/OpenGVLab/OmniQuant)。训练与模型加载使用 [Hugging Face Transformers](https://github.com/huggingface/transformers) 和 [Accelerate](https://github.com/huggingface/accelerate)。感谢上游作者与贡献者。
