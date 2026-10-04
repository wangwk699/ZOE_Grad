# ZOE Grad

统一运行五个模型上的八项任务与四种量化梯度估计方法。

- 模型：OPT 1.3B、OPT 6.7B、Llama-2 7B、Llama-2 13B、Qwen3 8B。
- 分类任务：SST2、RTE、CB、BoolQ、WSC、WIC、MultiRC。
- 生成任务：SQuAD。
- 梯度估计：STE、HTGE、Uniform、Normal。

## 环境

```bash
conda create -n ZOE python=3.10.19 -y
conda activate ZOE
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## 运行

在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run.sh opt-1.3b SST2 STE
CUDA_VISIBLE_DEVICES=0 scripts/run.sh llama2-7b SQuAD Normal
CUDA_VISIBLE_DEVICES=0 scripts/run.sh qwen3-8b WSC HTGE
```

模型别名：`opt-1.3b`、`opt-6.7b`、`llama2-7b`、`llama2-13b`、`qwen3-8b`。每个别名均可与上述八项任务、四种方法组合。分类任务使用选项概率；SQuAD 用答案 token 的 teacher forcing 损失训练，用生成结果的 F1 评估。默认权重精度为 W4A16，初始量化参数从 `pre_quantized_models/` 对应模型文件读取。所有任务统一使用 WikiText2 标定。

`run.sh` 通过环境变量覆盖常用设置；额外的 `train_main.py` 参数可以放在三个必选参数之后：

```bash
CUDA_VISIBLE_DEVICES=0 STEPS=256 LR=1e-6 NUM_TRAIN=64 scripts/run.sh qwen3-8b RTE Uniform --no_eval true
DRY_RUN=1 scripts/run.sh opt-6.7b SQuAD Normal
```

第一条命令会用 Qwen3 8B 在 RTE 任务上以 Uniform 方法训练。`scripts/run.sh` 前面的 `名称=值` 是仅对这一条命令生效的环境变量：

- `CUDA_VISIBLE_DEVICES=0`：只让程序使用编号为 0 的 GPU。
- `STEPS=256`：将最大训练步数设为 256（脚本默认是 5000）。
- `LR=1e-6`：将学习率设为 0.000001。
- `NUM_TRAIN=64`：抽取 64 条训练样本。

命令末尾的 `--no_eval true` 是传给 `train_main.py` 的额外参数，表示跳过训练结束后的评估。

第二条命令指定 OPT 6.7B、SQuAD 任务和 Normal 方法。`DRY_RUN=1` 表示只打印即将执行的 `python train_main.py ...` 命令，不加载模型或启动训练，便于先检查参数；脚本仍会检查预量化参数文件是否存在。前置环境变量不会永久改变终端设置，也不会自动作用于下一条命令。

支持的环境变量包括 `STEPS`、`LR`、`BATCH_SIZE`、`WBITS`、`ABITS`、`RESUME_PATH`、`NUM_TRAIN`、`NUM_EVAL`、`NUM_DEV`、`DELTA_OVERRIDE`、`T`、`OUTPUT_DIR`、`HF_HOME`。

`DRY_RUN=1 scripts/run_matrix.sh` 生成全部 5 × 8 × 4 = 160 个命令，不启动训练。直接运行 `CUDA_VISIBLE_DEVICES=0 scripts/run_matrix.sh` 会逐个执行所有组合。

## 本地实验文件

正式开始实验前，请创建以下三个目录并放入所运行模型需要的文件；缺少预量化参数会使所有模型的实验无法启动，缺少激活统计文件会使默认启用 LET 的 OPT 实验无法正常运行。

- `pre_quantized_models/` 存放已量化模型的参数文件。`scripts/run.sh` 默认读取所选模型对应的 `*-w4a16.pth`，以该量化结果作为训练起点。
- `act_scales/`、`act_shifts/` 存放 OPT 模型的激活统计。默认脚本为 OPT 1.3B、6.7B 启用 LET，量化时会读取对应模型的两个 `.pt` 文件；Llama-2 7B、13B 和 Qwen3-8B 默认不启用 LET，不会读取这两个目录。所有任务统一使用 WikiText2 标定数据。

预量化参数和激活统计文件可从 [OmniQuant](https://github.com/OpenGVLab/OmniQuant) 获取；其未提供的模型文件需另行准备。
