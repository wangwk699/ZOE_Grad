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
scripts/run.sh opt-1.3b SST2 STE
scripts/run.sh llama2-7b SQuAD Normal
scripts/run.sh qwen3-8b MultiRC HTGE
```

模型别名：`opt-1.3b`、`opt-6.7b`、`llama2-7b`、`llama2-13b`、`qwen3-8b`。每个别名均可与上述八项任务、四种方法组合。分类任务使用选项概率；SQuAD 用答案 token 的 teacher forcing 损失训练，用生成结果的 F1 评估。默认权重精度为 W4A16，初始量化参数从 `pre_quantized_models/` 对应模型文件读取。所有任务统一使用 WikiText2 标定。

`run.sh` 通过环境变量覆盖常用设置；额外的 `train_main.py` 参数可以放在三个必选参数之后：

```bash
CUDA_VISIBLE_DEVICES=0 STEPS=256 LR=1e-6 NUM_TRAIN=64 scripts/run.sh qwen3-8b RTE Uniform --no_eval true
DRY_RUN=1 scripts/run.sh opt-6.7b SQuAD Normal
```

支持的环境变量包括 `STEPS`、`LR`、`BATCH_SIZE`、`WBITS`、`ABITS`、`RESUME_PATH`、`NUM_TRAIN`、`NUM_EVAL`、`NUM_DEV`、`DELTA_OVERRIDE`、`T`、`OUTPUT_DIR`、`HF_HOME`。

`DRY_RUN=1 scripts/run_matrix.sh` 生成全部 5 × 8 × 4 = 160 个命令，不启动训练。直接运行 `scripts/run_matrix.sh` 会逐个执行所有组合。

## 本地实验文件

- `pre_quantized_models/` 存放运行所需的预量化参数及原有其他位宽的 checkpoint。
- `act_scales/`、`act_shifts/` 存放 OPT/Llama-2 的激活统计，所有任务使用同一套 WikiText2 标定文件。

预量化参数和激活统计文件可从 [OmniQuant](https://github.com/OpenGVLab/OmniQuant) 获取；其未提供的模型文件需另行准备。
