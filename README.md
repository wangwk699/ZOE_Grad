# ZOE Grad

A unified way to run five models across eight tasks with four quantized gradient estimator methods.

- **Models:** OPT 1.3B and 6.7B; Llama-2 7B and 13B; Qwen3 8B.
- **Classification tasks:** SST2, RTE, CB, BoolQ, WSC, WIC, and MultiRC.
- **Generative task:** SQuAD.
- **Gradient estimators:** STE, HTGE, Uniform, and Normal.

## Environment

```bash
conda create -n ZOE python=3.10.19 -y
conda activate ZOE
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Run

Run the following commands from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run.sh opt-1.3b SST2 STE
CUDA_VISIBLE_DEVICES=0 scripts/run.sh llama2-7b SQuAD Normal
CUDA_VISIBLE_DEVICES=0 scripts/run.sh qwen3-8b WSC HTGE
```

Supported model aliases are `opt-1.3b`, `opt-6.7b`, `llama2-7b`, `llama2-13b`, and `qwen3-8b`. Each model can be paired with any of the eight tasks and four estimators listed above. Classification tasks score candidate answers by their model probabilities. SQuAD uses teacher-forced loss on answer tokens for training and F1 on generated answers for evaluation.

The default quantization configuration is W4A16. Each run loads its initial quantization parameters from the corresponding file in `pre_quantized_models/`. All tasks use WikiText2 data for quantization calibration.

`run.sh` accepts environment variables to override common settings; additional `train_main.py` options can be placed after the three required arguments:

```bash
CUDA_VISIBLE_DEVICES=0 STEPS=256 LR=1e-6 NUM_TRAIN=64 scripts/run.sh qwen3-8b RTE Uniform --no_eval true
DRY_RUN=1 scripts/run.sh opt-6.7b SQuAD Normal
```

The first command trains Qwen3 8B on RTE using Uniform gradient estimator. The assignments before `scripts/run.sh` apply only to this invocation:

- `CUDA_VISIBLE_DEVICES=0` makes GPU 0 available to the process.
- `STEPS=256` sets the maximum number of training steps to 256 instead of the default 5,000.
- `LR=1e-6` sets the learning rate to 0.000001.
- `NUM_TRAIN=64` selects 64 training examples.

The trailing `--no_eval true` is passed to `train_main.py` and skips the evaluation performed after training.

The second command runs OPT 6.7B on SQuAD with the Normal gradient estimator. With `DRY_RUN=1`, the script prints the resulting `python train_main.py ...` command without loading a model or starting training. It still checks that the required OmniQuant parameter checkpoint exists. Environment assignments do not persist in the shell or affect subsequent commands.

Supported environment variables include `STEPS`, `LR`, `BATCH_SIZE`, `WBITS`, `ABITS`, `RESUME_PATH`, `NUM_TRAIN`, `NUM_EVAL`, `NUM_DEV`, `DELTA_OVERRIDE`, `T`, `OUTPUT_DIR`, and `HF_HOME`.

`DRY_RUN=1 scripts/run_matrix.sh` prints all 5 × 8 × 4 = 160 model–task–estimator combinations without training. To run them sequentially, use `CUDA_VISIBLE_DEVICES=0 scripts/run_matrix.sh`.

## Required Local Files

Before starting experiments, create the three directories below and place the files required by the selected model in them. Missing the required OmniQuant parameter checkpoint causes any run to fail with an error; default OPT runs with LET also fail with an error if their activation statistics are missing.

- `pre_quantized_models/` stores per-layer OmniQuant parameter checkpoints. During quantization, `scripts/run.sh` loads the selected model's `*-w4a16.pth` file before downstream training.
- `act_scales/` and `act_shifts/` store activation statistics. The default script enables learnable equivalent transformation (LET) for OPT 1.3B and 6.7B and loads their corresponding `.pt` files. LET is disabled by default for Llama-2 7B and 13B and Qwen3 8B, so those runs do not read these directories. All tasks use WikiText2 calibration data.

OmniQuant parameter checkpoints and activation statistics can be obtained from [OmniQuant](https://github.com/OpenGVLab/OmniQuant); model files unavailable there must be obtained by training them independently.
