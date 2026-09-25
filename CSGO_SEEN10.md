# RDT 接入 CSGO Benchmark v2 Seen-10

本文是 RDT 的 Seen-10 定位任务运行说明。命令从当前服务器的 RDT 项目根目录执行。实验设计、实现细节和验收证据见 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)；本文记录使用者需要的环境、命令、输出和当前结果。正式训练、全量推理及评测均由使用者手动启动。

legacy 与 aligned 的运行说明统一维护于本文；aligned 的验收记录统一维护于 PLAN 第 9 节，不再保留独立的 aligned 文档。

## 1. 实验范围与数据

只使用 CSGO Benchmark v2 的 Seen-10 定位任务，不含生成任务或 CrossMap。官方数据位于 `/home/jiahao/task/UniLIP/data/csgo_benchmark_v2`，共享评测器位于 `/home/jiahao/task/csgo_benchmark_v2_eval_general`。固定的十张地图为 `cs_agency`、`cs_italy`、`de_ancient`、`de_anubis`、`de_dust2`、`de_inferno`、`de_mirage`、`de_nuke`、`de_overpass`、`de_train`。发布划分包含 50,000 条 `seen_train`、5,000 条 `seen_validation` 和 20,000 条 `seen_discrete_test`，测试集每张地图 2,000 条。

模型输入为当前 FPV RGB、对应地图 radar 和包含地图名的定位指令；不输入 GT 坐标、历史位姿或真实机器人状态。对外预测单步 5D pose `[x, y, z, pitch, yaw]`。预测文件保存归一化值，官方评测器反归一化后计算 XY、Z、Pitch、Yaw 误差，数值越低越好。

本项目保留两套独立配置：

| 实验 | 配置 | seed | checkpoint 目录 | 结果目录 | 主结果选择 |
| --- | --- | ---: | --- | --- | --- |
| 首次接入 legacy | `configs/csgo_seen10.yaml` | 0 | `checkpoints/csgo_benchmark_v2_seen10/RDT/seed_0` | `outputs/csgo_benchmark_v2_seen10/RDT/seed_0` | validation `best` |
| 公平对比 aligned | `configs/csgo_seen10_aligned.yaml` | 42 | `checkpoints/csgo_aligned_aug_v1/RDT/seed_42` | `outputs/csgo_aligned_aug_v1/RDT/seed_42` | 完成后的 `late` |

legacy 保留最初接入的 1,000 updates、有效 batch 4、只监督 5D 的 clean-action loss、无图像增强，以及推理各步对无效 action 维度的旧处理。aligned 使用有效 batch 128、19,500 updates 和完整 128D 原生扩散训练与五步 DPM-Solver 推理；5D pose 后的 123 维零目标仍进入完整 128D 加噪及主 loss，最后才裁剪为 5D 预测。结构性的全零 state 槽位仍参与计算，不代表额外输入状态。aligned 冻结 SigLIP-384 和 T5，训练 RDT LoRA 及适配层，共 73,164,928 个可训练参数；仅训练图像按视角独立以 0.5 概率启用原生颜色、噪声与模糊增强。两套设置的目标函数和训练量不同，旧结果只能作为首次接入的历史结果。

aligned 与 UniLIP `exp32_loc` 对齐官方 split、5D 目标、样本暴露量、有效 batch 和评测口径，保留 RDT 原生模型及求解器差异。完整对齐依据与比较边界见 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)。

## 2. 环境与权重

项目使用项目内的 `.venv`，不要求安装 ControlAR，也不要求存在 `/home/jiahao`。准备脚本保留已有环境，缺少环境时可独立创建；资产脚本将官方 RDT、SigLIP 和 T5 缓存在当前项目的 `.cache/csgo_seen10/models/`，支持中断后继续下载。下面的 `cd` 按实际 clone 位置调整：

```bash
cd ~/task/RoboticsDiffusionTransformer
bash scripts/setup_csgo_seen10.sh
.venv/bin/python scripts/prepare_csgo_assets.py
bash scripts/setup_csgo_seen10.sh --check
```

环境脚本优先复用已有 `.venv`；新建时使用 Python 3.11 的 venv，若本机没有 Python 3.11 则通过 conda 创建。可用 `RDT_SETUP_PYTHON` 指定 Python 3.11，或显式设置 `RDT_CLONE_SOURCE` 克隆已有 conda 环境。已存在的环境不会因克隆源缺失而失败。

新环境的 PyTorch/torchvision 按 `nvidia-smi` 报告的 CUDA 版本选择成对的 wheel；可用 `RDT_TORCH_BACKEND` 指定后端，具体选项见 `--help`。没有可见 GPU/驱动时自动选择 CPU；在不显示 GPU 的登录节点上准备训练环境，应显式指定计算节点支持的 CUDA 后端。`RDT_TORCH_INDEX_URL` 可指定对应 wheel 源，其他 Python 依赖使用 pip 自身的源配置。

另一台服务器已确认是 A100、驱动 580.125.09、`nvidia-smi` 显示 CUDA 13.0；新建环境可明确使用下面的 cu128 配置。它安装 PyTorch 2.8.0 / torchvision 0.23.0 的 CUDA 12.8 wheel，CUDA 13.0 的显示值不是必须安装的 wheel 版本。配对依据 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/)，驱动向后兼容依据 [NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

```bash
cd ~/task/RoboticsDiffusionTransformer

RDT_TORCH_BACKEND=cu128 bash scripts/setup_csgo_seen10.sh
bash scripts/setup_csgo_seen10.sh --check

.venv/bin/python scripts/prepare_csgo_assets.py
bash scripts/setup_csgo_seen10.sh --check
```

```bash
bash scripts/setup_csgo_seen10.sh --dry-run  # 只显示准备方案
bash scripts/setup_csgo_seen10.sh --check    # 检查已有环境，不安装或升级包
```

全新环境使用 [requirements_csgo.txt](requirements_csgo.txt) 中的兼容版本，包括 NumPy 1.26.4；原生 `imgaug` 不兼容 NumPy 2.x。已有环境保留已安装版本并检查实际导入，发现冲突时报告错误。定位流程不要求安装 DeepSpeed、TensorFlow 或生成评测依赖。准备脚本不下载模型；上面的资产准备命令才会下载或校验官方权重。

OpenCV 是上述“保留已有版本”的例外：准备脚本在所有依赖安装结束后，统一为单一的 `opencv-python-headless==4.11.0.86`。`imgaug` 的包依赖声明会拉入 GUI 版 `opencv-python`，而 OpenCV 的四种发行包共用 `cv2` 文件；混装可能导致无图形界面的服务器报 `libGL.so.1` 缺失。脚本会先卸载冲突的 OpenCV 包，再以 `--no-deps --force-reinstall` 安装固定 headless 版，避免卸载共用文件后留下损坏的 `cv2`，也避免改变 NumPy/PyTorch。已正确安装时重复执行不会重装；`--check` 只诊断，不修复。此处理遵循 [OpenCV 的单一发行包安装说明](https://pypi.org/project/opencv-python-headless/4.11.0.86/)。

遇到 `imgaug`/`cv2` 导入时报 `libGL.so.1`，同步代码后重跑准备脚本即可；无需安装系统 GUI 库或使用 sudo：

```bash
bash scripts/setup_csgo_seen10.sh
bash scripts/setup_csgo_seen10.sh --check
```

以后再次用 pip 安装 `imgaug` 或其关联依赖，可能重新拉入 GUI 版，此时也应重跑准备脚本。`pip check` 可能仍报告 imgaug 声明的 `opencv-python` 未安装，这是两个 OpenCV 发行包的名称不互相替代所致；不要为了消除这一元数据提示再次混装，运行检查应确认 `cv2` 与 `imgaug` 均可导入且仅保留 headless 版。

T5 tokenizer 还需要 `sentencepiece` 和 `protobuf`（导入名为 `google.protobuf`），两者均已纳入依赖。若旧版环境在训练启动时报 `requires the protobuf library but it was not found`，同步代码后重新执行 `bash scripts/setup_csgo_seen10.sh` 即可补装，无需重建 `.venv`。`--check` 会检查 SentencePiece protobuf schema；官方 T5 缓存存在时，还会仅用本地文件加载 tokenizer 并编码文本，不加载模型权重、不联网。资产尚未下载时会明确显示 tokenizer 检查跳过，因此应在资产准备完成后再执行一次 `--check`。

当前服务器已有 PyTorch nightly 环境会保留，新服务器默认采用上述稳定版，因此不承诺两台机器的浮点结果逐位一致；实验报告应保留各自 `--check` 输出及推理 provenance 中的依赖版本。

### 通用评测器的独立环境

RDT 与 OpenVLA 使用相同的评测解释器优先级：

```text
--eval-python（兼容 --unilip-python）
  > CSGO_EVAL_PYTHON
  > UNILIP_PYTHON
  > YAML unilip_python
  > <实际 shared_eval_dir>/.venv/bin/python
```

两份当前实验 YAML 均设 `unilip_python: null`，默认使用通用评测器自己的环境。`--eval-root` 或 `SHARED_EVAL_DIR` 改变评测器目录时，默认解释器随之改变。`--python` 仍指定 RDT 的启动解释器，不覆盖评测环境。显式 Python 路径不做存在性探测或回退；未安装时直接启动失败。RDT 的 eval 不安装、检查或修复评测环境。

同步完整评测器目录后，定位任务可单独初始化一次环境，跳过生成指标权重下载：

```bash
cd ~/task/csgo_benchmark_v2_eval_general
bash setup_env.sh --skip-weights
cd ~/task/RoboticsDiffusionTransformer

# 取消此前指向 RDT/UniLIP 的覆盖，使用统一默认环境
unset CSGO_EVAL_PYTHON UNILIP_PYTHON
bash scripts/run_csgo_seen10.sh eval \
  --config configs/csgo_seen10_aligned.yaml --print-paths
```

查看输出的 `evaluator_python`（保留的别名字段 `unilip_python` 值相同）。训练、训练内 validation 和测试集推理继续使用 RDT 环境；通用评测器的依赖与安装清单由评测器项目维护。

**旧训练恢复兼容性：** aligned checkpoint 校验完整配置哈希，本次把 YAML 的评测环境字段改为 null 也会改变该哈希。已开始的旧运行恢复时应使用启动时的原配置；评测可在原配置基础上用 `--eval-python` 或 `CSGO_EVAL_PYTHON` 指定统一环境，无需改动原训练配置。没有放宽恢复检查或改写历史产物。

### 两台服务器的路径配置

推荐在两台服务器均使用同级目录布局：

```text
task/
  RoboticsDiffusionTransformer/
  UniLIP/data/csgo_benchmark_v2/
  csgo_benchmark_v2_eval_general/run_eval.py
```

数据和评测器目录按显式 CLI → 环境变量 → YAML 解析。原 YAML 中本机的默认数据/评测器目录仍存在时沿用；若这些旧默认目录在另一台服务器不存在，自动使用上述同级目录。自定义的错误路径不会被静默替换。评测 Python 独立遵循上一节的优先级，旧 YAML 显式指定的 UniLIP 路径也不会自动回退。

若实际布局不同，在当前 shell 设置以下变量；训练、推理及 wrapper 评测会使用相同数据路径：

```bash
export CSGO_DATA_ROOT=/actual/path/to/csgo_benchmark_v2
export SHARED_EVAL_DIR=/actual/path/to/csgo_benchmark_v2_eval_general
export CSGO_EVAL_PYTHON="$SHARED_EVAL_DIR/.venv/bin/python"
```

也可使用 `--data-root`、`--eval-root`、`--eval-python`（兼容 `--unilip-python`）显式覆盖。旧别名 `DATA_ROOT`、`CSGO_EVAL_ROOT` 仍支持；不要同时设置相互冲突的别名。只查看实际解析结果、不执行训练或评测：

```bash
bash scripts/run_csgo_seen10.sh train \
  --config configs/csgo_seen10_aligned.yaml --print-paths
```

Git 不同步 `.venv`、权重缓存、benchmark 数据或共享评测器；另一台服务器需要单独准备这些内容。若复制已有权重缓存，仍需在目标服务器运行资产准备脚本以校验文件并刷新 manifest 中的本地路径。此变更支持在另一台服务器新建同配方实验，不放宽旧 checkpoint 的跨路径恢复一致性检查。

本机上述三组官方资产已缓存。aligned 从原始 `rdt-1b` 初始化，不从已有 CSGO checkpoint 热启动。`--dry-run` 只检查并打印解析后的参数，不构建模型，也不创建训练输出：

```bash
.venv/bin/python train_seen10.py train \
  --config configs/csgo_seen10_aligned.yaml --dry-run
```

## 3. 正式执行与恢复

### aligned：当前主实验

配置固定 seed 42，当前单 GPU microbatch 32 × 累计 4，得到有效 batch 128。新运行可以调整两者，只要求 `GPU 数 × 每卡 microbatch × 累计步数 = 128`，各项均为正整数。当前 validation batch 为 32，可通过 `training.eval_batch_size` 调整。训练共 19,500 optimizer updates，即 2,496,000 次样本暴露；验证与保存固定在 **4,000、8,000、12,000、16,000、19,500**。`best` 指向 validation 归一化有效 5D MSE 最低的 checkpoint，`late` 指向最新保存点，完成后为 `checkpoint-19500`；没有 `last` 别名。主比较使用完成后的 `late`，不能根据测试集挑选 checkpoint。

依次手动执行训练、推理、评测：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh train \
  --config configs/csgo_seen10_aligned.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer \
  --config configs/csgo_seen10_aligned.yaml
bash scripts/run_csgo_seen10.sh eval \
  --config configs/csgo_seen10_aligned.yaml
```

训练从已经完整保存的最新 checkpoint 恢复时，用相同的配置与输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train_seen10.py train \
  --config configs/csgo_seen10_aligned.yaml \
  --resume-from-checkpoint latest
```

恢复会核对保存的执行约定、optimizer、scheduler、随机数与采样进度；已有 checkpoint 仍须使用原配置和原 batch 划分，即使修改后的有效 batch 仍为 128，也不能绕过严格恢复检查。正式测试推理保持 seed 42、单进程和 manifest 顺序；batch 默认为 1，可通过 `inference.batch_size` 或 wrapper 的 `-- --batch-size 32` 指定任意正整数，不受训练有效 batch 128 的限制。实际 batch 会写入 provenance；不同 batch 可能改变随机采样与数值结果，不能根据 test 指标选择 batch。改变 batch 时使用新的结果目录，不能混用已有预测。

aligned 的部分预测文件不能按 ID 跳过后续跑，若推理中断，应使用新的结果目录从第一条重新推理，并在评测时指定同一目录。例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_csgo_seen10.sh infer \
  --config configs/csgo_seen10_aligned.yaml \
  --output-dir outputs/csgo_aligned_aug_v1/RDT/seed_42_infer_rerun
bash scripts/run_csgo_seen10.sh eval \
  --config configs/csgo_seen10_aligned.yaml \
  --output-dir outputs/csgo_aligned_aug_v1/RDT/seed_42_infer_rerun
```

已有完整预测可由入口核对 provenance；不要在同一目录混用不同 checkpoint。

### legacy：复现已有接入

legacy 配置默认 seed 0。`seed_0` 已有正式训练和评测产物，训练入口会拒绝覆盖非空目录。以下以新 seed 1 展示一套独立运行命令；同一个 seed 必须贯穿训练、推理和评测：

```bash
bash scripts/run_csgo_seen10.sh train --seed 1
bash scripts/run_csgo_seen10.sh infer --seed 1
bash scripts/run_csgo_seen10.sh eval --seed 1
```

legacy 默认推理取 validation `best`，默认 batch 来自配置的 `training.eval_batch_size=4`。`scripts/run_csgo_seen10.sh` 的模式专用参数可放在 `--` 后，例如 `infer --seed 1 -- --batch-size 4`。已有预测按样本 identity 续写，`inference_provenance.json` 会核对 checkpoint 与 seed。若仅重新评测现有 seed 0 的 20,000 条预测，应直接调用共享评测器并选择未使用的输出目录：

```bash
"${CSGO_EVAL_PYTHON:-${UNILIP_PYTHON:-${SHARED_EVAL_DIR:-../csgo_benchmark_v2_eval_general}/.venv/bin/python}}" \
  "${SHARED_EVAL_DIR:-../csgo_benchmark_v2_eval_general}/run_eval.py" localization \
  --pred-root outputs/csgo_benchmark_v2_seen10/RDT/seed_0/localization \
  --data-root "${CSGO_DATA_ROOT:-../UniLIP/data/csgo_benchmark_v2}" \
  --pose-space normalized \
  --output outputs/csgo_benchmark_v2_seen10/RDT/seed_0/evaluation/localization_recheck
```

共享评测器拒绝写入已有非空结果目录；重复评测时更换上述 `--output`。legacy 和 aligned 的输出根目录分别由各自 YAML 决定；自定义目录时，训练所用 `--checkpoint-dir`、`--output-dir` 要在后续推理中保持一致，评测使用相同 `--output-dir`。wrapper 的模式名必须放在第一位，`--config`、`--seed`、`--checkpoint-dir`、`--output-dir` 等公共参数放在模式名之后、`--` 之前；训练和推理的额外参数在 `--` 后传入。

## 4. 输出、可视化与指标

每个正式 seed 的 checkpoint 根目录下保存 `checkpoint-<step>` 和 `best`/`late` 相对符号链接；aligned 只保存上述五个正式节点。对应结果目录包含：

```text
training_metadata.json                 训练配置与完成状态
train_loss.jsonl、loss_curve.svg       逐 update loss 与曲线
validation/step-<step>/               validation 预测、指标与地图可视化
localization/predictions.jsonl         测试集归一化 5D 预测
localization/inference_provenance.json 推理 checkpoint、seed 等来源信息
localization/visualization/            正式推理的地图可视化
evaluation/localization/              官方评测输出
```

可视化每张地图用固定 seed 选 10 条样本：左侧 radar 以相同颜色标出 GT 实心点、预测空心点与连线，右侧排列对应 FPV，并显示物理空间的 `gt_xyzhw` 与 `pred_xyzhw`。测试集 GT 仅在预测完成后用于可视化，不进入模型输入或标准预测文件。最终指标以 `evaluation/localization/summary_equal_map.json` 的 equal-map macro XY、Z、Pitch、Yaw 为准。XY/Z 使用 benchmark 坐标单位，角度使用度，不将坐标单位标成米。完整测试每张地图恰好 2,000 条，因此这些样本均值指标的 pooled 与 equal-map macro 相等；评测文件没有独立的 pooled 字段。

### UniLIP 预测的统一评测格式

RDT 使用共享评测器的无 epsilon Z 归一化，UniLIP `exp32_loc`/`exp32` 的分母包含 `1e-6`。统一重评已有 UniLIP 预测时，先用下列工具保留物理 Z 并转换格式；`--input` 替换为实际包含 `pred_norm` 的 JSONL，输出路径必须未使用。工具只读取预测和样本身份，不读取 GT、不 clamp；转换公式见 PLAN 第 2 节。

```bash
.venv/bin/python scripts/export_unilip_seen_predictions.py \
  --input /path/to/unilip_predictions.jsonl \
  --output outputs/unilip_exp32_loc_export/localization/predictions.jsonl \
  --calibration "${CSGO_DATA_ROOT:-../UniLIP/data/csgo_benchmark_v2}/calibration/z_calibration.json" \
  --source-z-epsilon 1e-6
```

随后将共享评测命令的 `--pred-root` 指向导出的 `localization` 目录，`--output` 指向新的评测目录。此工具已实现，但尚未据此完成 UniLIP 的统一重评。

## 5. 当前结果与执行状态

legacy `seed_0` 已完成 1,000 updates；`best` 和 `late` 均指向 `checkpoint-1000`。现有 `localization/predictions.jsonl` 有 20,000 条，官方 [equal-map 摘要](outputs/csgo_benchmark_v2_seen10/RDT/seed_0/evaluation/localization/summary_equal_map.json) 给出 XY **208.0951**、Z **10.9845**、Pitch **7.7643°**、Yaw **84.0743°**。这是首次接入配置的历史结果；当时实际推理的 batch 大小与进程数没有可核实记录，不能据此声称其与 aligned 推理口径一致。

aligned `seed_42` 的正式训练、20,000 条测试推理和正式评测均尚未启动，也没有正式性能指标。当前完成的是 22 项自动检查、CPU 小模型单/双进程训练与恢复、单进程推理 smoke，以及官方 1B 权重加载、参数和哈希审计；**官方 1B 模型未做 GPU 前后向**。这些检查及具体证据见 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)，不能将 smoke 指标当作 Seen-10 正式结果。
