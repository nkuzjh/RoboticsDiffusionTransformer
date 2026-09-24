# RDT 接入 CSGO Benchmark v2 Seen-10

本文是 RDT 的 Seen-10 定位任务运行说明。命令从项目根目录 `/home/jiahao/task/RoboticsDiffusionTransformer` 执行。实验设计、实现细节和验收证据见 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)；本文记录使用者需要的环境、命令、输出和当前结果。正式训练、全量推理及评测均由使用者手动启动。

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

项目使用 `.venv`。新环境可按下列命令准备；脚本会在需要时克隆本机环境并安装缺失依赖，资产脚本将官方 RDT、SigLIP 和 T5 缓存在 `.cache/csgo_seen10/models/`，支持中断后继续下载。

```bash
cd /home/jiahao/task/RoboticsDiffusionTransformer
bash scripts/setup_csgo_seen10.sh
.venv/bin/python scripts/prepare_csgo_assets.py
```

本机上述三组官方资产已缓存。aligned 从原始 `rdt-1b` 初始化，不从已有 CSGO checkpoint 热启动。`--dry-run` 只检查并打印解析后的参数，不构建模型，也不创建训练输出：

```bash
.venv/bin/python train_seen10.py train \
  --config configs/csgo_seen10_aligned.yaml --dry-run
```

## 3. 正式执行与恢复

### aligned：当前主实验

配置固定 seed 42、单 GPU microbatch 4 × 累计 32，得到有效 batch 128。训练共 19,500 optimizer updates，即 2,496,000 次样本暴露；验证与保存固定在 **4,000、8,000、12,000、16,000、19,500**。`best` 指向 validation 归一化有效 5D MSE 最低的 checkpoint，`late` 指向最新保存点，完成后为 `checkpoint-19500`；没有 `last` 别名。主比较使用完成后的 `late`，不能根据测试集挑选 checkpoint。

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

恢复会核对保存的执行约定、optimizer、scheduler、随机数与采样进度；不能随意改变已有运行的进程数或 batch 划分。新运行使用多 GPU 时须保持 `GPU 数 × 每卡 microbatch × 累计步数 = 128`，并重新核对启动参数。正式测试推理固定 seed 42、单进程、batch 1 和 manifest 顺序；不要改用多进程或较大 batch 的加速命令。aligned 的部分预测文件不能按 ID 跳过后续跑，若推理中断，应使用新的结果目录从第一条重新推理，并在评测时指定同一目录。例如：

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
/home/jiahao/miniconda3/envs/UniLIP/bin/python \
  /home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py localization \
  --pred-root outputs/csgo_benchmark_v2_seen10/RDT/seed_0/localization \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
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
  --calibration /home/jiahao/task/UniLIP/data/csgo_benchmark_v2/calibration/z_calibration.json \
  --source-z-epsilon 1e-6
```

随后将共享评测命令的 `--pred-root` 指向导出的 `localization` 目录，`--output` 指向新的评测目录。此工具已实现，但尚未据此完成 UniLIP 的统一重评。

## 5. 当前结果与执行状态

legacy `seed_0` 已完成 1,000 updates；`best` 和 `late` 均指向 `checkpoint-1000`。现有 `localization/predictions.jsonl` 有 20,000 条，官方 [equal-map 摘要](outputs/csgo_benchmark_v2_seen10/RDT/seed_0/evaluation/localization/summary_equal_map.json) 给出 XY **208.0951**、Z **10.9845**、Pitch **7.7643°**、Yaw **84.0743°**。这是首次接入配置的历史结果；当时实际推理的 batch 大小与进程数没有可核实记录，不能据此声称其与 aligned 推理口径一致。

aligned `seed_42` 的正式训练、20,000 条测试推理和正式评测均尚未启动，也没有正式性能指标。当前完成的是 22 项自动检查、CPU 小模型单/双进程训练与恢复、单进程推理 smoke，以及官方 1B 权重加载、参数和哈希审计；**官方 1B 模型未做 GPU 前后向**。这些检查及具体证据见 [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)，不能将 smoke 指标当作 Seen-10 正式结果。
