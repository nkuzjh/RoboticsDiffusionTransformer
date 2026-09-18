# CSGO Benchmark v2 Seen-10 接入方案

- 数据：新增 `data/csgo_seen10.py`，严格读取 `minimal_dataset_report.json`、`benchmark_manifest.json`、发布 calibration 及 `splits/seen/<map>/{train,validation,discrete_test}.json`；固定 Seen-10 地图顺序和 `map/file_frame` identity，返回 FPV/radar、归一化 5DoF、物理 metadata 与可选每地图预计算语言 embedding。
- 模型适配：VLA localization only；两张经原生 SigLIP processor 的 aspect-pad 图像，zero/masked state，horizon=1 的 128 维 action 中前 5 维为 `[x,y,z,pitch,yaw]`，prediction JSONL 使用同一归一化空间。
- 训练：在 `train/train.py` 保留原 RDT 构造、optimizer、checkpoint 与训练循环，由 `train_seen10.py`/相关小分支接入 Seen10 dataset/collator；`configs/csgo_seen10.yaml` 和 `scripts/run_csgo_seen10.sh` 提供 train、infer、eval、`--seed` 命令。
- 推理/可视化：`infer_seen10.py` 生成标准 localization JSONL；`train/csgo_visualize.py` 固定 seed 按地图抽样 10 个样本，输出雷达 GT/预测叠加与右侧 FPV 及物理 `gt_xyzhw/pred_xyzhw`。
- 验收：独立环境完成 dataset batch、一次 forward/backward、checkpoint load、标准预测和 shared evaluator smoke；`RUN_FULL=0` 时不执行完整训练，DATA_ROOT 只读且不扫描图像目录。
