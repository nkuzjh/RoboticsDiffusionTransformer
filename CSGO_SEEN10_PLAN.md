# RDT 接入 CSGO Benchmark v2 Seen-10：实现方案与验收记录

本文面向维护者，记录当前实现的设计依据、技术协议、文件职责、兼容边界及验收证据。
环境准备、正式训练/推理/评测命令、输出目录和已有正式结果统一见 [CSGO_SEEN10.md](CSGO_SEEN10.md)。
本文统一覆盖 legacy 与 aligned 的实现方案；aligned 验收表格、证据索引及未确认项集中于第 9 节。

当前 aligned profile 为 `aligned_native_aug_v1`，已经实现并完成隔离验收，尚未正式训练。
legacy 已有正式训练和完整测试结果；“本轮未启动正式实验”不等于“项目从未运行过正式实验”。
此前文档整理没有改变代码、配置或实验策略；后续跨服务器准备脚本变更及其验收见第 9.5 节。

## 1. 范围、比较对象与设计依据

只覆盖 Seen-10 localization-only，不包括 CrossMap-4、生成任务或 exp32_gen。
主要比较 UniLIP exp32_loc 的 final19500；exp32 的定位结果作为多任务训练的次要参考。

| 分类 | 依据 | 本项目的处理 |
| --- | --- | --- |
| A：必须对齐 | 发布 benchmark、通用 evaluator、参考实验实际产物 | 数据/ID/split、输入信息边界、外部 5D pose、定位预算、checkpoint 选择与评测 |
| B：保留原生 | RDT 作者原始实现与官方 base | 内部128D、完整噪声/loss/采样、clean-action objective、SigLIP384、冻结 T5、求解器和图像增强操作 |
| C：任务适配 | A/B 接口缺口及当前架构 | 5D 通道映射、零 state 槽、LoRA/角色 LR、逐视图增强、全局 update sampler 和恢复约束 |

“原生”依据 [RDTRunner](models/rdt_runner.py)、[RDT 网络](models/rdt/model.py)、
[原始数据处理](train/dataset.py) 和官方权重；首次 CSGO 子类的 valid5 扩散方式不属于原生。
不机械复制其他 VLA 的内部维度、模块名字或优化策略；不新增 LR sweep、参数量匹配、qnorm、state 或分辨率消融。

## 2. 数据、输入与外部坐标协议

数据根为 `/home/jiahao/task/UniLIP/data/csgo_benchmark_v2`。读取发布 manifest、minimal report、
逐地图 calibration 与 split JSON，不扫描图像目录重建划分。十张地图及顺序固定为
[配置](configs/csgo_seen10_aligned.yaml) 和 `data.csgo_seen10.SEEN_MAPS` 中的 Seen-10。

| Split | 每地图 | 总数 | 用途 |
| --- | ---: | ---: | --- |
| seen_train | 5,000 | 50,000 | 训练 |
| seen_validation | 500 | 5,000 | 验证选择 |
| seen_discrete_test | 2,000 | 20,000 | 最终评测 |

模型输入只有当前 FPV RGB、对应地图 radar 和包含地图名的定位指令。
不输入 GT 坐标、历史位姿、真实 robot state 或测试集统计。身份为 map/file_frame；metadata 与可视化 GT 不作为模型条件。
外部 action 为单步 `(B,1,5)`，顺序 `[x,y,z,pitch,yaw]`。

```text
x'     = x / 1024
y'     = y / 1024
z'     = (z - z_min_map) / (z_max_map - z_min_map)
pitch' = angle_v / (2*pi)     # 原始角度为弧度
yaw'   = angle_h / (2*pi)
```

反归一化：XY 乘1024；Z 乘 span 再加 z_min；角度乘360得到度。
XYZ 指标为 benchmark 坐标单位，不擅自标成米。三 split 共用发布标定：
approved full corpus 在划分前计算的逐地图 exact min/max；不重新计算测试范围。
不新增 action qnorm，不 clamp，不添加 Z epsilon；benchmark 抽样 quantile bins 不是 action normalization。

通用 evaluator 使用无 epsilon 的 Z/span，UniLIP exp32_loc/exp32 使用 Z/(span+1e-6)。
RDT 遵循 evaluator；导出 UniLIP 已有预测时显式执行：

```text
q_evaluator = q_unilip * (span + 1e-6) / span
```

这保留预测的物理 Z，不读取 GT 或拟合测试统计。工具
[export_unilip_seen_predictions.py](scripts/export_unilip_seen_predictions.py)
只消费身份和 `pred_norm`，忽略源记录的 GT，输出标准 JSONL 与来源 sidecar。
通用 evaluator 不修改；工具已实现不代表已完成 UniLIP 的统一重评。工具调用方式见运行说明。

## 3. Action 路径、state 与 loss

aligned 数据集生成5D target，训练 hook 补成128D：前5D为pose，后123D为零。
保留原始 head 宽度，horizon 从 base64 适配为1；当前双图输入需要对应的序列位置 embedding 形状适配，
不改写 RDT blocks 的原生序列拼接结构。

| 环节 | aligned 已实现行为 |
| --- | --- |
| State | `(B,1,128)` 全零，state indicator 全零；结构槽位仍参与注意力和后续计算 |
| Action indicator | 前5D为1，其余为0；与 state indicator 分开 |
| 控制频率 | 固定1，不承载样本状态信息 |
| 训练噪声 | 完整128D Gaussian，原生 DDPM forward noising，1000个训练 timestep |
| 预测目标 | `prediction_type=sample`，clean-action prediction |
| 主 loss | 完整128D MSE，包含零 padding 维度的预测误差 |
| 推理 | 完整128D高斯初始化，原生 DPM-SolverMultistep，5步 |
| 中间通道 | 全128D参与每一步，不反复清零额外维度 |
| 对外输出 | 采样结束才应用 action mask，返回前5D |

零 state 不等于删除 token，也不等于 attention mask；它不提供真实状态，但保留预训练结构。
不为了靠近 UniLIP 而改成32D或flow matching。validation valid5 MSE不替代full128主loss，不增加额外定位监督。

legacy 默认仍为 `legacy_valid5`：前5D加噪/loss，每轮采样清零其他维度。
aligned 显式启用 `diffusion_channel_policy=native_full`；旧配置缺省值不切换为新行为。

## 4. 原生图像增强与标签一致性

训练的每张 FPV/radar 独立以0.5概率触发增强；触发后等概率选 `color_only`、`corrupt_only`、`both`。
操作和参数分布来自原生 `train/dataset.py`、`train/image_corrupt.py`：

- ColorJitter：brightness .3、contrast .4、saturation .5、hue .03。
- Corruption 必选 Gaussian/Laplace/Poisson 加性噪声之一，保留原生0～.05×255的scale/lam范围。
- 模糊可选零个或一个：Gaussian sigma0～3、Average kernel2～7、Median kernel3～11、Motion kernel3～36；保留原生组合和随机顺序。
- 顺序为 RGB → 训练增强 → square padding → SigLIP processor，输入384。

以上操作不改变相机pose/地图坐标系，标签保持不变；MotionBlur是观测滤波，不模拟相机轨迹。
crop、flip、rotation、translation、透视变换关闭：没有可靠的FPV/radar/相机/pose联合变换前不启用。
自动暗图增亮、state噪声和条件丢弃为独立机制，本profile不启用。validation/test无随机增强、无TTA。

增强随机源由run seed、epoch、global sample occurrence、样本ID和view派生，imgaug子节点受控；
其构造涉及的全局随机状态保存/恢复，不依赖worker分配或跳过batch次数。
同一occurrence跨worker/恢复应产生相同图像。像素强度饱和不属于action/pose clamp。
增强不增加数据记录，重复暴露仍计入固定预算。

## 5. 模块、LoRA 与优化

T5提供冻结文本特征；视觉/语言特征经各自adaptor，由RDT blocks的cross-attention交替读取。
任务适配由adaptor、policy LoRA和action模块承担，不解冻T5。

| 模块 | 训练方式 | 可训练参数量 | 初始LR |
| --- | --- | ---: | ---: |
| SigLIP vision tower | 冻结 | — | — |
| T5 encoder | 冻结，按地图指令预计算 | — | — |
| 28个RDT blocks | base冻结，LoRA训练 | 31,195,136 | 1e-4 |
| img_adaptor | 全量训练 | 6,557,696 | 1e-4 |
| lang_adaptor | 全量训练 | 12,587,008 | 1e-4 |
| state_adaptor | 全量训练 | 8,919,040 | 1e-4 |
| model.final_layer | 全量训练 | 4,460,672 | 1e-4 |
| model.t_embedder.mlp | 全量训练 | 4,722,688 | 1e-4 |
| model.freq_embedder.mlp | 全量训练 | 4,722,688 | 1e-4 |
| 三组位置embedding | 适配形状后冻结 | — | — |

LoRA固定 `r=32, alpha=64, dropout=.05, bias=none`，每个block targets为
`attn.qkv`、`attn.proj`、`cross_attn.q`、`cross_attn.kv`、`cross_attn.proj`、`ffn.fc1`、`ffn.fc2`，共196层。
保存配置后重载时重建LoRA，缺失/不匹配权重拒绝静默加载。

真实官方权重实例化确认：policy总参数 **1,253,414,016**，可训练 **73,164,928**，冻结 **1,180,249,088**，
policy口径训练比例约5.84%。7个optimizer组覆盖全部且仅覆盖可训练参数，无重复。
冻结SigLIP为428,225,600参数；policy+vision为1,681,639,616。T5 encoder另外披露，不能因预计算而省略其模型来源。

AdamW：betas(.9,.999)、epsilon1e-8、weight decay0、gradient clipping1。
可训练参数及moments为FP32，计算BF16；7组base LR均1e-4，59-update warmup后cosine到1e-5。
scheduler仅随成功optimizer update推进一次，不按microbatch/world size重复推进。
不启用DeepSpeed、8-bit Adam或gradient checkpointing。

该LoRA/LR属于预先固定的适配选择，不声称已有官方同配方的稳定运行证据。
不因为UniLIP冻结InternVL projector而机械冻结RDT的image/language adaptor，其结构功能不同。

## 6. 预算、checkpoint 与恢复

默认seed42，单GPU microbatch4 × accumulation32，全局有效定位batch **128**。
每epoch打乱50,000条、取49,920条，形成390个完整global batch；50epochs合计
**19,500 updates、2,496,000次定位样本暴露**。尾部80条不padding/复制，下一epoch重排。
先组成global update再拆rank/microbatch，避免分布式重复填充或二次切分。

| 节点 | 验证与保存的optimizer step |
| --- | ---: |
| 第1次 | 4,000 |
| 第2次 | 8,000 |
| 第3次 | 12,000 |
| 第4次 | 16,000 |
| 训练结束 | **19,500** |

**最终策略：每4000 updates验证/保存，末尾19500补做一次，不再使用3900间隔。**
目录为`checkpoint-<step>`。`late`链接到最近保存的step，完成后为`checkpoint-19500`；
`best`链接到五个候选中validation normalized valid5 MSE最低者。RDT不另设`last`别名。
主比较用final/late，best-val仅为标注清楚的附加结果；历史UniLIP每2000步保存，不能声称旧运行搜索密度相同。

每次validation使用全部5000条，固定seed和原生5-step推理，保存/恢复训练RNG。
默认validation batch4，与正式测试batch1分别记录；主loss与验证指标不同。
不使用test选择checkpoint、LR、增强或推理设置。

checkpoint保存模型、config、optimizer、scheduler、各rank RNG、完成update与sampler恢复位置。
先写临时目录再原子发布，随后更新相对best/late链接；不在根目录重复保存final权重。
smoke在每次保存时即写`smoke_only.json`，正式推理拒绝使用。

同实验的完整恢复校验YAML、实际batch/world size、资产、manifest、语言embedding哈希和optimizer设置。
新运行更改GPU数仍须保持batch128；不承诺任意改变正在恢复的rank/microbatch/accumulation后逐位一致。

## 7. 配置与文件级实现边界

| 文件/对象 | 职责 |
| --- | --- |
| configs/csgo_seen10.yaml | 保留legacy默认行为 |
| configs/csgo_seen10_aligned.yaml | 显式aligned profile、增强、LoRA、预算、保存节点和推理协议 |
| data/csgo_seen10.py：Seen10Dataset/collate_seen10 | 发布数据、身份与归一化、双视图；aligned外部5D，legacy128D；metadata不进入模型条件 |
| data/csgo_augmentation.py | 原生增强、局部RNG、操作白名单、标签不变约束 |
| data/csgo_update_sampler.py：AlignedUpdateSampler | global128/update、rank/microbatch分配、出现编号、恢复 |
| models/csgo_rdt.py：CSGORDTRunner | legacy_valid5/native_full、state indicator、原生采样、checkpoint加载 |
| models/csgo_adaptation.py：build_role_adaptation | LoRA、冻结、七组参数、dtype/数量审计 |
| train/csgo_hooks.py | base/冻结编码器/文本缓存、5D→128D、验证、链接、可视化 |
| train/train.py | 按profile分派；保留legacy/native分支 |
| train/csgo_aligned.py | update/optimizer/scheduler、协议约束、五次保存验证、完整恢复和审计 |
| train_seen10.py | CLI/YAML解析、原始参数记录、预算/节点校验、dry-run与smoke边界 |
| infer_seen10.py | aligned默认late、seed42/batch1/单进程、求解器provenance、标准预测和coverage |
| scripts/run_csgo_seen10.sh | 所选YAML决定默认seed/输出目录；eval调用共享原evaluator |
| scripts/csgo_paths.py | 跨服务器数据/评测路径解析；CLI、环境变量覆盖及旧默认路径的迁移回退，不改写实验 YAML |
| scripts/setup_csgo_seen10.sh | 创建或复用项目环境，检查依赖；不要求另一项目的环境存在 |
| train/csgo_visualize.py | 固定种子按地图取样；GT只供后处理 |
| scripts/export_unilip_seen_predictions.py | Z epsilon及预测格式转换，不修改evaluator |

数据/评测路径按显式CLI、环境变量、YAML的顺序解析；原机器默认路径不可用时回退到同级数据/评测目录和项目 Python，自定义路径不自动替换。训练与推理复用同一解析器。此变更不修改已保存协议中的 YAML，也不放宽跨服务器恢复校验。
显式CLI seed等覆盖YAML默认；YAML training映射为native参数后，透传参数最后解析。
aligned随后校验最终运行值，违反固定协议的覆盖报错，不能只凭YAML认定实际值。
审计记录raw_config、原始CLI/native argv、resolved_args、optimizer实际组和checkpoint状态。
contract包含manifest、完整YAML、语言缓存等哈希；不宣称已复制X-VLA的“30个split文件统一data-contract SHA256”实现。

正式初始化要求已审计缓存路径；RDT重新计算权重SHA256，编码器记录revision/大小/来源。
T5缓存由官方编码器按地图指令生成，不接受未审计外部embedding作为正式训练输入。
旧配置缺省语义不改变，新路径只由aligned profile启用。

aligned推理不允许部分结果按ID跳过后继续采样，以免噪声分配变化；须新目录从首样本重跑。
完整输出可核对provenance，不能混用checkpoint/batch/进程数。
推理记录样本顺序、manifest/config哈希、有效求解器参数与依赖版本。

## 8. 与原生 RDT、UniLIP 的比较边界

| 项目 | RDT原始方法 | 当前aligned | UniLIP exp32_loc |
| --- | --- | --- | --- |
| 外部任务 | 机器人action | Seen-10单步5D | Seen-10单步5D |
| 内部action/主loss | 128D/full128 clean-action MSE | 保留 | 32D/full32 flow-matching MSE |
| State | 真实proprio token | 零结构槽位仍参与计算 | 无独立state token |
| 文本编码器 | 冻结T5 | 保留 | LLM base冻结，LoRA训练 |
| 视觉/适配器 | 冻结SigLIP，image/language adaptor训练 | 保留 | vision/InternVL projector冻结，定位connector训练 |
| Policy | 原始入口全量训练 | base冻结+LoRA，适配模块全量训练 | action expert base冻结+LoRA，定位模块全量训练 |
| 可训练参数 | 随任务形状 | 实际73,164,928 | 实际55,688,992 |
| 图像 | 384；微调脚本启用原生增强 | 384；原生颜色/噪声/模糊 | 224；实际无随机增强 |
| Action qnorm/clamp | 主路径无，部分下游min-max | 无 | 无 |
| 外部Z | 任务相关 | 无epsilon | 分母有1e-6，导出需转换 |
| Batch/updates | 非CSGO预算 | 128/19500 | 128/19500，已有实际运行 |
| Scheduler | 微调脚本constant，历史运行未确认 | warmup59+cosine to1e-5 | warmup ratio.003+cosine-with-min-LR |
| 保存 | 微调脚本每1000，历史运行未确认 | 4000/8000/12000/16000/19500 | 配置每2000；主比较final19500 |
| 推理 | DPM-SolverMultistep5步 | 保留 | Euler flow10步 |

exp32_loc定位connector/norm/projector初始LR5e-4，LLM/action LoRA和IO/time为1e-4；
末端分别约5e-5和1e-5，不能说“所有组都到1e-5”。其整模型参数含未激活分支，
与RDT policy统计口径不同，不以总参数或训练比例机械匹配。

exp32最终实际19550updates，包含尾部update；代码/状态重建约2.5M loc+2.5M gen task samples，不能直接算19550×128。
exp32_loc的2496000暴露量同样由sampler/状态联合重建，历史逐ID轨迹和epoch shuffle行为未完全确认。
新增RDT不承担生成预算。

增强、384/224分辨率、冻结文本编码器、零state槽、预训练语料、objective和计算量仍有差异，
不能把性能差异全部归因于架构；不同objective的训练loss不可直接比较。
统一evaluator报告XY/Z/Pitch/Yaw逐地图与equal-map macro；每图完整2000条且指标为样本均值时pooled等于macro，
不声称存在独立pooled输出字段。

## 9. 验收要求与已完成记录

### 9.1 必须维持的检查

1. 地图、三split数量/ID、重复与交集；无GT/state/history输入。
2. Normalization、单位、物理量round-trip；无qnorm/clamp/测试统计。
3. 增强实际执行、pose不变、几何约束、val/test关闭；worker/恢复一致性。
4. 零state及独立indicator、5D→128D、完整噪声/loss/采样、最终5D输出。
5. 最终requires_grad、LoRA targets、真实参数量、optimizer成员/dtype/LR。
6. 有效batch、update/scheduler、五个节点、best/late、原子保存和完整恢复。
7. 单batch前后向、隔离smoke、标准预测到原evaluator闭环、legacy兼容。
8. 正式训练后再检查全部20000测试ID与指标；smoke不代替正式结果。

### 9.2 2026-09-24 aligned 验收结果

变更基于main@a912be0。以下为已完成的历史验收记录，本次文档整理没有重新执行这些检查。


| 检查 | 结果与证据 |
|---|---|
| 自动检查 | 22 项通过：数据 9、模型 5、训练 4、入口/导出 4；见 `.cache/csgo_seen10/aligned_acceptance_20260924/unit_tests.log` |
| 发布数据 | 50,000 train / 5,000 validation / 20,000 test；ID 唯一，split 无交集 |
| Normalization | pose 物理量 round-trip；无 epsilon 的 evaluator Z 约定；UniLIP epsilon 转换保留物理预测、忽略 GT 字段 |
| 增强 | 原生颜色/噪声/模糊，无几何变换；pose 不变；0 与 2 workers、恢复后的同一 occurrence 图像一致；不推进全局 Python/NumPy/Torch/imgaug RNG |
| 输入边界 | validation/test 不执行随机增强；无标签样本不返回 GT；state 和 state indicator 为零，结构槽位保留 |
| 原生扩散 | 完整 128D 噪声/主 loss；采样中保留全维状态，最后才 mask；aligned 对外输出 5D |
| 官方权重加载 | 618/618 个权重键加载成功，无缺失或跳过；仅预期的序列位置 embedding 形状适配 |
| 官方模型实际参数审计 | policy 总参数 1,253,414,016；可训练 **73,164,928**；196 个 LoRA 层；7 组完整且不重叠；训练参数 FP32，冻结 base BF16 |
| 官方 base 身份 | 重新计算 RDT 大文件 SHA256，与资产清单一致：`fe217f4491ea882b0b52df1cb23ae4e8a11c1328ed29f2ed712e02aad2c02102` |
| 单进程 CPU 小模型 | 实际完成 5 个 optimizer updates、逐节点保存/验证、5-step 推理 10 条样本；不是正式实验 |
| 单进程中断恢复 | 从 step3 恢复至 step5；100/100 模型 tensor 逐位一致，optimizer 和 scheduler 状态完全一致 |
| 双进程 CPU 小模型 | 2 ranks × microbatch1 × accumulation2，5 updates，20 次暴露；每个保存点的 optimizer step/scheduler step 一致 |
| 双进程中断恢复 | 从 step3 恢复至 step5，同样模型/optimizer/scheduler 完全一致 |
| Checkpoint 文件与链接 | smoke 的 5 个完整 checkpoint，每份仅一个模型文件；保存时即带 smoke 标记；`best`/`late` 为相对符号链接 |
| 推理重读 | 单进程 batch1；完整输出的 provenance 可重复核对；部分输出不允许跳过 ID 后继续采样 |
| 原 evaluator 闭环 | 使用原 `run_eval.py smoke localization --limit 1` 成功读取标准预测；结果标明 `formal=false`、`official_output_written=false` |
| Legacy 兼容 | 原 tiny 配置独立完成 5-step CPU smoke，原 valid5/旧训练行为保留 |
| Shell/CLI | wrapper dry-run 正确解析 seed42、新输出路径、batch128 和五个保存点；Python 编译、shell 语法与 `git diff --check` 通过 |

实际参数清单：
[aligned_actual_parameter_audit.json](.cache/csgo_seen10/aligned_actual_parameter_audit.json)。

其余验收证据均位于：
`.cache/csgo_seen10/aligned_acceptance_20260924/`，主要包括：

- `official_asset_audit.json`：官方资产标识及 RDT 实际哈希。
- `resume_comparison.json`：单进程连续/恢复一致性。
- `ddp_resume_comparison.json`：双进程连续/恢复一致性。
- `additional_checks.json`：双进程 update/exposure/checkpoint 检查。
- `train.log`、`resume.log`、`ddp.log`、`ddp_resume.log`、`legacy.log`。
- `current_infer.log`、`current_infer_recheck.log`、`evaluator.log`。

### 9.3 官方资产与首次接入的历史 smoke

官方资产清单见 [.cache/csgo_seen10/asset_manifest.json](.cache/csgo_seen10/asset_manifest.json)。

```text
RDT revision: eb09036cc64ca4945051acbd1bd581d30a1d7711
RDT pytorch_model.bin SHA256:
fe217f4491ea882b0b52df1cb23ae4e8a11c1328ed29f2ed712e02aad2c02102
```

首次seed0 tiny fixture位于`.cache/csgo_seen10/tiny_fixture_seed0_v2/`。
历史环境为Python3.11.14、Torch2.11.0.dev20260124+cu128，执行强制CPU。
该次5-step smoke每次validation100条、推理100条（每图10），evaluator smoke limit1，
结果为`smoke_only=true, formal=false, official_output_written=false`。
产物位于`checkpoints/csgo_benchmark_v2_seen10/RDT/smoke/seed_0/`和
`outputs/csgo_benchmark_v2_seen10/RDT/smoke/seed_0/`。
它与本次aligned十条推理smoke、legacy seed0正式20000条结果分别记录。

### 9.4 验收边界与未确认项

- aligned未正式运行，尚无checkpoint-4000至checkpoint-19500；正式节点通过代码检查/dry-run验证，tiny的1～5节点不是正式模型。
- 所有前后向、恢复和推理闭环使用随机初始化小模型fixture并强制CPU；官方1B做了真实加载、LoRA/参数组及哈希审计。
- 官方1B GPU前后向、峰值显存、吞吐、正式推理延迟和收敛尚未实测，不从smoke指标推断性能。
- UniLIP完整原始资产哈希、逐样本轨迹及部分验证/推理细节仍未确认；不把配置/注释/相邻实验当作运行证据。
- legacy历史provenance未保存实际推理batch/进程数；已有正式结果与当前默认重跑不保证逐位一致。
- 此前文档整理没有运行训练、推理、评测或测试。此前aligned验收也未安装依赖、下载权重或修改外部UniLIP/数据/evaluator。

### 9.5 跨服务器环境准备变更与验收

准备脚本取消对原机器 ControlAR 环境的强制依赖，支持新建 Python 3.11 环境、显式克隆、复用已有环境和首次安装中断后重试。新环境安装完整兼容依赖；已有环境补依赖时约束已安装版本，避免静默替换训练环境。`--check` 只读导入验证，`--dry-run` 只显示准备方案。

已知目标服务器为 A100、驱动 580.125.09、CUDA 13.0 报告值；新环境选择 PyTorch 2.8.0 / torchvision 0.23.0 cu128。本机现有 nightly 环境保留。benchmark/evaluator 路径通过共享解析器迁移，实验 YAML、有效 batch、update、增强及五个 checkpoint 保存节点没有变更。

- 原有 aligned 22 项检查及新增路径 5 项检查通过；新增环境脚本 7 项模拟检查通过，共 34 项。覆盖原路径保留、新服务器回退、CLI/环境变量优先级、无 ControlAR 环境、新建 conda、CUDA 13.0 选择 cu128、安装中断重试。
- 本机实际 `setup_csgo_seen10.sh --check` 通过：Python 3.11.14、Torch 2.11.0.dev20260124+cu128、torchvision 0.25.0.dev20260124+cu128、NumPy 1.26.4；训练/推理模块导入成功，CUDA 可用。
- aligned dry-run、wrapper 路径打印、shell 语法、文档命令语法和 `git diff --check` 通过。
- 没有执行依赖安装、模型下载、正式训练/推理/评测。新服务器的实际安装与 GPU 运行尚待用户同步后验证；不把模拟安装分支测试当作另一台机器的实测。

### 9.6 2026-09-25 tokenizer 依赖修复

另一台服务器训练启动日志报告缺少 protobuf。此前准备脚本只检查模型类导入，未覆盖 T5 tokenizer 转换；本机已有 protobuf，因此此前导入检查不能证明新环境具备这一依赖。

- 依赖清单新增 `protobuf==6.33.4`；已有环境检测使用真实导入名 `google.protobuf`，兼容 `google` 命名空间本身也不存在的情况，已有版本不强制重装。
- `--check` 检查 protobuf message 和 SentencePiece schema；缓存存在时实际离线加载 T5 tokenizer 并编码，不加载 T5 权重。缓存不存在时明确跳过，要求资产下载后重查。
- 10 项准备脚本测试通过（原 7 项加缺失命名空间、缺失 protobuf、已安装 protobuf 三种回归场景）；本机真实 tokenizer 加载/编码和完整 `--check` 通过。
- 未安装依赖、下载模型或启动训练；截图只展示异常尾部，另一台服务器修复后的运行仍需用户验证。

### 9.7 2026-09-25 无图形界面服务器的 OpenCV 修复

补装 protobuf 后，另一台服务器报告 `imgaug`/`cv2` 因缺少 `libGL.so.1` 导入失败。本机包元数据确认 `imgaug==0.4.0` 声明依赖 `opencv-python`；即使清单指定 headless，pip 仍可能同时安装 GUI 版。四种 OpenCV 发行包共用 `cv2` 文件，不能依赖安装顺序决定实际加载哪一版。

- 准备脚本在所有依赖安装后检查四种发行包，检测到混装、GUI 包、版本不符或导入失败时，卸载已有 OpenCV 发行包并以 `--no-deps --force-reinstall` 安装 `opencv-python-headless==4.11.0.86`；不改动 NumPy、PyTorch 或训练配置。
- 健康检查验证实际 OpenCV 构建的 `GUI: NONE`；`--check` 只读报错并提示重跑准备脚本。pip 操作失败会停止，不输出环境 ready。
- 10 项准备脚本检查与 4 项 OpenCV 模拟检查通过，包括混装修复、重复执行不重装、损坏的 headless 导入、卸载/安装失败传播。
- 临时目录实际安装 headless wheel，使用子进程阻断 libGL：原 GUI OpenCV 导入失败，headless OpenCV 与 RDT 增强正常运行。64 次固定 occurrence 覆盖颜色、corruption、组合及不增强，和原环境输出 SHA256 一致：`cd1c755d3d5f3ef854204e810d31f7a178153d2baae681effafe6b0739858782`。这验证本次样本上的一致性，不代表所有平台逐位等价。
- 实测摘要见 [.cache/csgo_seen10/opencv_headless_acceptance_20260925.json](.cache/csgo_seen10/opencv_headless_acceptance_20260925.json)。本机现有 `.venv` 未修改，其中也检测到 GUI/headless 混装，新 `--check` 正确拒绝该状态；只有用户手动执行 setup 才会修复。本轮仅在 `/tmp` 安装 OpenCV 测试包，没有下载模型或启动正式任务。
