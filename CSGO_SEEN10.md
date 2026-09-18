# RDT CSGO Benchmark v2 Seen-10

The adapter uses the released read-only bundle at
`/home/jiahao/task/UniLIP/data/csgo_benchmark_v2`, the project environment
`.venv`, and the shared evaluator at
`/home/jiahao/task/csgo_benchmark_v2_eval_general`.

Prepare the project environment and official model cache with:

```bash
scripts/setup_csgo_seen10.sh
.venv/bin/python scripts/prepare_csgo_assets.py
```

`prepare_csgo_assets.py` is resumable and keeps downloaded model artifacts in
the project `.cache/` tree. The official RDT checkpoint is present and has been
structurally verified; official SigLIP and T5 remain to be downloaded if a
formal run is started.

The completed smoke environment used Python 3.11.14 and
`torch 2.11.0.dev20260124+cu128` (CUDA build; execution was forced to CPU).

For a CPU smoke run, create a new tiny fixture. The generator builds real
`T5EncoderModel`, `SiglipVisionModel`, and `CSGORDTRunner` instances with a
deterministic seed and refuses to overwrite an existing target:

```bash
.venv/bin/python scripts/make_csgo_tiny_fixture.py \
  --output .cache/csgo_seen10/tiny_fixture_seed0_v2
```

The completed seed-0 smoke was run with:

```bash
scripts/run_csgo_seen10.sh smoke --seed 0 \
  --config .cache/csgo_seen10/tiny_fixture_seed0_v2/config.yaml --cpu
```

`--cpu` is forwarded to both train and infer. Training used one sample per
map for five updates; each of the five validation passes wrote 100 rows (10
per map), and inference wrote 100 rows (10 per map). The shared evaluator
smoke used `--limit 1` and exited 0 with `smoke_only=true`, `formal=false`,
`official_output_written=false`, and `sample_count=1`.

```text
exit=0
smoke_only=true  formal=false  official_output_written=false
sample_count=1
```

Smoke artifacts are under
`checkpoints/csgo_benchmark_v2_seen10/RDT/smoke/seed_0/` and
`outputs/csgo_benchmark_v2_seen10/RDT/smoke/seed_0/`. Checkpoints 1–5,
`best`, and `late` are present. Validation metrics have counts of 100 and the
localization provenance records `model_inputs_contain_ground_truth: false`.

Mode-specific options after `--` are forwarded to the active entry point; for
example, `scripts/run_csgo_seen10.sh infer --seed 0 -- --batch-size 4`.
Use a new seed for a new run. Existing predictions are resumed by identity;
the inference provenance file rejects a different checkpoint or seed.

For a formal released-model run, use a new seed and the native commands below
after the official SigLIP/T5 assets are available:

```bash
scripts/run_csgo_seen10.sh train --seed 0
scripts/run_csgo_seen10.sh infer --seed 0
scripts/run_csgo_seen10.sh eval --seed 0
```

Formal checkpoints and predictions use
`checkpoints/csgo_benchmark_v2_seen10/RDT/seed_<seed>/` and
`outputs/csgo_benchmark_v2_seen10/RDT/seed_<seed>/localization/`; formal
evaluator output is under `evaluation/localization/`.

The implementation files are `data/csgo_seen10.py`, `models/csgo_rdt.py`,
`train/csgo_hooks.py`, `train/csgo_visualize.py`, `train_seen10.py`,
`infer_seen10.py`, `scripts/run_csgo_seen10.sh`,
`scripts/make_csgo_tiny_fixture.py`, `configs/csgo_seen10.yaml`, and
`CSGO_SEEN10.md`; the native integration is in `train/train.py` and `main.py`.

`RUN_FULL=0` was respected, so no formal 1B training or formal evaluator run
was started. The tiny smoke checkpoint is smoke-only and is rejected by
formal inference. Smoke outputs are engineering validation, not benchmark
results; the released data root and shared evaluator were not modified.
