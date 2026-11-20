# finfm

finfm combines financial news, a typed economic sheaf, investor personas, and a mean-field trading game to construct cost-aware portfolios. the sheaf models how a news shock moves through asset relationships, while the persona branch models investor beliefs and trading pressure. the final framework combines both forecasts and applies portfolio constraints and transaction costs.

## architecture

![finfm architecture](assets/architecture.png)

## repository structure

the experiment entry points are grouped by dataset, model, and stage. each stage calls the shared implementation, so the scientific code stays consistent across runs.

```text
finfm/
├── databucket1/
│   ├── olmo/       ── sheaf/, personas/, final_framework/
│   ├── gemma/      ── sheaf/, personas/, final_framework/
│   └── llama/      ── sheaf/, personas/, final_framework/
├── tessera/
│   ├── olmo/       ── sheaf/, personas/, final_framework/
│   ├── gemma/      ── sheaf/, personas/, final_framework/
│   └── llama/      ── sheaf/, personas/, final_framework/
├── databucket3/
│   ├── olmo/       ── sheaf/, personas/, final_framework/
│   ├── gemma/      ── sheaf/, personas/, final_framework/
│   └── llama/      ── sheaf/, personas/, final_framework/
├── finfm/          # shared scientific implementation
├── reader2/        # news reader, validation, and shard scheduling
├── pilot/          # reader prompts and supporting inference code
├── config/         # experiment settings and small lookup tables
├── dataset/tessera/ # anonymized Tessera input dataset and standalone loader
├── scripts/        # build, merge, freeze, and input conversion tools
├── tests/          # synthetic tests of numerical methods and pipeline behavior
├── assets/architecture.png
├── run_pipeline.py
└── pyproject.toml
```

each stage directory contains `run.py`. `sheaf/reader.py` runs the news reader before cleaning and fitting the sheaf. `final_framework` contains the entry point for the sheaf, persona-game, and merged forecast comparisons.

## models and datasets

| directory | model identifier in configuration | checkpoint |
| --- | --- | --- |
| `olmo` | `llm2` | `allenai/Olmo-3-7B-Instruct` |
| `gemma` | `llm1` | `google/gemma-4-12B-it-qat-w4a16-ct` |
| `llama` | `llm3` | `meta-llama/Llama-3.1-8B-Instruct` |

gemma uses the quantized checkpoint with its full-precision fallback configured in `config/engine.yaml`.

| experiment directory | internal dataset identifier | description |
| --- | --- | --- |
| `databucket1` | `db1` | financial news with geopolitical and social supplements |
| `tessera` | `tessera` | point-in-time financial news package |
| `databucket3` | `db3` | financial and general news with social posts |

tessera's actual dataset files are included in [`dataset/tessera/`](dataset/tessera/), alongside a standalone loader and file checksums. the other datasets and model weights are supplied separately, and personal dataset-hosting links are omitted. the experiment loaders and preprocessing code are in `finfm/data/` and `finfm/stage/`. experiment windows are in `config/windows.yaml` and `config/clean.yaml`.

the nine dataset/model combinations are supported entry points. the four additional final-build configurations use their dataset's existing time windows and paths; their presence does not assert that those experiments were completed. tessera's gemma and llama readers retain the configured hash-based subset selection in `reader2/settings.py`.

## installation

use python 3.11 or later, from this repository's root:

```bash
python -m pip install -e '.[dev]'
```

for model inference on a compatible CUDA GPU:

```bash
python -m pip install -e '.[gpu]'
```

stage model weights locally using `FINFM_WEIGHTS_ROOT`, or supply any required access credential through the process environment. credentials belong outside source files. GPU memory requirements and parallelism are set in `config/engine.yaml`; full experiment runs are computationally intensive.

## local inputs

`--root` points to a local working tree with `dataset/`, `store/`, and generated `outputs/` folders. paths for each final build are declared in `config/clean.yaml`. provide the reader tape, staged units, asset links, market prices, and factor table for that run. the persona branch also needs investor skeletons, biography inputs, and reader bundles; the merged framework reads the matching persona bundles.

the news reader accepts `--input-root` containing a prepared reader pack: `RUN_PLAN.json`, per-dataset `data/<dataset>/units.parquet`, alias tables, and matching local model inputs. its schema and integrity checks are implemented in `reader2/plan.py` and `reader2/session.py`. persona inputs are resolved by `finfm/stage/run_stage.py` and `finfm/stage/persona_stage.py`.

generate tapes, manifests, fitted parameters, and freeze records from this source and configuration before running later stages. configuration identifiers and hashes must match throughout the pipeline. existing freeze records from a different configuration cannot be reused.

## running the stages

show the common options:

```bash
python tessera/olmo/sheaf/run.py --help
```

run the reader with local prepared inputs:

```bash
python tessera/olmo/sheaf/reader.py --input-root /path/to/reader-pack
```

run persona inference with locally staged market, reader, and biography inputs:

```bash
python tessera/olmo/personas/run.py --input-root /path/to/persona-inputs
```

check the sheaf's declared input paths, then run cleaning and sheaf fitting:

```bash
python tessera/olmo/sheaf/run.py --root /path/to/local-tree --check-inputs
python tessera/olmo/sheaf/run.py --root /path/to/local-tree
```

the final framework requires matching generated build artifacts and freeze records. create those with the freeze tools under `scripts/` after preparing the inputs and development fit, then run:

```bash
python tessera/olmo/final_framework/run.py --root /path/to/local-tree
```

replace `tessera` and `olmo` in these commands with the desired directory names. use `--out` to choose a generated-output folder and `--config` to supply an alternate configuration directory. the lower-level build scripts expose additional fitting and sharding options.

## implementation

| component | source |
| --- | --- |
| news prompts, inference, and validation | `reader2/`, `pilot/`, `finfm/reader/` |
| cleaning and point-in-time controls | `finfm/clean/`, `finfm/common/timing.py` |
| economic graph and sheaf fitting | `finfm/road1/`, `finfm/build/` |
| investor panels and persona answers | `finfm/personas/`, `finfm/stage/persona_build.py` |
| mean-field game | `finfm/mfg/` |
| forecast combination and final framework | `finfm/framework/` |
| costs, constraints, and portfolio execution | `finfm/portfolio/` |

## validation

```bash
python -m pytest
```

the tests generate small synthetic fixtures in temporary folders. full pipeline replication and GPU inference require the remaining local inputs and suitable hardware. this release includes Tessera, the architecture image, and configuration constants; saved notebook executions, experiment results, model weights, and checkpoint bundles are excluded.
