# PLI

PLI ("Parallel Linked Inference") is an LLM inference server for Apple Silicon, tuned for the Ultra chips.
It is a fork of [TensorFold](https://github.com/ashhart/TensorFold) 0.6.6 (Apache-2.0) and keeps its
OpenAI- and Anthropic-compatible HTTP API, its exact speculative decoding and its model families.

PLI adds work that spreads one model across the GPU, the CPU and the Neural Engine, and kernels measured
for the dense Qwen3.8-27B. It is validated on one machine: a Mac Studio M1 Ultra, 64 GB, macOS 27.

- **Apple Silicon only.** The CUDA backend of TensorFold is removed; `pli serve` refuses to start elsewhere.
- **Main target: Qwen3.8-27B (dense)** with a DFlash2 draft model. The other families inherited from
  TensorFold run on the standard MLX path; see [docs/recipes](docs/recipes/README.md).
- **Settings by machine profile.** PLI's additions are off by default and switched on by `PLI_*`
  variables, grouped in profiles (`--profile m1-ultra`).

## What PLI adds to TensorFold 0.6.6

| Area | What it does | Where |
| --- | --- | --- |
| Prompt reading (27B) | Weights unpacked before the product, attention in f16, tuned GEMM tiles | `tensorfold/pli/deplie.py`, `kernels/qwen/dense/v1/` |
| Neural Engine (27B) | Part of each FFN layer runs on the Neural Engine while the GPU reads the prompt (one or two blocks) | `tensorfold/pli/ane/` |
| Writing (27B) | Own decode attention (`attm`), weights laid out for the decode kernel, optional reduced attention on long contexts | `tensorfold/pli/attention_attm.py`, `pli/range_poids.py` |
| Scheduling | Cost arbiter, small requests first, several conversations sharing a round | `tensorfold/pli/arbitre.py`, `server/scheduler.py` |
| Drafts | Recall: drafts copied from finished requests | `tensorfold/pli/rappel.py` |
| Operations | Host/Origin guard, per-conversation event journal (no message content) | `server/host_guard.py`, `pli/evenements.py` |

Each change and its measurement is listed in [CHANGELOG.md](CHANGELOG.md).

## Requirements

- A Mac with Apple Silicon. The `m1-ultra` profile assumes an Ultra chip (two Neural Engines) and 64 GB.
- macOS 14 or later (validated on macOS 27), Python 3.11 or later.
- For the Neural Engine path: the Xcode Command Line Tools (`xcode-select --install`). The client library is
  compiled on first use; without it PLI keeps the GPU path.

## Install

```bash
git clone git@github.com:nicopingu/pli.git
python3 -m venv ~/venvs/pli
~/venvs/pli/bin/pip install -e ./pli            # add [vision] for image input, [test] for the test suite
~/venvs/pli/bin/pli --version
```

Step-by-step checks, memory sizing and service setup are in [RUNBOOK.md](RUNBOOK.md).

## Run

```bash
pli pull mlx-community/Qwen3.8-27B-4bit z-lab/Qwen3.8-27B-DFlash2
pli serve mlx-community/Qwen3.8-27B-4bit --profile m1-ultra --drafter auto --port 8080
```

The server answers OpenAI clients at `http://127.0.0.1:8080/v1` and Anthropic clients at `http://127.0.0.1:8080`.
`pli serve --help` lists every option; `pli info MODEL` shows what a checkpoint needs; `pli models` lists the
families and their tested checkpoints.

### Profiles

A profile is a folder of `tensorfold/pli/profils/`: `commun.env` for every model and `<model_type>.env` for one
family (`qwen3_5` for the Qwen3.8 dense models). `--profile NAME` or `PLI_PROFIL=NAME` loads it before any kernel is
imported. A variable already set in the environment always wins over the profile.

Machine-specific values stay out of profiles and go in the environment:

| Variable | Use |
| --- | --- |
| `PLI_HOTES_ADMIS` | extra host names accepted by the Host/Origin guard (e.g. the machine's LAN name) |
| `PLI_EVENEMENTS` | path of the event journal (one JSON line per scheduler action) |

## Tests

```bash
~/venvs/pli/bin/pip install -e "./pli[test]"
TENSORFOLD_NO_UPDATE_CHECK=1 ~/venvs/pli/bin/python -m pytest -q tests
```

Tests that need hardware PLI's GPU lacks are skipped with the reason (Metal 4 tensor units on M5-generation GPUs,
or 1,024 threads per threadgroup for GLM-5.3-Flash's kernels). Measurement probes that are not unit tests live in
`tools/sondes/`.

## Naming

The Python module is still called `tensorfold`, and so are its log prefix (`[tensorfold]`) and its `TENSORFOLD_*`
variables; the command is `pli` (`tensorfold` remains an alias).

## License

Apache-2.0, as TensorFold. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The Neural Engine client reuses code from Splash 1.3.2-m1 (Apache-2.0, `tensorfold/pli/ane/natif/LICENSE-splash`).
