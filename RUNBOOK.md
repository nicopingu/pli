# Installation runbook

PLI runs on Apple Silicon only (macOS 14 or later, Python 3.11 to 3.14). It is validated on a Mac Studio
M1 Ultra, 64 GB, macOS 27. Choose a checkpoint from the [recipes](docs/recipes/README.md) and check disk space
and memory before downloading it.

## 1. Install

```bash
xcode-select --install                     # once: needed to compile the Neural Engine client
git clone https://github.com/nicopingu/pli.git ~/dev/pli
python3 -m venv ~/venvs/pli
~/venvs/pli/bin/pip install -e ~/dev/pli   # extras: [vision] image input, [test] test suite, [ssd] experts on SSD
~/venvs/pli/bin/pli --version
~/venvs/pli/bin/pli models
```

## 2. Get a model

The main target is Qwen3.8-27B (dense, MLX 4-bit) with its DFlash2 draft model:

```bash
pli info mlx-community/Qwen3.8-27B-4bit     # reads config.json only
pli pull mlx-community/Qwen3.8-27B-4bit z-lab/Qwen3.8-27B-DFlash2
```

`info` reads configuration only; `pull` downloads weights; `serve` completes a missing download.
A local folder with `config.json` and the weights works in place of a repo id.

## 3. Serve

```bash
pli serve mlx-community/Qwen3.8-27B-4bit --profile m1-ultra --drafter auto \
  --name qwen3.8-27b --port 8080 --parallel 4 \
  --snapshot-dir ~/pli-state/27b --spill-gib 40
```

- `--profile m1-ultra` sets the `PLI_*` settings measured on an M1 Ultra (shared ones and the `qwen3_5` family's);
  the first log line says which files were read. Without a profile, PLI's additions stay off, except three that
  are on by default in the code: `PLI_CONTROL_AS_TEXT`, `PLI_SPILL_PROMPT` and `PLI_FORK_MIN`.
- On a chip that is not an Ultra, PLI uses one Neural Engine block even if the profile asks for two, and says so.
- The first use of the Neural Engine path compiles `libpli_ane.dylib` (a few seconds). If the compiler is missing
  the server logs the reason and keeps the GPU path.
- Machine-specific values go in the environment, not in a profile: `PLI_ALLOWED_HOSTS` (extra host names accepted
  by the Host/Origin guard, e.g. the machine's LAN name) and `PLI_EVENTS` (path of the event journal).
- Step trace, to find slow steps on real traffic: start the server with `PLI_TRACE_DIR=<dir>` (nothing is
  written while it is off). `python -m tensorfold.pli.trace on --dir <dir> --duration 24h` switches it on within a
  second, without a restart, and it switches itself off at the end; `off` and `status` stop it or show it. Each
  server writes `<dir>/trace-<served name>.jsonl` (one JSON line per request, prefill chunk, decode round and
  draft; same text produced, no GPU synchronisation added). Report: `python -m tensorfold.pli.trace_analysis
  <dir>/trace-<name>.jsonl --md report.md`.
- `--snapshot-dir` keeps conversation prefixes on disk across restarts (`none`: memory only); `--spill-gib`
  writes evicted prefixes to disk.

## 4. Check the endpoint

Leave the server running and use another terminal:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128}'
```

OpenAI clients use `http://127.0.0.1:8080/v1`, Anthropic clients `http://127.0.0.1:8080`. Reasoning can appear
separately from the answer; `--no-thinking` turns it off by default. See [API fields](docs/api.md).

## 5. Run as a service

PLI does not install a service. On macOS a LaunchAgent or a supervisor (for example llama-swap) starts
`pli serve ...` with the environment above. Stop it with one SIGTERM and let it finish: a second signal interrupts
the writing of conversation snapshots.

## Memory and context

Omit `--context` to fit the default window to the model and the memory budget, then read the capacity the server
reports. `--context 0` removes the metadata cap while memory admission still applies; a positive context that cannot
fit is refused at startup.

`PLI_MEMORY_LIMIT_GB` sets the process budget in GiB in place of the default 70% of RAM, within physical RAM
and the GPU's recommended working set. The budget reserves 3 GiB outside the allocator. On an M1 Ultra 64 GB the
production setting for the 27B is `--context 229376 --parallel 4`.

Requested replies need cache space too: after a memory refusal, reduce the context, the reply length, the retained
prefixes or the checkpoint size. The model file size is not the whole process footprint.

## Updating

```bash
git -C ~/dev/pli pull --ff-only
~/venvs/pli/bin/pip install -e ~/dev/pli
```

Restart the server afterwards. `pli update` is disabled in this fork: update with `git pull` and `pip install` as
above.

## Tests

```bash
~/venvs/pli/bin/pip install -e "$HOME/dev/pli[test]"
cd ~/dev/pli && PLI_NO_UPDATE_CHECK=1 ~/venvs/pli/bin/python -m pytest -q tests
```

Tests needing hardware this Mac lacks are skipped with the reason. They use the GPU lightly; run them while a
production server is idle if timings matter elsewhere.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `PLI runs on Apple Silicon (macOS, MLX) only` | PLI has no other backend |
| `unknown profile` | `--profile` names a folder of `src/tensorfold/pli/profiles/` |
| `libpli_ane.dylib missing and not compiled` | `xcode-select --install`, then `bash src/tensorfold/pli/ane/build.sh` |
| Download failure | Repository id, access and free disk space |
| Rejected checkpoint | Quantization, model family and draft-head requirements (`pli info`) |
| Client cannot connect | Server process, `/health`, base URL, model id, `PLI_ALLOWED_HOSTS` for a LAN name |
| Start stops after `loading …`, GPU idle | `kill -USR1 <pid>` prints every thread's Python stack |
| A family's tests are skipped for the threadgroup limit | Its kernels need a newer GPU (GLM-5.3-Flash: M3 Ultra) |

Unsupported architectures or formats need a family implementation; see
[adding a family](docs/recipes/adding-a-family.md).
