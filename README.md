<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pli-logo-dark.svg">
    <img src="docs/assets/pli-logo.svg" alt="PLI logo: a folded-paper figure drawn in engraved lines" width="200">
  </picture>
</p>

# PLI

PLI ("Parallel Linked Inference") is an LLM inference server for Apple Silicon, tuned for the M1 Ultra.
It is a fork of [TensorFold](https://github.com/ashhart/TensorFold) 0.6.6 (Apache-2.0) and keeps its
OpenAI- and Anthropic-compatible HTTP API, its exact speculative decoding and its model families.

Speed of PLI 4.0.0 on the validated machine (Mac Studio M1 Ultra, 64 GB, Qwen3.8-27B 4-bit, one request at a time;
how it is measured: [Measurements](#measurements)):

| Measure | Min | At | Max | At |
| --- | ---: | --- | ---: | --- |
| Prompt reading (tok/s) | 191 | 62-token prompt | 323 | 8k-token prompt |
| Writing (tok/s) | 34.1 | 96k-token context | 111.2 | 62-token context |
| Time to the first token (ms) | 320 | 62-token prompt | 450 030 | 96k-token prompt |

PLI adds work that spreads one model across the GPU, the CPU and the Neural Engine, kernels measured for the dense
Qwen3.8-27B, and a draft that picks, per conversation, between DFlash2 and the model's own MTP head.

**4.1.2 is a pre-release.** It is validated on one machine and one model family. Each change and its measurement is in
[CHANGELOG.md](CHANGELOG.md); every `PLI_*` setting is in [docs/switches.md](docs/switches.md).

- **A dedicated Mac.** PLI is made for a Mac that runs nothing else: with the 27B it keeps about 36 GB of memory
  wired, uses the whole GPU and the Neural Engine, and writes up to 40 GB of conversation snapshots to disk. Other
  heavy programs running at the same time slow it down and can make it refuse requests for lack of memory.
- **Apple Silicon only.** The CUDA backend of TensorFold is removed; `pli serve` refuses to start on a system other
  than macOS, and MLX itself needs Apple Silicon.
- **Main target: Qwen3.8-27B (dense)** with the DFlash2 draft and the 27B's MTP head. The other families inherited
  from TensorFold run on the standard MLX path and are not validated under PLI (see [Model families](#model-families)).
- **Settings by machine profile.** PLI's additions are off by default and switched on by `PLI_*` variables, grouped in
  profiles (`--profile m1-ultra`). Three are on by default in the code: `PLI_CONTROL_AS_TEXT`, `PLI_SPILL_PROMPT` and
  `PLI_FORK_MIN`.

## Quick start

```bash
git clone https://github.com/nicopingu/pli.git
cd pli
./install.sh install
```

`install.sh` finds Python 3.11 to 3.14, installs PLI in its own virtual environment under `~/pli`, then runs
`pli setup`, which does the rest: it checks the Mac (Apple Silicon, macOS 14 or later, memory, free disk, Xcode Command
Line Tools for the Neural Engine), downloads the models, writes the MTP head in 4 bits, downloads and converts A1's map
(on an Ultra chip), and starts PLI as a per-user
service that starts again at login and after a crash.

Before anything is downloaded, `pli setup` lists each item with its Hugging Face repo and page, its revision (`main`,
not pinned, with the commit it points to), its license as its model card gives it ("see model card" otherwise), its
size as the Hub lists it (read from the Hub's API, no weights fetched; "about" when the Hub does not answer), where it
goes on disk, and what it brings:

| Item | What it brings |
| --- | --- |
| [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | the model (required) |
| [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | DFlash2 draft: faster replies, the same text |
| [EigenLabs/Qwen3.8-27B-MTP-bf16](https://huggingface.co/EigenLabs/Qwen3.8-27B-MTP-bf16) | the model's MTP head, the hybrid draft with DFlash2: faster, the same text |
| MTP head in 4 bits | written on the Mac from the head above, nothing more to download |
| [A1 map](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/d-lsa-weights-v1) of Latent Sparse Attention (6 MB, Apache-2.0; with the `m1-ultra` profile only) | long prompts read faster (A1, below; the text can change). Downloaded from the GitHub release, checked against its SHA-256, converted to NumPy on the Mac (no torch), set in the service with `PLI_S3B_THRESHOLD=16384` and `PLI_S3B_ROWS=4096`; left out when the model's `config.json` is not Qwen3.8-27B's |

A file already in the Hugging Face cache is marked "already downloaded" and not counted. Then come the total to
download and the free disk, and the question `Download them? [y/N]` (no by default). `--yes` answers yes; without a
terminal and without `--yes`, `pli setup` stops before downloading and says what to add.

Each item can be left out: `--no-hybrid` (no MTP head: DFlash2 alone), `--no-a1` (no A1 map), `--no-drafter` (no draft at all: slower, the
same text), `--model DIR` (a model already on the Mac, not downloaded; also `--drafter DIR`, and `--mtp-head FILE` for
a head already written), `--no-models` (nothing downloaded: the service starts only if the model and the draft are
already on the Mac, and uses the hybrid draft only with `--mtp-head`). The other options: `--prefix DIR`, `--port N`
(default 8080), `--model REPO` (a Qwen3.8-27B checkpoint or fine-tune: the drafts are the 27B's), `--context N`,
`--no-start`. `./install.sh install` passes every option to `pli setup`; a second `pli setup` keeps the prefix, port
and model of the first unless told otherwise. On a Mac without a screen, turn on automatic login so that PLI starts
after a reboot.

Then (`pli` is `~/pli/venv/bin/pli`; `./install.sh status` and the others run the same commands):

| Command | What it does |
| --- | --- |
| `pli status` | is the service loaded, is the server answering, which model |
| `pli start`, `stop`, `restart` | start or stop the server (stopping lets it finish writing its snapshots) |
| `pli logs` | the last lines of the server log |
| `pli bench` | the public bench suite against the local server (see [Benchmark](#benchmark)) |
| `pli uninstall` | removes the service, `~/pli`, the settings, the logs and the snapshots on disk; asks before deleting each downloaded model (`--keep-models` keeps them; `--yes` deletes them without asking) |

`pli` alone gives a short help; `pli --help` lists every command.

The server answers OpenAI clients at `http://127.0.0.1:8080/v1` and Anthropic clients at `http://127.0.0.1:8080`.
PLI ships no model weights. Each checkpoint keeps its own license (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Validated hardware

| Item | Validated |
| --- | --- |
| Machine | Mac Studio, M1 Ultra (20-core CPU, 64-core GPU, two Neural Engine blocks), 64 GB unified memory |
| OS | macOS 27 (27.0.1) |
| Models | Qwen3.8-27B dense in MLX 4-bit ([mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit), license: Apache-2.0), and [sabicap/sabi-noema](https://huggingface.co/sabicap/sabi-noema) (license: Apache-2.0), a Qwen3.8-27B fine-tune with the same architecture, in MLX 4-bit |
| Drafts | DFlash2 ([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), license: Apache-2.0) and the MTP head published with Qwen3.8-27B |
| Profile | `m1-ultra` |

Other Apple Silicon Macs can run PLI with its additions off, or with a profile of their own; they are not measured.
`pli setup` gives the `m1-ultra` profile to any Ultra chip with 64 GB or more: an M2 Ultra or M3 Ultra gets it
without having been measured.

## What PLI adds to TensorFold 0.6.6

| Area | What it does | Where |
| --- | --- | --- |
| Prompt reading (27B) | Weights unpacked before the product, attention in f16, tuned GEMM tiles, last tokens read with the chunk before | `src/tensorfold/pli/unpack.py`, `src/tensorfold/pli/merge_tail.py`, `src/tensorfold/kernels/qwen/dense/v1/` |
| Neural Engine (27B) | Part of each FFN layer runs on the Neural Engine while the GPU reads the prompt; its weights are requantized on the CPU | `src/tensorfold/pli/ane/` |
| Writing (27B) | Own decode attention (`attm`), weights laid out for the decode kernel, reduced attention on long contexts anchored to positions, Kapitsa's 8-row verify GEMM | `src/tensorfold/pli/attention_attm.py`, `src/tensorfold/pli/relayout.py`, `src/tensorfold/kernels/qwen/dense/v1/qmm8.py` |
| Scheduling | Cost arbiter, small requests first, several conversations sharing a round, heartbeat between requests | `src/tensorfold/pli/arbiter.py`, `src/tensorfold/server/scheduler.py` |
| Drafts | Hybrid DFlash2 + MTP head chosen per conversation; Recall drafts copied from finished requests | `src/tensorfold/families/qwen3_5_moe/hybrid.py`, `src/tensorfold/pli/recall.py` |
| Agent turns | The reply is kept for the next turn instead of being read again; checkpoints keep a length instead of a copy of the KV cache; control tokens written in a request stay text | `src/tensorfold/server/text.py`, `src/tensorfold/server/prompt_fill.py`, `src/tensorfold/pli/kv_sharing.py` |
| Operations | Host/Origin guard, per-conversation event journal (no message content), step trace for debugging | `src/tensorfold/server/host_guard.py`, `src/tensorfold/pli/events.py`, `src/tensorfold/pli/trace.py` |

## Stable features

In production on the validated machine. All are set by `--profile m1-ultra`, except the machine settings marked
(env). "Same bits" means the result is bit-identical to the path it replaces; "text can change" means another
arithmetic, judged on whole conversations before it went into production.

| Feature | Settings | Text | New in |
| --- | --- | --- | --- |
| Unpacked 4-bit weights in prompt reading, one-pass unpacking kernel | `PLI_PREFILL_UNPACK`, `_MLP`, `_NN`, `_KERNEL` | can change (f16) | 3.6.0 or earlier |
| Prompt attention: f16 products, tuned GEMM tiles | `PLI_ATTENTION_F16`, `PLI_ATTENTION_GEMM` | can change (f16) | 3.6.0 or earlier |
| Neural Engine share of the FFN on two blocks | `PLI_ANE_FFN`, `PLI_ANE_BLOCKS`, `PLI_ANE_FFN_UNITS` | can change | 3.6.0 or earlier |
| Neural Engine kept awake, units by chunk size | `PLI_ANE_KEEPAWAKE`, `PLI_ANE_STEPS` | can change | 3.7.0 |
| Neural Engine weights requantized on the CPU | `PLI_ANE_REQUANT_CPU`, `PLI_ANE_REQUANT_THREADS` | same bits | 3.7.0 |
| Decode attention `attm`, 128-key slices on short calls | `PLI_ATTENTION_ATTM`, `PLI_ATTM_SHORT` (fallbacks: `PLI_ATTENTION_ATTEC`, `PLI_ATTENTION_PER_ROW`) | can change | 3.7.0: `PLI_ATTM_SHORT` |
| Reduced attention beyond 20 000 tokens, anchored to positions | `PLI_ATTENTION_REDUCED`, `PLI_REDUCED_ROUNDS`, `PLI_REDUCED_ANCHOR` | can change | 3.7.0: `_ROUNDS`, `_ANCHOR` |
| Decode kernels: weight layout, bf16 fragments, one tile per threadgroup | `PLI_RELAYOUT`, `PLI_XF_BF16`, `PLI_TILES_PER_GROUP` | same bits | 3.7.0: `PLI_TILES_PER_GROUP` |
| Cost arbiter, small requests first, prompt chunks between decode rounds | `PLI_ARBITER`, `PLI_ADMIT_SMALL`, `PLI_ROUNDS_BETWEEN_PREFILLS`, `PLI_CHUNK_LAYERS`, `PLI_INTERLEAVE_DENSE` | same text | 3.6.0 or earlier |
| Last prompt tokens read with the chunk before | `PLI_MERGE_TAIL` | can change | 3.7.0 |
| Reply kept for the next agent turn | `PLI_KEEP_REPLY` | can change | 3.7.0 |
| Heartbeat between requests | `PLI_HEARTBEAT_MS` | same text | 3.7.0 |
| Recall drafts | `PLI_RECALL` | same text | 3.6.0 or earlier |
| Hybrid draft: DFlash2 and the MTP head, chosen per conversation; one-turn questions with DFlash2 alone | `PLI_HYBRID` (env), `PLI_MTP_HEAD` (env), `PLI_MTP_DRAFTS`, `PLI_MTP_WINDOW`, `PLI_HYBRID_MESSAGES` | same text | 3.7.0 |
| Kapitsa's 8-row verify GEMM for rounds of up to 128 rows | `PLI_V8` | can change | 3.7.0 |
| Checkpoints without a copy of the KV cache | `PLI_KV_SHARING` | same text | 3.7.0 |
| Control tokens written in a request kept as text | `PLI_CONTROL_AS_TEXT` (on by default) | same text without control tokens | 3.7.0 |
| Host/Origin guard, arrival and end times in the log | `PLI_HOST_GUARD`, `PLI_LOG_TIMES` | same text | 3.6.0 or earlier |
| Draft trees for one-turn questions (KFlash, verified through `attm`): the DFlash2 chain stays while it keeps 4.5 tokens per round; agent sessions keep the chain | `PLI_KFLASH`, `PLI_TREES_ATTM`, `PLI_HYBRID_TREES`, `PLI_TREE_DELTANET`, `PLI_TREES_SHORT_ONLY`, `PLI_TREES_TOKEN_THRESHOLD` | can change (near ties) | 4.0.0 |
| Copy of the context or DFlash2 chain, chosen per stream from what each would have given | `PLI_HEAD_BEFORE_COPY` | same text | 4.0.0 |
| Extended draft lists: a file of 2 515 token ids, one per line, added to the reduced vocabularies of both drafts (for each draft, the 2 048 tokens it missed most, merged; shipped with PLI). Learned on the author's traffic, mostly French; its gain for other use is not measured | `PLI_EXTENDED_DRAFT_LIST` | same text | 4.0.0 |
| Draft kernels: attention, stacked projections, q and k norms with RoPE in one kernel, top-16 spread, block projections laid out | `PLI_DRAFT_ATTN`, `PLI_DRAFT_STACK`, `PLI_DRAFT_QKR`, `PLI_DRAFT_TOPK`, `PLI_RELAYOUT_DRAFT` | same text (drafts are verified) | 4.0.0 |
| `attm` fragments loaded by simdgroup; 8-row tables prepared in parallel | `PLI_ATTM_LOAD`, `PLI_V8_PREP2` | same bits | 4.0.0 |
| Reduced attention, anchored, for prompt chunks of up to 2 047 rows beyond 65 536 tokens of context | `PLI_S3B`, `PLI_S3B_THRESHOLD` | can change | 4.0.0 |
| Snapshot on disk read again at the first slice of a resumed prompt | `PLI_REREAD_DISK` | see CHANGELOG | 4.0.0 |
| Prompt checkpoint spilled to disk even when a reply extends it; checkpoint where a prompt forks in its last message (both on by default) | `PLI_SPILL_PROMPT`, `PLI_FORK_MIN` | same text | 4.0.0 |
| Checkpoint at the start of the prompt's last message, past 1 024 tokens read there: a request that keeps the history but not the last message starts from there. On the agent session of the public suite: the same 8 replies, prompt served from the cache 73.5% instead of 58.9%, sum of the times to the first token 238.0 s instead of 357.3 s (-33%), same writing speed | `PLI_LAST_MESSAGE_CHECKPOINT` | same text | 4.0.1 |
| MLX command buffers of at most 500 MB and 100 operations (env, set by `pli setup`) | `MLX_MAX_MB_PER_BUFFER`, `MLX_MAX_OPS_PER_BUFFER` | same text | 4.0.0 |
| Checkpoints kept in 4 bits (rotated) when evicted from memory or written to disk, restored to 16 bits on resume: disk of the checkpoints divided by 4, time to the first token of resumed turns -54% under memory pressure; on conversations judged turn by turn, no right answer became wrong | `PLI_CHECKPOINT_BITS` | can change (resumed turns) | 4.1.0 |
| A1: prompt chunks past 16 384 positions read, row by row, 10% of the tiles of 32 keys scored by the LSA index, plus their last 1 024 positions; decode unchanged. time to the first token at 146k tokens 712.5 s instead of 844.6 s (-15.6%), offline NLL equal to the full prefill at 48k and 97k | `PLI_A1_LSA_MAP` (env), `PLI_A1_PREFILL_ONLY`, `PLI_S3B_THRESHOLD`, `PLI_S3B_ROWS` (env) | can change | 4.1.0 |
| `checkpoint` field of `/health` and `/v1/models`: served weights (folder or Hugging Face repo and revision, model type, quantization), draft, MTP head and PLI version | none | same text | 4.1.0 |

## Experimental features

In the code, off by default, not in any profile. They are there to be measured; none is supported.

| Feature | Settings | Status |
| --- | --- | --- |
| FALCON: capture of training examples for the MTP head (hidden state, chosen token, top logits, prompt rows in 8 bits) | `PLI_FALCON`, `PLI_FALCON_MAX_GB`, `PLI_FALCON_*` | Experimental. Same text; rounds slower by 0.7%, prompt reading slower by 1.5%. No learned head is served. |
| FALCON hot swap of a published MTP head | `PLI_FALCON_HEAD` | Experimental, not in production |
| MTP head alone in place of DFlash2 | `PLI_MTP_HEAD` without `PLI_HYBRID` | Experimental |
| Tree gain floor: a KFlash tree only when it brings that many more expected tokens than the chain | `PLI_TREE_GAIN_MIN` | Experimental |
| Round profile, copy log, lattice log (to replay drafts offline) | `PLI_ROUND_PROFILE*`, `PLI_COPY_LOG`, `PLI_LATTICE_LOG` | Diagnostic, off by default |
| DFlash2 chain drawn on the GPU | `PLI_GPU_CHAIN` | Experimental, no gain measured |
| Checkpoint log | `PLI_CHECKPOINT_LOG` | Diagnostic, off by default |
| Part of the prompt rows computed on the CPU | `PLI_PREFILL_CPU*` | Experimental, text can change |
| Arbiter cost table observed in service, draft floor | `PLI_ARBITER_TABLE`, `PLI_ARBITER_MIN_ROWS` | Experimental |
| Row-invariant attention for windows | `PLI_ATTENTION=gemm` or `m1`, `PLI_ATTN_*` | Experimental |
| Neural Engine call at the heartbeat's pace | `PLI_HEARTBEAT_ANE` | Experimental |
| A1 with a cap on the tiles read per row (faster at long context: first token at 146k -22% to -27% against the full prefill instead of -15.6%, but the NLL rises at 97k) | `PLI_A1_MAX_TILES` | Experimental, not in production |
| A1 for decode and trees too (`PLI_A1_PREFILL_ONLY=0`) | `PLI_A1_LSA_MAP` | Experimental: decode 5 to 21% slower on long conversations |

The full list, with each variable's code default, is in [docs/switches.md](docs/switches.md). Every setting starts with
`PLI_`; the names used before 3.7.0 and 4.0.0 (French `PLI_*` names, `TF_*`, `TENSORFOLD_*`) are still accepted.

## Model families

PLI's kernels, Neural Engine path and drafts are written and measured for the dense Qwen3.8-27B. The other families
come from TensorFold 0.6.6 and keep its MLX engines; PLI has not validated them.

| Family | `model_type` | Status under PLI | Model on Hugging Face | Recipe |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B (dense) and its fine-tunes | `qwen3_5` | **Validated** (M1 Ultra, 64 GB, `m1-ultra` profile) | [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (installed by `install.sh`); fine-tune validated: [sabicap/sabi-noema](https://huggingface.co/sabicap/sabi-noema) | [qwen3.8-27b.md](docs/recipes/qwen3.8-27b.md) |
| Qwen3.6-35B-A3B (MoE) | `qwen3_5_moe` | Not validated under PLI | [TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP](https://huggingface.co/TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP) | [qwen3.6-moe.md](docs/recipes/qwen3.6-moe.md) |
| GLM-5.3-Flash | `glm5_next` | Not validated under PLI (written for a 256 GB Mac) | [TensorFold/GLM-5.3-Flash-MLX-4bit-MTP](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP) | [glm-5.3-flash.md](docs/recipes/glm-5.3-flash.md) |
| DeepSeek-V4-Flash | `deepseek_v4` | Not validated under PLI (written for a 256 GB Mac) | [mlx-community/DeepSeek-V4-Flash-4bit](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit) | [deepseek-v4-flash.md](docs/recipes/deepseek-v4-flash.md) |
| Nemotron 3.5 Lightning | `nemotron_h` | Not validated under PLI | [TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit](https://huggingface.co/TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit) | [nemotron-3.5.md](docs/recipes/nemotron-3.5.md) |
| Gemma 4 26B-A4B | `gemma4` | Not validated under PLI | [mlx-community/gemma-4-26b-a4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) | [gemma-4.md](docs/recipes/gemma-4.md) |
| Qwen3.8 Flash Next | `qwen4_exp` | Not validated under PLI (sized for a 128 GB Mac) | [TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP) | [qwen3.8-flash-next.md](docs/recipes/qwen3.8-flash-next.md) |
| Ternary Bonsai 2 27B | `bonsai` | Not validated under PLI | [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) | [ternary-bonsai-2.md](docs/recipes/ternary-bonsai-2.md) |

The recipes are TensorFold's; their CUDA sections describe the upstream project and do not apply to PLI.

## Neural Engine

> **Warning.** The Neural Engine path relies on private Apple APIs; a macOS update can break it.

The `m1-ultra` profile switches it on (`PLI_ANE_FFN=1`). Its client library is compiled on first use, which needs the
Xcode Command Line Tools; without them PLI keeps the GPU path. Two Neural Engine blocks are used only on an Ultra chip.
On the first error, PLI computes the layer on the GPU. To run without it, set `PLI_ANE_FFN=0` in the environment.

## Manual install and run

What `install.sh` and `pli setup` do, by hand (without the checks, the MTP head and the service):

```bash
git clone https://github.com/nicopingu/pli.git
python3 -m venv ~/pli/venv
~/pli/venv/bin/pip install ./pli            # add [vision] for image input, [test] for the test suite
~/pli/venv/bin/pli pull mlx-community/Qwen3.8-27B-4bit z-lab/Qwen3.8-27B-DFlash2
~/pli/venv/bin/pli serve mlx-community/Qwen3.8-27B-4bit --profile m1-ultra --drafter auto --port 8080
```

The MTP head in 4 bits: `~/pli/venv/bin/python -m tensorfold.install.mtp_head --model MODEL_DIR SOURCE OUTPUT`
(or `tools/quantize_mtp_head.py` from a clone). `pli serve --help` lists every option; `pli info MODEL` shows what a
checkpoint needs; `pli models` lists the families
and their tested checkpoints; `pli service --help` manages the per-user service. Memory sizing, health checks and
updates are in [RUNBOOK.md](RUNBOOK.md).

### Profile `m1-ultra`

A profile is a folder of `src/tensorfold/pli/profiles/`: `common.env` for every model and `<model_type>.env` for one
family (`qwen3_5` for the Qwen3.8 dense models). `--profile NAME` or `PLI_PROFILE=NAME` loads it before any kernel is
imported. A variable already set in the environment always wins over the profile.

`m1-ultra` sets every stable production setting of the validated machine. What names a file, a directory or a host
stays out of it and goes in the environment:

| Variable | Use |
| --- | --- |
| `PLI_HYBRID=1` and `PLI_MTP_HEAD=<file>` | hybrid draft: DFlash2 and the 27B's MTP head (4-bit safetensors). `PLI_HYBRID` needs `PLI_MTP_HEAD` and `--drafter`. The head: [EigenLabs/Qwen3.8-27B-MTP-bf16](https://huggingface.co/EigenLabs/Qwen3.8-27B-MTP-bf16) (license: Apache-2.0), the 15 `mtp.*` tensors of Qwen3.8-27B copied unchanged; `tools/quantize_mtp_head.py` writes it in 4 bits (810 MiB to 228 MiB) |
| `MLX_MAX_MB_PER_BUFFER=500`, `MLX_MAX_OPS_PER_BUFFER=100` | MLX command-buffer limits; MLX reads them when it starts, before any profile, so they go in the environment |
| `PLI_ALLOWED_HOSTS` | extra host names accepted by the Host/Origin guard (for example the machine's LAN name) |
| `PLI_EVENTS` | path of the event journal (one JSON line per scheduler action, no message content) |
| `PLI_FALCON`, `PLI_FALCON_MAX_GB` | experimental FALCON capture: directory and disk cap in GB |
| `PLI_A1_LSA_MAP=<file.npz>`, with `PLI_S3B_THRESHOLD=16384` and `PLI_S3B_ROWS=4096` | A1: prompt chunks past 16 384 positions read 10% of the tiles of 32 keys, scored by the distilled map of [Latent Sparse Attention](https://github.com/Prefrontal-AI/latent-sparse-attention) for Qwen3.8-27B (release [`d-lsa-weights-v1`](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/d-lsa-weights-v1), file `qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt`, SHA-256 `24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b`, license: Apache-2.0), converted once by `tools/convert_lsa_map.py` (NumPy only); `pli setup` does both and sets the three variables. The two S3b values go with the map: without it, keep the profile's 65 536 |

Production setup of the validated machine:

```bash
export PLI_HYBRID=1
export PLI_MTP_HEAD=~/pli/models/mtp-head-4bit.safetensors
export PLI_EVENTS=~/pli/logs/events.jsonl
export MLX_MAX_MB_PER_BUFFER=500 MLX_MAX_OPS_PER_BUFFER=100
export PLI_A1_LSA_MAP=~/pli/models/qwen38_27b_d2048.npz PLI_S3B_THRESHOLD=16384 PLI_S3B_ROWS=4096
pli serve mlx-community/Qwen3.8-27B-4bit --profile m1-ultra --drafter auto --context 229376 --parallel 4 --spill-gib 40
```

## Benchmark

`pli bench` measures prompt reading and writing speed of the local PLI server (the port of `pli setup`), with thinking
off. From a clone, `tools/pli_bench.py` runs the same code against any server that speaks the OpenAI chat API with
streaming, so PLI and other engines can be compared on the same Mac with the same requests; it needs only standard
Python.

```bash
pli bench --salt pub --only single --sizes code,1k,8k,32k,64k
pli bench --salt pub --only single --sizes 96k
pli bench --salt pub --only agent
python3 tools/pli_bench.py --url http://127.0.0.1:8080/v1 --salt pub --only agent    # any server
```

The requests are fixed: they come from the suite pli-bench-v1 (`src/tensorfold/bench/data/suite-v1.json`, installed
with PLI) and its corpus (PLI's own source code), checked by SHA-256 before the run, so every run on every machine sends
the same texts.

| Part | Requests |
| --- | --- |
| `single` | one message each: a short code request (400 tokens written), then prompts of 1k, 8k, 32k, 64k and 96k tokens (Qwen3.8 tokenizer) followed by one question (200 tokens written) |
| `agent` | one session of an agent reading 25 files of a repository, growing from about 6k to 50k tokens; at 8 points a question is asked and its answer measured (200 tokens), then the session goes on without it, so the prompt prefix is shared as in real agent work |

Only the first line of each conversation changes with `--salt` (random by default), so that no prompt cache from an
earlier run serves it; with `--salt pub` the texts are those of the measurements below. For each request: reading =
prompt tokens / time to the first written token (in the agent part the prefix may come from the cache); writing =
(written tokens - 1) / (end - first written token); the agent result is all written tokens over all writing time.
Decoding is greedy. Memory (wired, available, swap, server resident memory with `--pid`) is sampled every second.
Results go to a JSON file (`--out`). `--free` keeps the older mode with free sizes (`--contexts`).

`tools/pli_bench.py` sends no thinking field, so measure with thinking off (`pli bench` passes
`--extra '{"chat_template_kwargs": {"enable_thinking": false}}'`): start PLI with `pli serve ... --no-thinking`, or pass
`--extra` for other servers (`'{"chat_template_kwargs": {"enable_thinking": false}}'` for PLI and mlx-lm,
`'{"reasoning_effort": "none"}'` for Kapitsa); for llama.cpp, start its server with `--reasoning off`.

## Measurements

The figures in [CHANGELOG.md](CHANGELOG.md) come from the validated machine below with PLI's own benches, as each
line says. The figures in this section come from the public suite (see [Benchmark](#benchmark)), run with
`--salt pub`, so the texts are the same for every engine and can be run again on another Mac.

| Item | Value |
| --- | --- |
| Machine | Mac Studio, M1 Ultra (20-core CPU: 16 performance and 4 efficiency cores; 64-core GPU; two Neural Engine blocks), 64 GB |
| OS | macOS 27.0.1 |
| Suite | pli-bench-v1 (`src/tensorfold/bench/data/suite-v1.json`), `--salt pub`; one engine at a time, nothing else running; greedy decoding, thinking off |
| PLI 4.0.0 | set up as `install.sh` (now `pli setup`) does: `--profile m1-ultra`, hybrid draft (DFlash2 and the MTP head of Qwen3.8-27B in 4 bits), `--context 229376 --parallel 4`; Python 3.12, MLX 0.32.3, mlx-lm 0.32.0 |
| [Kapitsa](https://github.com/nace-ai/kapitsa) 0.2.24 | its defaults with the DFlash2 draft, `--context 131072 --slots 4`, `reasoning_effort: none` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) b11926 | Qwen3.8-27B in GGUF Q4_K_M with its MTP head (GGUF Q8_0) as draft (`--spec-type draft-mtp --spec-draft-n-max 2`), `-fa on -ngl all`, `--reasoning off` |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) 0.32.2 | `mlx_lm.server` (MLX 0.32.4), no draft, thinking off |
| Models | mlx-community/Qwen3.8-27B-4bit for PLI, Kapitsa and mlx-lm; drafts z-lab/Qwen3.8-27B-DFlash2 |
| Units | tokens per second (tok/s), times in milliseconds (ms), prompt sizes in tokens (Qwen3.8 tokenizer) |
| Dates | 2 and 3 November 2026 |

### One request at a time

Prompt reading (higher is better):

| Prompt | Tokens read | PLI 4.0.0 reading (tok/s) | Kapitsa 0.2.24 reading (tok/s) | llama.cpp reading (tok/s) | mlx-lm reading (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| short code request | 62 | 191 | 156 | 121 | 91 |
| 1k | 1 001 | 294 | 210 | 193 | 159 |
| 8k | 8 001 | 323 | 209 | 192 | 165 |
| 32k | 32 002 | 285 | 189 | 168 | 153 |
| 64k | 64 002 | 245 | 167 | 144 | 136 |
| 96k | 96 002 | 213 | 151 | 126 | not measured (prompt reading over 13 min) |

Writing (higher is better; 400 tokens written for the code request, 200 for the others):

| Prompt | Tokens read | PLI 4.0.0 writing (tok/s) | Kapitsa 0.2.24 writing (tok/s) | llama.cpp writing (tok/s) | mlx-lm writing (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| short code request | 62 | 111.2 | 116.1 | 31.1 | 24.1 |
| 1k | 1 001 | 53.2 | 65.3 | 25.1 | 24.0 |
| 8k | 8 001 | 45.8 | 54.2 | 23.0 | 22.6 |
| 32k | 32 002 | 49.2 | 43.1 | 19.9 | 18.8 |
| 64k | 64 002 | 40.8 | 32.9 | 17.1 | 17.2 |
| 96k | 96 002 | 34.1 | 33.6 | 15.5 | not measured |

Time from the request to the first token written (lower is better; measured to 10 ms):

| Prompt | Tokens read | PLI 4.0.0 time (ms) | Kapitsa 0.2.24 time (ms) | llama.cpp time (ms) | mlx-lm time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| short code request | 62 | 320 | 400 | 510 | 680 |
| 1k | 1 001 | 3 410 | 4 770 | 5 180 | 6 300 |
| 8k | 8 001 | 24 810 | 38 210 | 41 650 | 48 620 |
| 32k | 32 002 | 112 200 | 169 280 | 190 040 | 208 580 |
| 64k | 64 002 | 261 810 | 382 840 | 443 660 | 469 130 |
| 96k | 96 002 | 450 030 | 634 250 | 761 140 | not measured |

PLI reads prompts 1.4 to 1.5 times as fast as Kapitsa from 1k to 96k (1.2 times on the short code request). Kapitsa
writes faster on short prompts (code request, 1k, 8k); PLI writes faster at 32k and 64k, and level at 96k. Both draft with DFlash2; llama.cpp drafts with the MTP head, mlx-lm
without a draft. Writing speed depends on how many drafted tokens the model accepts, so it depends on the text: two
prompts of the same size can differ by several percent.

### Agent session

One session that grows from 6 468 to 50 706 tokens, with 8 measured answers of 200 tokens (see
[Benchmark](#benchmark)). Each run was stopped after 13 minutes.

| Measure | PLI 4.0.0 | Kapitsa 0.2.24 | llama.cpp | mlx-lm |
| --- | ---: | ---: | ---: | ---: |
| Answers measured in 13 minutes | 8 of 8 | 6 of 8 | 8 of 8 | 6 of 8 |
| Writing, first 6 answers (tok/s) | 47.0 | 46.7 | 24.3 | 20.9 |
| Writing, all 8 answers (tok/s) | 47.1 | | 22.7 | |
| Mean time to the first token, first 6 answers (ms) | 31 100 | 97 300 | 31 300 | 121 400 |
| Prompt tokens served from the cache, first 6 answers | 57 % | 0 % | 73 % | 0 % |

The session drops each measured answer before it goes on, so a cache must reuse a prefix that ends inside the
conversation. llama.cpp reuses any common prefix of the previous request; PLI reuses the checkpoint kept where an
earlier prompt forked, which here is one request older, and reads the rest again, faster. Kapitsa and mlx-lm read the
whole prompt again at each request in this run.

## Debugging

The step trace writes one JSON line per engine step of each request. It is off by default and costs nothing while
off; use it to find where time goes, not in normal service.

| Variable | Use |
| --- | --- |
| `PLI_TRACE_DIR=<dir>` | trace available but off; switch it on without a restart: `python -m tensorfold.pli.trace on --dir <dir> [--duration 24h]`, off with `off`; read it with `python -m tensorfold.pli.trace_analysis` |
| `PLI_TRACE=<file>` | trace on from the start (benches, tests) |
| `PLI_TRACE_LAYERS`, `PLI_ANE_STATS`, `PLI_ANCHOR_LOG`, `PLI_HYBRID_LOG` | per-chunk layer times, Neural Engine load, reduced-attention reasons, hybrid draft switches |

## Tests

```bash
~/pli/venv/bin/pip install "./pli[test]"
PLI_NO_UPDATE_CHECK=1 ~/pli/venv/bin/python -m pytest -q tests
```

Tests that need hardware PLI's GPU lacks are skipped with the reason (Metal 4 tensor units on M5-generation GPUs,
or 1 024 threads per threadgroup for GLM-5.3-Flash's kernels). Measurement probes that are not unit tests live in
`tools/probes/`.

## Inspirations and third-party code

Code taken from other projects, with its license, is listed file by file in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). What PLI adds to TensorFold's list:

| Source | What PLI takes | License |
| --- | --- | --- |
| [Splash](https://github.com/incoai/splash), as ported to M1/M2 GPUs by [paperniuk/splash](https://github.com/paperniuk/splash) (1.3.2-m1, commit 5c484b2) | Neural Engine client (`src/tensorfold/pli/ane/native/`): `ane/ProgramInstrumentation.hpp`, `AwakeClock.hpp` and `Checked.hpp` unmodified; `ane/Program.mm` and `ane/Program.hpp` modified by PLI to pin a program to one Neural Engine block (each change marked `PLI`); the MIL program of the shared FFN and its split rewritten in Python from Splash's `runtime/ops/AneFfn.cpp` | Apache-2.0 (`src/tensorfold/pli/ane/native/LICENSE-splash`) |
| [Kapitsa](https://github.com/nace-ai/kapitsa) v0.2.23 | 8-row verify GEMM (`src/tensorfold/kernels/qwen/dense/v1/qmm8.py`): the X8 input table and the dense 4-bit GEMM of `src/kernels/verify8.metal`, with Kapitsa's M1 tiling, turned into MLX kernels | Apache-2.0 ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)) |

`pli_ane.mm`, `metal/MetalBackend.hpp`, the Host/Origin guard (`src/tensorfold/server/host_guard.py`) and everything
else under `src/tensorfold/pli/` are written for PLI.

Ideas used without their code:

- **A1, reduced prompt reading scored by the LSA index** (`PLI_A1_LSA_MAP`): the index keys and queries of
  [Latent Sparse Attention](https://github.com/Prefrontal-AI/latent-sparse-attention) (Prefrontal-AI, D-LSA, commit
  906be19, Apache-2.0) score PLI's tiles; the distilled map is downloaded from its release, not shipped with PLI. No code
  is copied: the kernels are written for PLI on top of `attm`.

- **Recall drafts** (`PLI_RECALL`): the "Recall" of
  [TensorFold 1.0.12](https://github.com/ashhart/TensorFold/releases/tag/v1.0.12) (written in Zig; no code copied).
- **Checkpoint at the fork point** (`PLI_FORK_MIN`): the rule of TensorFold pull request
  [#1291](https://github.com/ashhart/TensorFold/pull/1291) (Zig, closed without merging), rewritten in Python.
- **Control tokens kept as text** (`PLI_CONTROL_AS_TEXT`): the behaviour of
  [TensorFold 1.1.2](https://github.com/ashhart/TensorFold/releases/tag/v1.1.2) for issue
  [#1117](https://github.com/ashhart/TensorFold/issues/1117), rewritten in Python.
- **Heartbeat between requests** (`PLI_HEARTBEAT_MS`, `PLI_HEARTBEAT_ANE`): [Kapitsa](https://github.com/nace-ai/kapitsa)
  [#206](https://github.com/nace-ai/kapitsa/pull/206) (residency heartbeat); keeping the Neural Engine awake between
  requests follows the same idea.
- **CPU requantization for the Neural Engine** (`PLI_ANE_REQUANT_CPU`): the CPU as a third unit writing straight into
  the Neural Engine's surfaces, from [ane-infer](https://github.com/thebasedcapital/ane-infer) and
  [ANEDVD](https://github.com/midasdf/ANEDVD).
- **FALCON** (`PLI_FALCON`): training the draft head by unrolling its chain, as EAGLE-3 does (["EAGLE-3: Scaling up
  Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840)).

Drafts used, weights not distributed: [DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) by z-lab (license:
Apache-2.0; its MLX model code is vendored from [z-lab/dflash](https://github.com/z-lab/dflash), MIT) and the MTP head
that Qwen publishes with Qwen3.8-27B ([EigenLabs/Qwen3.8-27B-MTP-bf16](https://huggingface.co/EigenLabs/Qwen3.8-27B-MTP-bf16),
license: Apache-2.0).

## Naming

PLI stands for "Parallel Linked Inference". The Python module is still called `tensorfold` and its log prefix is
still `[tensorfold]`; the command is `pli` (`tensorfold` remains an alias). Every setting is a `PLI_*` variable, and
PLI's own log lines start with `[pli]`.

## License

Apache-2.0, as TensorFold. PLI is a fork of TensorFold 0.6.6 (commit cb2ebf05); copyright 2026 Nicolas F. (nicopingu)
for PLI's changes, and the TensorFold contributors for TensorFold. See [LICENSE](LICENSE), [NOTICE](NOTICE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
