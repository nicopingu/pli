# PLI benchmark and measurements

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

The requests are fixed: they come from the suite pli-bench-v1 (`src/pli/bench/data/suite-v1.json`, installed
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

The figures in [CHANGELOG.md](../CHANGELOG.md) come from the validated machine below with PLI's own benches, as each
line says. The figures in this section come from the public suite (see [Benchmark](#benchmark)), run with
`--salt pub`, so the texts are the same for every engine and can be run again on another Mac.

| Item | Value |
| --- | --- |
| Machine | Mac Studio, M1 Ultra (20-core CPU: 16 performance and 4 efficiency cores; 64-core GPU; two Neural Engine blocks), 64 GB |
| OS | macOS 27.0.1 |
| Suite | pli-bench-v1 (`src/pli/bench/data/suite-v1.json`), `--salt pub`; one engine at a time, nothing else running; greedy decoding, thinking off |
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
