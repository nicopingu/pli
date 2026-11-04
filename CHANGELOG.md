# What's new in PLI

Each release's section below lists its changes and the measurements behind them. `pli update` is disabled in this
fork: update with `git pull` and `pip install` (see [RUNBOOK.md](RUNBOOK.md)).


## Unreleased

- **`pli setup` installs PLI; `install.sh` only makes the virtual environment.** `./install.sh install` finds Python
  3.11 to 3.14, creates `~/pli/venv` (or `--prefix`), installs PLI in it and runs `pli setup` with the same options.
  `pli setup` does what the rest of `install.sh` did, with the same checks, messages, defaults and service settings
  (profile `m1-ultra` on an Ultra chip with 64 GB, `PLI_ANE_FFN`, `MLX_MAX_MB_PER_BUFFER=500`,
  `MLX_MAX_OPS_PER_BUFFER=100`, `--context 229376 --parallel 4`, `--spill-gib 40` with 70 GB free, the hybrid draft),
  through `pli pull` and `pli service`. `pli status`, `start`, `stop`, `restart`, `logs` and `uninstall` replace the
  commands of `install.sh`, which still run them (`./install.sh status` is `pli status`).
- **Nothing is downloaded without consent.** Before downloading, `pli setup` lists each item: Hugging Face repo, page
  and revision (`main`, not pinned, with its commit), license from the model card, size read from the Hub's API
  ("about" when the Hub does not answer), where it goes, the free disk, and what it brings. Files already in the
  Hugging Face cache are marked "already downloaded" and not counted. It then asks `[y/N]`; `--yes` answers yes, and
  without a terminal and without `--yes` it stops and says what to add. It stops too when the disk is too small.
- **Each item is optional.** `--no-hybrid` (no MTP head), `--no-drafter` (new: no draft at all, `--drafter none
  --no-drafts`), `--no-models` (nothing downloaded; the service now starts when the model and the draft are already on
  the Mac), `--model DIR`, `--drafter REPO|DIR` (new) and `--mtp-head FILE` (new: a head already written).
- **`pli bench`** runs the public suite pli-bench-v1 against the local server (the port of `pli setup`), thinking off.
  The bench and its texts (811 KB) are now part of the package (`src/tensorfold/bench/`); `tools/pli_bench.py` runs the
  same code from a clone, with standard Python only. `tools/quantize_mtp_head.py` likewise runs
  `tensorfold.install.mtp_head`, which `pli setup` uses.
- **`pli` alone** prints a short help with examples instead of an argument error.
- **Checkpoint at the start of the last message** (`PLI_LAST_MESSAGE_CHECKPOINT`, 1,024 in the `m1-ultra` profile,
  off in the code; in production since 3 Nov). Besides the history checkpoint, PLI keeps the state at the start of
  the prompt's last message (the turn marker that opens the reply, looked for backwards in the history) when at least
  N tokens are read there since the resume. The next request that keeps the history but not that message (a question
  replaced, a reply thrown away, a sub-agent on the same history) starts from there instead of from the previous
  request's checkpoint. Same text. On the agent session of the public suite pli-bench-v1: the same 8 replies, prompt
  served from the cache 73.5% instead of 58.9%, sum of the times to the first token 238.0 s instead of 357.3 s
  (-33%), writing speed unchanged. The production name `PLI_DEBUT_DERNIER` is still accepted.

## 4.0.0 (2026-11-02)

Everything below went into production between 27 October and 2 November 2026, after 3.7.0, and is switched on by
`--profile m1-ultra` unless the line says otherwise. Measured on the validated machine: Mac Studio M1 Ultra, 64 GB,
serving Qwen3.8-27B in MLX 4-bit with the DFlash2 draft. Terms as in 3.7.0: "same text" means replies identical to the
previous production; "text can change" means another arithmetic. Every setting is described in
[docs/switches.md](docs/switches.md).

- **Profile `m1-ultra` aligned with production.** The profile now sets the production settings listed below (draft
  trees, copy or chain per stream, extended draft list, draft kernels, S3b). MLX's command-buffer limits
  (`MLX_MAX_MB_PER_BUFFER=500`, `MLX_MAX_OPS_PER_BUFFER=100`) go in the server's environment, not in the profile,
  because MLX reads them when it starts, before any profile: `install.sh` sets them. Machine settings
  (`PLI_HYBRID`, `PLI_MTP_HEAD`, hosts, paths) stay in the environment, as before.
- **Public bench suite** (`tools/pli_bench.py`, suite `tools/bench_data/suite-v1.json`). By default the bench sends
  fixed public requests: single requests of about 1k to 96k tokens and an agent session from 6k to 50k tokens, cut from
  a fixed corpus (PLI 3.7.0's own source code) checked by SHA-256, so every run sends the same requests (only a
  first-line tag changes). `--free` keeps the
  older mode (free sizes, filler text).
- **English names for the new settings.** The 23 settings added in production under French names have English names
  (for example `PLI_BROUILLON_ATTN` is `PLI_DRAFT_ATTN`, `PLI_LISTE_PLUS` is `PLI_EXTENDED_DRAFT_LIST`,
  `PLI_DEVERSE_INVITE` is `PLI_SPILL_PROMPT`); the old names are still accepted (165 in all), and the new name wins
  when both are set. The draft list shipped with PLI is `extended-list-2048.txt` (`liste-plus-2048.txt` is still
  found). The log lines of the new diagnostics use English keys.
- **Draft trees for short questions** (`PLI_KFLASH=1`, `PLI_TREES_ATTM=1`, `PLI_HYBRID_TREES=1`,
  `PLI_TREE_DELTANET=3`, `PLI_TREES_SHORT_ONLY=1`, `PLI_TREES_TOKEN_THRESHOLD=4.5`; 28 to 30 Oct). These settings were
  in 3.7.0, off. DFlash2 proposes trees built in KFlash's manner (method of [Kapitsa](https://github.com/nace-ai/kapitsa),
  Apache-2.0), verified through `attm`, only for prompts of at most 3 messages (`PLI_HYBRID_MESSAGES`); agent
  sessions keep the chain. On a short question the chain stays while it keeps 4.5 tokens per round or more on average.
  The model verifies every draft.
- **Copy of the context or DFlash2's chain, chosen per stream** (`PLI_HEAD_BEFORE_COPY=3`, 30 Oct). When a copy of
  the context and DFlash2's queued chain are both ready, both are scored afterwards on the written text, and the copy
  goes first while the moving average of copy minus chain stays above minus 1 token (`PLI_HEAD_COPY_WEIGHT`,
  `PLI_HEAD_COPY_THRESHOLD`). The MTP head keeps one draft when a copy exists, as before (measured with the MTP head
  weighed too: 1 to 6 ms more per round). The model verifies every draft.
- **Extended draft lists** (`PLI_EXTENDED_DRAFT_LIST=extended-list-2048.txt`, 29 Oct). The drafts never propose a token
  outside their reduced lists: 3.89% of the written tokens were outside DFlash2's list and 5.6% outside the MTP
  head's. With 2,048 learned ids, 0.69% and 1.20%. The file shipped with PLI holds 2,515 ids (the 2,048 most frequent
  for each list, together); they are added once at load, with no extra product per round. Same text: the verify stays
  exact.
- **Draft attention in a PLI kernel** (`PLI_DRAFT_ATTN=1`, 31 Oct and 1 Nov). 127 µs per layer in a chain against
  302 µs for MLX's attention; the cache and the block are read where they are, without the two concatenations of
  2,057 rows per layer (same bits as the concatenated read). Bits other than MLX's, with an error against an f32
  reference at most MLX's; the model verifies the drafts.
- **Draft projections stacked** (`PLI_DRAFT_STACK=1`, 1 Nov). [q | k | v] of the block and [k | v] of the context in
  one call of the 8-row product each, `fc` relaid out, the draft head in two calls instead of three. Same bits,
  checked at install.
- **Draft norms and RoPE in one kernel** (`PLI_DRAFT_QKR=1`, with `PLI_DRAFT_STACK`, 1 Nov). The q and k norms, the
  RoPE and the reordering of the draft in one kernel per layer instead of about eight MLX operations. Same computation
  as MLX 0.32.3: same bits.
- **Draft top-16 split** (`PLI_DRAFT_TOPK=1`, 1 Nov). The top-16 of the draft's lattice over 16 slices per row:
  152 µs instead of 241 in a chain. Same outputs.
- **Draft block projections relaid out** (`PLI_RELAYOUT_DRAFT=1`, 30 Oct). The draft's projections that read only the
  8 rows of the block use the relaid-out 8-row kernel (302 GB/s against 220). Each matrix is checked at the same bits.
- **`attm` fragments loaded with `simdgroup_load`** (`PLI_ATTM_LOAD=1`, 30 Oct). Keys and values laid out without
  permutation. Same products in the same order: same bits.
- **8-row tables prepared over several threadgroups** (`PLI_V8_PREP2=1`, 31 Oct). The preparation of `PLI_V8`'s
  tables cost 4.3 ms per round; it is now spread over several threadgroups per row. Same bits.
- **Wider MLX command buffers** (`MLX_MAX_MB_PER_BUFFER=500`, `MLX_MAX_OPS_PER_BUFFER=100`, in the environment,
  1 Nov). Same output.
- **Reduced prompt reading at long context, S3b** (`PLI_S3B=1`, `PLI_S3B_THRESHOLD=65536`, 2 Nov). In the production
  trace of 1 November, prompt chunks of 256 to 511 rows at 64K tokens and more were read at 130 tokens/s, the
  attention taking 31 to 47% of it. Chunks of at most 2,047 rows past 65,536 positions now go through the anchored
  reduced attention of decode (`PLI_REDUCED_ANCHOR`); anchors are bounded at the start of the request's prompt
  reading, and the full layers are read whole. Text can change.
- **Checkpoint at the fork point** (`PLI_FORK_MIN`, 4,096 tokens, on by default; 27 and 28 Oct, from TensorFold
  [#1291](https://github.com/ashhart/TensorFold/pull/1291)). A long document sent again in the same message with another question was read in full each time. The state
  where the prompt forks inside its last message is now kept; without a resume, the fork is looked for in the kept
  prompts. Same text.
- **KV layers shared by checkpoints counted once** (with `PLI_KV_SHARING`, no switch, 29 Oct). A prompt checkpoint and
  the reply end of the same turn hold the same KV layers, and each one counted them in full (3.78 GiB each at 59,269
  tokens). In one night, 35 next turns of 110 beyond 30,000 tokens read almost everything again; in 15 of them, the
  reply end had evicted the prompt checkpoint under the 8 GiB budget. Each KV layer now counts once.
- **Evicted prompt checkpoints go to disk** (`PLI_SPILL_PROMPT`, on by default; 29 Oct). A reply end no longer
  counts as a continuation of the prompt checkpoint (an agent that rewrites its tool calls never resumes it), so an
  evicted prompt checkpoint is spilled to disk. In one night, 15 turns of 30,000 to 80,000 tokens read everything
  again for want of it. Cost: a write of about 1 s for 4 GiB (median 0.75 s), 0.4 s to read it back.
  `PLI_SPILL_PROMPT=0` keeps the former behaviour.
- **Disk looked at again at the first chunk** (`PLI_REREAD_DISK=1`, 30 Oct). Between a request's opening and its
  first chunk, its checkpoint can be evicted by another request's (two agent sessions taking turns); it is then read
  back from disk instead of reading the whole prompt again.
- **Fix: a system message in mid-conversation is accepted on the Anthropic route** (no switch, 29 Oct). When the
  model's template does not keep a system message in place, it is given the `late_system` role ("user"), as the
  OpenAI route already did. Before: error 400.
- **Diagnostics and trials, off by default.** `PLI_ROUND_PROFILE=1` (time of a decode round step by step, with
  `PLI_ROUND_PROFILE_N` and `PLI_ROUND_PROFILE_BUCKET`), `PLI_LATTICE_LOG=<folder>` (DFlash2's lattices, to replay
  chain and trees offline), `PLI_COPY_LOG=<file>` (copy rounds), `PLI_TREE_GAIN_MIN=x` (a KFlash tree only if it
  brings x expected tokens more than the chain). The three logs do not change the text.
- **Overall**, on 18 recorded real requests (agent sessions up to 50K tokens): 51.4 tokens/s on 30 October, 55.9 on
  1 November.

## 3.7.0 (2026-10-28)

Pre-release. Everything below went into production between 21 and 28 October 2026, after 3.6.0, and is switched on by
`--profile m1-ultra` unless the line says otherwise. Measured on the validated machine: Mac Studio M1 Ultra, 64 GB,
macOS 27.0.1, Python 3.12, MLX 0.32.3, mlx-lm 0.32.0, serving the production 27B in MLX 4-bit ([sabicap/sabi-noema](https://huggingface.co/sabicap/sabi-noema), a
Qwen3.8-27B fine-tune with the same architecture) with the DFlash2 draft of Qwen3.8-27B. Contexts are in tokens; a
round is one verify pass of the model over its draft rows; prompt reading is the prefill. "Same text" means replies
identical to the previous production; "text can change" means another arithmetic, judged on whole conversations.

- **English names everywhere.** Every setting is now a `PLI_*` variable with an English name (for example
  `PLI_HYBRIDE` is `PLI_HYBRID`, `PLI_MTP27` is `PLI_MTP_HEAD`, `TF_FLASH_MTP` is `PLI_FLASH_MTP`,
  `TENSORFOLD_MEMORY_LIMIT_GB` is `PLI_MEMORY_LIMIT_GB`); the 142 old names and the old values of
  `PLI_PREFILL_CPU_MODE` are still accepted, and the new name wins when both are set. Files, modules, code, comments
  and messages are in English (`tensorfold/pli/profils/` is `profiles/`, `commun.env` is `common.env`, still read).
  The step trace, the event journal and the `[tensorfold] done` line use English keys; their readers in PLI accept the
  old keys, but a tool of your own that parses them needs updating. Prefix snapshots written by 3.6.0 are not reused
  (they are rebuilt). Same text: checked on 18 real requests against 3.6.0.
- **Install script** (`install.sh`): checks the Mac, installs PLI in its own virtual environment, downloads the
  models, writes the MTP head in 4 bits and runs PLI as a per-user service; `start`, `stop`, `restart`, `status`,
  `logs` and `uninstall` (which removes everything, and asks before deleting each model).
- **Benchmark script** (`tools/pli_bench.py`): prompt reading and writing speed by context size, and memory, against
  any OpenAI-compatible server, to compare engines on the same Mac.
- **MTP head in 4 bits** (`tools/quantize_mtp_head.py`): writes the 27B's MTP head in 4 bits for `PLI_MTP_HEAD`
  (810 MiB to 228 MiB); same file as the one served in production.
- **`pli service install --env PLI_...`** is accepted (only `TENSORFOLD_*` and `TF_*` were).
- **Step trace for debugging** (`PLI_TRACE_DIR`, off by default, 21 Oct). One JSON line per engine step of each
  request, switched on without a restart by `python -m tensorfold.pli.trace on --dir <dir> [--duration 24h]`, read
  with `python -m tensorfold.pli.trace_analysis`. Same text. Nothing is written while off; cost while on not measured.
- **Short decode attention calls in 128-key slices, and a kept page choice** (`PLI_ATTM_SHORT=12288`,
  `PLI_REDUCED_ROUNDS=4`, 21 Oct). Below 12,288 keys, `attm` works in 128-key slices instead of 512, about 2.4 ms less
  over 16 layers at 6,000 tokens. Beyond 20,000 tokens, the reduced attention recomputes its page choice one round in
  4; the choice cost 10 ms per round at 22,000 tokens and 20 ms at 60,000. Time per round lower by 6% (first
  repetition) to 17% (mean) from 20,000 to 90,000 tokens. Text can change: judged on three whole conversations, no turn
  became wrong where the previous production was right.
- **Neural Engine weights requantized on the CPU** (`PLI_ANE_REQUANT_CPU=1`, `PLI_ANE_REQUANT_THREADS=8`, 22 Oct). The
  4-bit weights of the next layer's Neural Engine share are requantized by the CPU (NEON, 8 threads) while the GPU and
  the Neural Engine compute the current layer, instead of by a GPU kernel (5.45 ms per layer). No extra memory. Same
  bits: 100% of codes and scales equal to the GPU kernel's, 3 replies of 3 identical with the arbiter off. Prompt
  reading faster by 1.3% (206.2 to 208.9 tok/s, 5 requests of 22,000 to 90,000 tokens) and by 2.2% with the arbiter off
  (244.5 to 249.8 tok/s); decode unchanged.
- **Neural Engine kept awake, and its share sized by chunk** (`PLI_ANE_KEEPAWAKE=1`,
  `PLI_ANE_STEPS=256:14,512:16,1536:18`, 23 Oct). The Neural Engine slows down after about 2 ms idle (a 256-row call
  takes 10.1 ms without rest and 19.9 ms after 20 ms; 2,048 rows, 40.1 and 53.0 ms). Tiny calls now keep it busy
  between two shared layers, its units follow the chunk size, and chunks from 256 rows get a share. On 26 real agent
  turns: prompt reading 4.1% shorter (8.2% on chunks of 256 to 511 tokens, 6.2% on 512 to 1,023), decode unchanged
  (1.4% difference); 15 replies of 26 identical, the others of equal quality (text can change). Replayed on the
  production trace, the median time to first token goes from 3.99 to 3.78 s.
- **Heartbeat between requests** (`PLI_HEARTBEAT_MS=500`, 23 Oct; idea from Kapitsa [#206](https://github.com/nace-ai/kapitsa/pull/206)). Metal releases the wired
  weights about 2 s after the last GPU work, and the next pass waits for them: 256 ms instead of 29 ms on 14 GiB of
  weights. A tiny GPU job every 500 ms, for 600 s after the last activity, keeps them wired. After 6 s idle, the extra
  time over a warm start is 70 ms instead of 219 ms for a new 100-token request, and 137 ms instead of 451 ms for 355
  tokens. Same text. Cost at idle: about 0.3 W (5.26 W against 4.98 W). A Neural Engine call at the same pace
  (`PLI_HEARTBEAT_ANE`, idea from Kapitsa #243) gained 5 ms more and stays off.
- **Last prompt tokens read with the chunk before** (`PLI_MERGE_TAIL=16`, 23 Oct). The tokens after the last checkpoint
  (the reply prompt, 7 tokens in general) took a full pass of the model of their own; they are now read in the same
  pass as the chunk before, and the checkpoint is taken from that pass. On 10 recorded agent turns: prompt reading 3.1%
  shorter (131 ms less per turn), checkpoints always resumed at the next turn, replies of equal quality (text can
  change).
- **One 8-row tile per threadgroup in the exact row kernel** (`PLI_TILES_PER_GROUP=1`, 23 Oct). Same bits. Decode
  faster by 12.6% with 3 requests writing together and by 5.5% with 4; a 16-row round takes 133.5 ms instead of 150.3;
  same text alone, at 3 and at 4 requests.
- **Reply kept for the next agent turn** (`PLI_KEEP_REPLY=1`, 23 Oct). With thinking off, the reply prompt writes an
  empty think block that the history then drops, so the next turn read the reply again: 78,000 tokens and 735 s over
  29.6 hours of the production trace. The chat template now keeps the empty block, and the engine keeps the state it
  wrote. In an 8-turn agent loop, the previous reply was resumed 6 turns of 7, and reading of turns 2 to 8 was 11.6%
  shorter; 0 errors, tool calls well formed. Text can change.
- **Reduced attention anchored to positions** (`PLI_REDUCED_ANCHOR=32`, 23 Oct). The reduced attention chose its pages
  on all rows of a round, drafts included, so the text depended on the drafts. A row now reads the pages chosen at its
  anchor (every 32 positions, on the anchor row's query only), for rounds of any width. The text no longer depends on
  the drafts: same text between repetitions and between two servers, 0 unstable. At 44,611 and 58,223 tokens, decode
  ran 3.2% and 1.0% faster; remeasured at equal rows per round, rounds were 0.1% and 3.7% faster. Text can change
  against the previous production.
- **Fix: evicted conversations are written to disk again** (24 Oct, no switch). PLI's runtime attributes on the caches
  (kept page choice, anchors) made every spill of an evicted conversation fail: 0 of 300 succeeded. Conversation
  snapshots now skip them. After the fix: 47 conversations written, 0 failures.
- **Hybrid draft: DFlash2 and the 27B's MTP head together** (`PLI_HYBRID=1` and `PLI_MTP_HEAD=<file>`, machine
  settings; `PLI_MTP_DRAFTS=7`, `PLI_MTP_WINDOW=2048`, set by the profile; 24 Oct). Each conversation drafts
  with the one that gives it more tokens per millisecond, measured on its own rounds, and the choice follows the
  conversation from one request to the next. The model verifies every draft: same text. The MTP head keeps only its
  last 2,048 rows, which bounds its memory; a checkpoint at the end of the prompt is now kept with the MTP head.
  Measured on 18 real requests (sabicap/sabi-noema) replayed one after another: 51.4 tokens/s on average against 47.4 for DFlash2
  alone, faster by 8.4%, same text 18 of 18; with the MTP head in 4 bits (`tools/quantize_mtp_head.py`), 51.3.
- **FALCON capture, experimental** (`PLI_FALCON=<directory>`, `PLI_FALCON_MAX_GB`, machine settings, off by default;
  25 Oct). For the rows kept at each verify round, it writes the hidden state, the token the model chose and its 20 top
  logits, with the last 2,048 prompt rows, as training examples for the MTP head (idea from [EAGLE-3](https://arxiv.org/abs/2503.01840)). Nothing in the
  computation changes: same text, 18 requests of 18. Rounds slower by 0.7%, prompt reading slower by 1.5%. Two days in
  production: 0 errors, 0 lost rows. About 10 MB per request read, bounded by `PLI_FALCON_MAX_GB`. No learned head is
  served.
- **FALCON prompt rows in 8 bits** (`PLI_FALCON_CONTEXT8`, on by default with `PLI_FALCON`; 27 Oct). Prompt rows, 89%
  of the captured rows, are stored in 8 bits per row (scale = max abs / 127) instead of 16: half the space. Measured on 18 requests:
  same text 18 of 18; examples folder 149 MB instead of 277 MB; against capture off, rounds slower by 1.2% and prompt
  reading slower by 1.6%. Tokens accepted per anchor when the head is judged on these rows: 3.53 against 3.51 in
  16 bits.
- **Kapitsa's 8-row verify GEMM** (`PLI_V8=1`, 28 Oct). Verify rounds of up to 128 rows run the 4-bit GEMM ported from
  [Kapitsa](https://github.com/nace-ai/kapitsa) v0.2.23 (Apache-2.0, see THIRD_PARTY_NOTICES.md), reading PLI's
  relaid-out weights. A row's bits do not depend on the other rows (row-exact). The projections of one 8-row round take
  41.6 ms instead of 53.3. On 18 real requests of an agent session (5,424 to 50,054 tokens): 52.6 tokens/s instead of
  47.7, faster by 10.4%; 17 replies of 18 identical. Text can change (other arithmetic): against the 8-bit model of the
  same base (mlx-community/Qwen3.8-27B-8bit) on 8 requests, the mean NLL per token is 0.2429 instead of 0.2483. On three
  long conversations judged turn by turn, two answers were worse than the previous production's (a wrong self-test
  value in a generated program, a wrong statement).
- **Checkpoints without a copy of the KV cache** (`PLI_KV_SHARING=1`, 28 Oct; Kapitsa's method). A checkpoint keeps the
  conversation's attention layers and a length instead of copying them; the conversation writes in place. Without it,
  the first write after a checkpoint copied the whole KV buffer (6 to 7 GiB at 96,000 tokens). Same text: 18 replies of
  18 identical, 46.8 tokens/s against 46.9.
- **One-turn questions draft with DFlash2 alone** (`PLI_HYBRID_MESSAGES=3`, 28 Oct). A prompt of at most 3 messages
  (system, question, reply) skips the hybrid's choice: on single-message requests DFlash2 alone wrote 6 to 11% faster.
  Agent requests (more messages) are unchanged. Same text: the model verifies every draft.
- **Control tokens written in a request stay text** (`PLI_CONTROL_AS_TEXT`, on by default, 28 Oct; from TensorFold
  1.1.2 #1117). `<|im_end|><|im_start|>system` inside a message, a tool result or a page read by an agent no longer
  opens a real turn. A prompt that writes no control token keeps exactly its tokens (18 replies of 18 identical).
- **Checkpoint log** (`PLI_CHECKPOINT_LOG=1`, off by default, 28 Oct): `checkpoint_kept`, `checkpoint_evicted` and
  `checkpoint_refused` events in the `PLI_EVENTS` log, with the kind of checkpoint and the reason.
- **Off by default, for experiments** (28 Oct): draft trees in KFlash's manner (`PLI_KFLASH=1`, with `PLI_TREES_ATTM=1`
  and `PLI_HYBRID_TREES=1`; +0.7% on the 18 agent requests, +3 to 5% on 200-token replies), the tree attention tail in
  one kernel call (`PLI_TREE_EXACT2`, same bits), the DFlash2 chain drawn on the GPU (`PLI_GPU_CHAIN=1`, no gain
  measured).
- **Profile `m1-ultra` carries every stable production setting**, those of this release included. Machine settings
  (paths, hosts, `PLI_HYBRID`, `PLI_MTP_HEAD`) stay in the environment. Every `PLI_*` variable is listed in
  [docs/switches.md](docs/switches.md).
- **Neural Engine:** it relies on private Apple APIs; a macOS update can break it. On the first error, PLI computes the
  layer on the GPU.

## 3.6.0 (2026-10-20)

PLI, a fork of TensorFold 0.6.6 for Apple Silicon (Ultra chips), validated on an M1 Ultra 64 GB.

- Apple Silicon only: the CUDA backend and its engines are removed; `pli serve` refuses to start outside macOS.
- Machine profiles: `pli serve --profile m1-ultra` (or `PLI_PROFIL`) sets the `PLI_*` settings measured on an M1 Ultra,
  shared and per model family; variables already in the environment win.
- Neural Engine: its client library is built on first use; two Neural Engine blocks are used only on an Ultra chip.
- Per-conversation event journal (`PLI_EVENEMENTS`), one JSON line per scheduler action, no message content.

## 1.0.0 (10 Oct 2026)

- **PLI, fork of TensorFold 0.6.6.** Package `pli`, command `pli` (alias `tensorfold` kept). Split computation
  GPU + CPU on Apple Ultra chips (`PLI_ARBITRE=1`, now `PLI_ARBITER=1`), validated on M1 Ultra. `pli update` disabled.
  The sections below are TensorFold's.

## 0.6.6 (6 Oct 2026)

- **`--name-priority ID=background` on the CUDA server.** A request that names that served id (`--name` or an
  `--alias`) and sends no `priority` of its own is served as `priority: "background"`, so it yields to foreground
  requests. A request's own `priority` always wins. This suits clients that can choose a model id but cannot add a
  field to the request, such as batch extractors. On one DGX Spark, a foreground request behind four background ones
  got its first token in 0.2 s instead of 21 s, with replies unchanged (#445). Thanks to @philip-pentatonic.

## 0.6.5 (3 Oct 2026)

- **Qwen3.6-35B-A3B drafts with its own MTP layer on Macs,** as on CUDA. Chains of up to four drafts are verified in
  the lane rounds, alone or with other streams, and each round's depth comes from the Mac's measured costs, with plain
  rounds where drafts do not pay. Replies equal `"draft": false`. On an M3 Ultra it decodes 1.5-2.1x `--no-drafts` on
  short prompts, thinking on or off, and 1.3x at a 28,400-token prompt with the same time to first token: 1.5-1.7x
  mlx-vlm 0.7.4 with its MTP drafter. DFlash v1 stays available with `--drafter z-lab/Qwen3.6-35B-A3B-DFlash`.
- **Nemotron on CUDA drafts as deep as its rows pay for.** The MTP head drafts up to 15 levels, one CUDA graph each,
  and a chain stops where the next verify row would cost more time than its draft is expected to save. Window and
  level costs are measured at startup on each GPU. Against 0.6.4's default this decodes about 6% faster on one DGX
  Spark and on two, and 7% faster on an RTX PRO 6000, with replies equal to `"draft": false`. `--mtp-confidence` sets
  a fixed floor instead.
- **Nemotron verify windows on M5 Macs read each routed expert once from two rows** (eight before). One stream decodes
  1-3% faster at three to five lanes, with the same tokens.
- **API keys on both servers:** `--api-key` (repeatable), `--api-key-file` (one key, or `label: key`, a line; reread on
  `SIGHUP`) or `TENSORFOLD_API_KEY`. Requests send `Authorization: Bearer` or `x-api-key`. `/health` stays open;
  `/metrics` needs a key unless `--metrics-open`.
- **Thinking stays the chat template's default, and the server says so:** one startup line names `--no-thinking`, and
  a reply that reaches `max_tokens` before it leaves its think block (empty `content`, all `reasoning_content`) gets a
  warning line.
- The test suite collects on Macs that have PyTorch but not Triton.

## 0.6.4 (3 Oct 2026)

- **Flash Next on two DGX Sparks serves concurrent requests.** `--parallel N` with two CUDA ranks runs every stream
  in shared lane rounds, and drafted replies still equal one-token decoding (#141, #180, #219, closes #123). The two
  ranks now share one communicator interface, so a faster transport can plug in without touching the engines.
  Thanks to @BHCC2025, @jschmied, @plotarmordev and @jayleaton.
- **Startup memory on Macs:** prompt chunks are probed smallest first, and a larger probe runs only when its worst
  case fits the memory budget (#271). Thanks to @boxabirds for the report.
- **`tensorfold plan`** prints the model and context budget before any weights load (#281), and both `/metrics`
  routes report the process memory footprint (#277). Thanks to @akol1.
- Flash Next on CUDA builds its extensions before loading weights (#202) and gathers prompt n-gram rows before
  waiting on earlier copies (#201). Thanks to @mcclanahanaman.
- Flash Next on CUDA rereads EXL3 n-gram tables after warm-up and pins their page runs (#254). Thanks to
  @grearjake-star.
- `--decode-share` takes effect on Flash Next CUDA: shared prompt rows are sized from completed stand-alone passes
  (#248, closes #230). Thanks to @simon-lin88.
- CUDA tree attention folds its partial sums in fp32 groups. Precision is unchanged, but long replies can differ
  from 0.6.3 in the last bits (#268). Thanks to @Arminova.
- Flash Next on CUDA refuses cards below sm_120 at startup, before loading weights, and its CUDA tests skip there
  (#261). Thanks to @mcclanahanaman for the report.
- `docs/recipes/README.md` lists the minimum card and Mac memory for each family (#139). Thanks to @tomByrer for
  asking.
- Benchmark helpers fail early on Python below 3.11 (#276). Thanks to @akol1.
- Tests collect without the optional MLX or terminal-interface dependencies (#288), and the docs say `--alias`
  works on the CUDA server too (#289). Thanks to @plotarmordev.

## 0.6.3 (2 Oct 2026)

- **Nemotron on M5 Macs: copied text verifies up to 64 tokens a round.** A lone stream's copy window grows from 16 to
  64 rows while each lands whole, a window attends in one call, and wide windows keep their Mamba states every 8th
  row. On an M5 Max, one stream editing code decodes about 1.4x faster (550 to 760 tok/s, and up to 1,000 on a cool
  machine), prose and code writing gain 1.6-3.3%, and drafted replies still equal one-token decoding.
- **Anthropic Messages API.** `/v1/messages` and `count_tokens` on the Mac and CUDA servers, with streaming, tools,
  thinking and cache usage (#223, closes #168). Thanks to @kky42.
- **Control Room.** `tensorfold service` runs a model as a launchd service, and `tensorfold tui` shows its decode and
  prefill speed and its connections live.
- **Flash Next on Macs before M5:** `TF_FLASH_DENSE=matrix` runs every dense projection on the matrix units, and the
  fused DeltaNet stack uses the matrix kernel by default (#149). Thanks to @gilby.
- **Vision:** `--vision-offload` keeps the CUDA image tower in host RAM between images (#187, closes #185); image
  parts are accepted inside tool results (#235); `--vision-image-tokens` lets many images share a larger budget
  (#239); Flash Next on CUDA takes video (#240). Thanks to @barelyworkingcode and @MiaAI-Lab.
- **CUDA:** `/v1/decisions` on Flash Next scores labels from the prompt's last logits, the questions filling in one
  pass (#232); `TENSORFOLD_PREFILL_ROWS` sets the prompt piece rows (#238); NVFP4 and FP8 prompt rows add a tile's K
  slices in one block, with the same bits (#242); `TENSORFOLD_MEMORY_RESERVE_GIB` moves the memory floor the startup
  keeps free, by default a tenth of the pool and at least 4 GiB as before (#165); Flash Next's chain kernel takes
  5-17% less time at 2 to 8 rows, with the same bits. Thanks to @Mirrdhyn, @MiaAI-Lab, @jschmied and @eleqtrizit.
- **Server:** chunked request bodies are decoded before JSON parsing (#244); `/tokenize` and `/detokenize` carry
  vLLM's fields (#237); `chat_template_kwargs.thinking` is heard as `enable_thinking`, and GLM-5.3 keeps earlier
  turns' reasoning (#236); streaming usage gets its own chunk when `stream_options.include_usage` asks for it (#216);
  malformed tool-call history renders safely (#233); `/metrics` times each request's decode (#269); `/health` carries
  the live decode and prefill speed. Thanks to @JordiPosthumus, @MiaAI-Lab, @salmanarshad321 and @sxuff.
- **Fixes:** an EXL3 layer with no bias no longer faults in the split-K reduction (#186); an 8-bit `lm_head` keeps its
  format in Qwen3.6's MTP draft head on CUDA (#270); a lone Flash Next stream's cache growth on CUDA stays inside the
  explicit memory budget; two ranks refuse to start with different prompt rows. Thanks to @barelyworkingcode and
  @BHCC2025.
- **Models moved to the TensorFold Hugging Face org.** `Vontra/<name>` ids redirect, and the tree now names
  `TensorFold/<name>`.
- **Contributing.** `CONTRIBUTING.md` says what a pull request needs to land and how it lands, and new pull requests
  open with a receipt template.

## 0.6.2 (2 Oct 2026)

- **Flash Next on Macs at 64k-128k.** On an M3 Ultra, one stream runs 1.2-3.4% faster at 64k and 3.9-5.5% at 128k,
  with the same tokens: a window's n-gram ids are hashed on the GPU, and the chain's first step is built while the
  GPU verifies.
- **27B with several streams on CUDA.** The GDN tree kernel takes 8-35% less time with the same bits. On an RTX PRO
  6000 at its 250 W limit, 4 and 8 streams of the NVFP4 27B decode 1.1-4.2% faster.
- **Fixes:** the config check accepts the FP8 n-gram table in NVIDIA's MIXED_PRECISION Flash Next export; GLM-5.3
  on CUDA counts its drafts in `/health`, `/metrics` and replies, and names a mixed-bit EXL3 checkpoint when it
  refuses one; a client that leaves is noticed past file descriptor 1023; the CUDA server prints a line a request, as
  the Mac server does; a failed snapshot write no longer leaves its partial file; Gemma 4's QKV kernel reserves its
  1024 threads for M1, M2 and macOS VMs, and a VM's GPU is no longer taken for an M5.

## 0.6.1 (1 Oct 2026)

- **NVFP4 checkpoints in their own math.** `nvidia/Qwen3.8-27B-NVFP4` runs the 4-bit activations its checkpoint
  names, as vLLM does; `--precision full` runs 16-bit activations against the same weights. On an RTX PRO 6000 at its
  250 W limit, one stream decodes 1.4-2.0x vLLM and prompts fill at 0.95-0.97x its speed.
- **Waiting prompts fill together on CUDA.** With `--parallel`, prompts that arrive together now share one prefill
  forward instead of filling one a round: on an RTX PRO 6000 at its 250 W limit, 8 streams of the 27B run 1.14-1.36x
  faster and the slowest first token comes in 0.05-0.10 s instead of 0.4-0.8 s (1.6 s to 0.15 s on a DGX Spark), with
  the same replies.
- **More 27B tokens with several streams on CUDA.** Streams plan on their measured round cost, and wider lane blocks
  on RTX PRO and RTX 50 cards add 4-8% at 8 streams, with the same tokens.
- **Flash Next on Macs at long context.** One stream runs up to 9.6% faster on code at 64k and 10.2% on chat at 128k
  on an M3 Ultra, with the same tokens.
- **`/v1/decisions`** scores a choice, a score or a yes/no from the next-token logits, on the shared prompt lanes.
- **Flash Next on CUDA:** image input, forks that resume from their shared prefix, shared system prompts copied
  instead of filled again, and short prompts admitted while a long one fills.
- **RTX cards without Docker:** pip alone installs and builds the CUDA kernels. Native Windows is in as an
  experimental host layer, not yet run on Windows hardware.
- **Fixes:** a refused request no longer breaks the next one on its connection; Gemma 4 thought blocks stay out of
  replies with thinking off; an unnamed reasoning effort goes to the nearest named level; mlx-lm 0.32 support.

## 0.6.0 (30 Sep 2026)

- **RTX 40 cards.** CUDA now runs on compute capability 8.9 (Ada). On one RTX 4090 the 27B serves with DFlash2 in a
  40,182-token window, exact: drafted replies equal serial ones, and a resume or resend gives a fresh run's reply.
  Prompts fill at 2.4-2.6k tok/s from 2k to 32k tokens, and replies decode at 64-86 tok/s. On one GPU the 27B's kept
  prompt states give way, oldest first, when a live reply needs the room, so admission counts one live window.
- **Prompts fill inside the decode rounds.** With `--parallel` on CUDA, Flash Next prefills a queued prompt in the
  same forward as the live replies instead of stopping them: on one DGX Spark, first tokens came 2.8-3.1x sooner than
  on 0.5.0. On Macs several prompts fill side by side, the fewest tokens left first: on an M3 Ultra, short requests
  queued behind a long prompt got their first token in a median 9.9 s instead of 113 s (the slowest 11.7 s, not 130).
- **Conversations resume on more engines.** An identical resend or the next thinking turn now resumes from the kept
  prompt state on Flash Next, Qwen3.6, Nemotron and GLM-5.3 on CUDA, as the 27B did, with a fresh run's reply: an
  18.7k-token Flash Next resend went from 8.3 s to 0.08 s. On Macs the prompt cache keeps each conversation's newest
  checkpoint and grows into memory the model leaves idle.
- **CUDA prompts run at bf16 by default.** Against an fp32 reference the 27B's prompt rows are 20x closer than with
  FP8 (KL 0.0031 against 0.0624). `--prefill-fp8` keeps 0.5.0's faster FP8 prompts for those who want them.
- **Tool calls for agents.** The CUDA server streams tool-call arguments as the model writes them (the longest
  silence in a long call fell from about 30 s to half a second), Python-spelled values like `False` and `None`
  decode to their schema types, Gemma 4's bare tool calls parse (#121), and a prompt past the context window gets
  OpenAI's `context_length_exceeded`, so clients compact instead of retrying.
- **More checkpoints.** GLM-5.3 on Macs reads 8-bit and Q8_0 GGUF checkpoints and takes images, and has an opt-in
  float32 activation mode (bf16 stays the default). On Macs, Flash Next loads oMLX's oQ checkpoints with scaled
  n-gram tables; on CUDA it reads consolidated EXL3 tables and block-scaled FP8 linears.
- **Faster.** Flash Next NVFP4 decodes 4.9-7.1% faster and fills prompts 15-16% faster at 2k-16k, bit-identical. On an
  M5 Ultra the 27B with DFlash2 on an oQ4e checkpoint went from 33 to 131-160 tok/s. Before M5, a lone stream's copy
  windows widen to 128 rows (an M3 Ultra edit ran 190 -> 233 tok/s), and Flash Next's concurrent rounds use the matrix
  units. GLM-5.3's DFlash2 reads only its sliding window: 8% faster at 52k, and 3.8 GiB lighter a rank on two Sparks.
- **Operations.** Prometheus `/metrics` on both servers, `TENSORFOLD_MEMORY_RESERVE_GIB` for the CUDA startup
  reserve, `--checkpoint-slots` for the 27B's concurrent decoder on CUDA, and an idle GLM rank no longer spins.
- **Apache-2.0.** TensorFold is licensed under the Apache License 2.0 from this release, which adds an explicit patent
  grant from contributors. Releases up to 0.5.0 stay MIT, and code written before 0.6.0 keeps its MIT notice in
  `LICENSES/MIT.txt`.
- **Community pull requests.**
  - Resumed resends and thinking turns on Flash Next (#124). Thanks to @benthecarman, and to @Arminova for the
    GB10 measurements and the third-resend test.
  - Streamed tool-call arguments on CUDA (#114), `--alias` on CUDA (#111) and Python-spelled tool parameters
    (#135). Thanks to @olexale, @philip-pentatonic and @outcastofmusic.
  - GLM-5.3 on Macs: image input (#101), 8-bit and Q8_0 GGUF checkpoints (#119), float32 activations (#118) and
    the backbone wiring notes (#120). Thanks to @mgoldwasser and @feni6.
  - GLM-5.3 on two Sparks: lighter prompt buffers, the EXL3 estimate, the MTP setting, the idle rank and DFlash2's
    ring (#128, #129, #131, #132, #134), visible-pool selection (#140), and the CUDA startup reserve setting (#133).
    Thanks to @MiaAI-Lab and @mikolaj92.
  - Flash Next: scaled n-gram tables (#148, reported in #142) and the M5 draft head fix (#147), consolidated EXL3
    tables (#145), block-scaled FP8 (#126) and the NVFP4 expert speedups (#102, #105). Thanks to @gilby,
    @cwschroeder, @shantanugoel, @jschmied and @tournierjc.
  - On M5 Macs the 27B fuses the projections of 4-bit group-32 checkpoints such as oQ4e too (#164): four streams went
    from 315-320 to 332-334 tok/s on an M5 Ultra, with the same bits. Thanks to @gilby.
  - `--checkpoint-slots` (#125), and the DFlash2 drafter's admitted size (#112, from @jkuepker's ROCm work). Thanks
    to @nood-co1 and @jkuepker.

## 0.5.0 (29 Sep 2026)

- **OpenAI's Responses API on both servers.** `/v1/responses` runs as a chat completion: every response has its chat
  completion's token SHA and prompt tokens, and a tool round trip gives the same reply through `previous_response_id`,
  the turn resent whole, or chat. A second turn resumes from the cache.
- **Long context on Sparks, the same bits as 0.3.6.3.** The 27B's attention reads each key chunk once for a round's
  rows, with 16-byte loads, and prompt attention runs in its own CUDA kernel. On one Spark (MLX 4-bit + DFlash2):
  decode at 128k 18.4 -> 38.9 tok/s, at 96k 23.6 -> 45.9; cold prefill at 128k 860 -> 1,173 tok/s and at 255k
  545 -> 809. On the same NVFP4 weights, prefill runs 1.16-1.27x vLLM at every depth from 32k to 255k.
- **The CUDA server reads requests as the Mac server does.** `reasoning_effort` and `thinking_budget`, usage with the
  cached prompt tokens in every reply (streamed ones too), typed tool arguments, `min_p` on both backends,
  `ignore_eos` on Flash Next and Nemotron, and `top_k` 0 drawing the whole nucleus on two GPUs as on one. Thinking
  uses the template's own default effort on both servers.
- **`priority: background` on CUDA.** A background request yields to foreground ones and replays from its prompt,
  so its reply still equals its solo run. Under `--parallel`, the 27B and Qwen3.6 prefill a background prompt 1,024
  rows a step (measured waits in docs/api.md).
- **Community pull requests.**
  - Faster CUDA startup and model loading (#82). Thanks to @pmeenan.
  - Qwen3.6-35B-A3B serves `--parallel N` on CUDA, each reply equal to its solo run (#84), and `response_format`
    JSON schemas on the 27B, exact under drafts (#80). Thanks to @philip-pentatonic.
  - `/health` publishes live token totals (#79). Thanks to @MiaAI-Lab.
  - The 27B's concurrent drafter keeps no stale context after a prefill step (#92). Thanks to @nood-co1.
  - Flash Next lists its sparse-attention blocks past 131,072 keys in tiles, the same lists (#93). Thanks to
    @MovieMaker93.

## 0.4.0 (29 Sep 2026)

- **oQ formats and MLX 8-bit at 4-bit speed on M1-M4.** 5-, 6- and 8-bit linears now run on the matrix units with the
  same exactness. Qwen3.8-27B at oQ4e has half the KL against bf16 of MLX's 4-bit for 6% more bytes. On an M3 Ultra
  it now verifies 8-row windows in 31.9 ms instead of 40.8, within 1-5% of the 4-bit at every width. Served with
  DFlash2 drafts, sampled code runs 14% faster, chat 5-7%, and greedy code is level. The 5- and 6-bit layers use the
  matrix chain's arithmetic now, so replies can differ from 0.3.x's by a token here and there, and still equal their
  own `"draft": false` replies.
- **More streams on Macs.** A stream holds memory for its next 2,048 tokens, not its whole reply. A shared round's
  working memory is charged to the streams that share it. On a 64 GB Mac the 27B serves 16 streams at 32k contexts,
  where 0.3.x served 4-9. When memory runs short, kept prompts go first, then the newest streams wait, then the
  newest ends with an error that names `--parallel`.
- **Prompts inside the promised window are admitted and kept (#95).**
  - A finished request's rollback rows are released before the next prompt is sized.
  - The startup line counts the probe round once: "0 streams of 8,192 tokens fit" became 13 with the reporter's
    flags.
  - The same flags give the same window on every start.

  Thanks to @benwilson.
- **Bonsai on Macs short of memory for the full widening** now widens as many layers as fit (60 of 64 at 25.2 GiB).
  Code runs 55 -> 78 tok/s on an M3 Ultra at the #94 reporter's budget. Thanks to @MESevenJourney.
- **Flash Next on M1-M4** runs 2-7% faster, drafted and serial, on an M3 Ultra.
  - The attention gate runs in the merge kernel, the PLE layer and router are fused, and chained drafts queue as
    they're built.
  - From two rows, the 4-bit dots skip the int-to-float convert (on M3 and M4 through half-precision nibbles), so
    windows of 3-16 rows cost 3-4% less with the same bits.
- **Flash Next's 2-8-bit and mixed checkpoints on Macs.** oQ4, oQ4e and oQ5e, and MLX 6- and 8-bit checkpoints,
  load and serve, each module in its checkpoint's own format. Before, a mixed checkpoint with a 4-bit base passed
  `tensorfold info` and then failed to load. 4-bit weights in groups of 32 keep their kernels. Other widths run new
  kernels whose rows keep their bits at any row count, so drafted replies still equal `"draft": false`.
- **GLM-5.3 oQ4 loads as downloaded.** Its config lists 46 `mlp_layer_types` for 45 layers (the MTP layer too), and
  transformers refused it, so the tokenizer didn't load. TensorFold now reads it as is.
- **A live line under `tensorfold serve` on Macs.** In a terminal, one line under the log shows open connections and
  decode and prefill tok/s, redrawn in place. Log files see nothing new, and `TENSORFOLD_NO_LIVE=1` turns it off.
- **MLX 0.32.3 (#88).**
  - TensorFold runs on MLX 0.32.2 and 0.32.3, and a fresh install gets 0.32.3. Output is the same bit for bit on
    both.
  - SSD expert streaming builds its extension with the nanobind each MLX was built with (the `ssd` extra brings it).
  - A prompt kernel that doesn't build prints a warning naming the MLX version and the fix, and
    `TF_REQUIRE_KERNELS=1` stops at startup instead.
- **DeepSeek-V4-Flash draft heads on Hugging Face.** `Vontra/DeepSeek-V4-Flash-DSpark-MLX` (the default once pulled)
  and `Vontra/DeepSeek-V4-Flash-MTP-MLX`, converted from DeepSeek's MIT releases. A head is a folder holding
  `model.safetensors` and a `config.json` that names it.
- **Fixes.**
  - Object and array tool arguments that the model leaves one closing bracket short are closed, and kept if they
    then match the schema (#87).
  - Qwen3.5 and Qwen3.6 chat templates get the same resume points as Qwen3.8, so a second turn reuses the first
    (#83). Thanks to @philip-pentatonic.
  - A prompt or image that can't fit even alone no longer evicts kept prompts first.
- **Correction to 0.3.6.3's notes.** Nemotron's chat decode on the M3 Ultra wasn't 2% slower. It spreads about
  ±2.5% between server starts, even at a fixed draft depth.

## 0.3.7 (29 Sep 2026)

- **Nemotron on Macs serves concurrent requests again.** 0.3.6.3 kept one draft slot for all of a server's streams,
  so several Nemotron requests at once could fail with HTTP 500 (5 of 6 in our test on an M3 Ultra). Each stream now
  keeps its own: six requests at once all finish, and each reply equals the same request sent alone.
- **Qwen3.8-27B on M1-M4 serves 9, 17 or 25 streams at once.** A round whose last group of recurrent streams held
  one stream built a kernel Metal refused, and every stream in that round failed. That group now builds like the
  others; rounds of 2-8 streams run the same kernels as before.
- Versions have three parts from here on.

## 0.3.6.3 (29 Sep 2026)

- **Images in, on Macs and Sparks.** `--vision` lets Qwen3.8-27B read up to four images a request, as data URLs or,
  with `--vision-urls`, HTTPS links. Image replies keep drafting and equal `"draft": false` ones (77.6 against 37.3
  tok/s serial on an M3 Ultra), and the first token comes as fast as mlx-vlm's. Thanks to @di37 for HTTPS-only
  fetching, bounded image preparation and redacted request logs (#64).
- **DeepSeek-V4-Flash on a 256 GB Mac.** The new `deepseek_v4` family serves `mlx-community/DeepSeek-V4-Flash-4bit`
  on the lane engine with DSpark or MTP drafts converted from DeepSeek's releases, and replies equal `"draft": false`
  ones. On an M3 Ultra it decodes 1.7-3.1x and reads prompts 1.8-2.0x as fast as mlx-lm PR #1797's server. Thanks to
  @jeffpeng3 (#14).
- **NVFP4 checkpoints on CUDA, as published.** Flash Next's NVFP4 exports (local-inference-lab's and RadixArk's)
  and NVIDIA's Qwen3.8-27B NVFP4 load through `tensorfold serve` and stay exact: drafted replies equal `"draft": false`
  ones, resumed prompts equal fresh ones, and 84 of 84 concurrent streams equal their solo runs. The experts read
  their routing on the GPU, so CUDA graphs replay any routing. On one Spark, Flash Next NVFP4 decodes 1.13-1.52x vLLM
  on the same checkpoint. Two ranks, `--ple-on-ssd` and images on NVFP4 checkpoints stop at startup with a message
  until they're qualified. Thanks to @tournierjc for the reader (#67).
- **Ternary Bonsai 2 27B** (prism-ml's 2-bit pack) serves on the lane engine with Qwen3.8-27B's DFlash2 drafts,
  exact. Against mlx_lm running the pack's own runtime on an M5 Max: decode 3.7-5.8x on code and 1.7-2.2x on chat,
  prompts 1.4-1.7x. Thanks to @gprot42 (#18).
- **CUDA server work from @nood-co1:** stop strings on every CUDA engine and `ignore_eos` on the 27B (#63);
  malformed requests get a 400, and failed ones a 500 or a stream error event (#61); a silent start explains itself
  with extension-build progress and stale-lock notes, and `kill -USR1 <pid>` prints every thread's stack, at start or
  while serving (#62); the 27B keeps its prompt cache entry one token before the prompt's end, so the next chat turn
  resumes from it (#65).
- **Replies keep decoding while long prompts prefill on Macs.** A prompt now fills one planned chunk at a time and
  running replies take rounds between its chunks, for `--decode-share` of each chunk's time (default 0.25, about a
  fifth of the time; 0 prefills whole prompts first, as before); every reply still equals its solo run. On an M3
  Ultra, a reply's longest pause while three 17K-token prompts arrive fell from 164 s to 7 s, and a request alone is
  unchanged. Thanks to @benwilson (#72).
- **Long conversations stay warm on Macs.** A checkpoint that can't fit no longer evicts the others first, a resumed
  turn no longer holds the stored prefix it copied, and the 27B's DFlash2 prompt taps take 64 KB a token instead of
  114. On a 64 GB budget (emulated on an M3 Ultra), one conversation grew to 143K tokens and each turn resumed in
  38-46 s, where 0.3.6.2 prefilled every turn past about 100K from the start. Startup names the longest request
  whose prompt is kept. Thanks to @sanjaibalajee (#74), and to @benwilson for the report and his
  64 GB measurements, now in the README labelled 0.3.5.1 (#71, #70).
- **Qwen3.6 on CUDA keeps serving past 8,192 rows** with expandable allocator segments: its graphs now capture into a
  new pool after the buffers grow, where every later request failed. Thanks to @philip-pentatonic (#78).
- **Typed tool arguments on CUDA:** a Qwen tool call's array, object, number and boolean parameters arrive as JSON
  values, as on the Mac. Thanks to @MiaAI-Lab (#75).
- **Flash Next reads a prompt chunk's n-gram rows on 16 threads,** same bytes, so prompt times vary less when the
  tables are not all in the page cache. Thanks to @MovieMaker93 (#73).
- **On M1-M4, the MoE prompt matmuls give each GPU tile one expert's rows,** the scheduling idea of MLX's gather_mm
  change (ml-explore/mlx#4567), written for 4-bit gathers. Same bits. On an M3 Ultra, Flash Next's prompts run 2-4%
  faster (1.02-1.20x oMLX's) and GLM-5.3's 3.4-3.8% faster (1.08-1.11x mlx-vlm's at 8k-32k).
- **Flash Next prompts on M1-M4:** the block scores run as one batched matmul, the GQA kernel scores two query heads
  a simdgroup and the top-512 selection finds its cuts with a simdgroup scan. Same bits, about 1% faster at 16k-64k;
  on an M3 Ultra it is level with oMLX at 32k and 64k.
- **MLX stays at 0.32.2 for now.** MLX 0.32.3 changed a kernel our prompt kernels build on, so fresh installs fell
  back to slower prompts with one log line; the next release follows the new kernel. Thanks to @ecohash-co (#88).

## 0.3.6.2 (28 Sep 2026)

- **EXL3 replies stop at the end of the turn.** Flash Next and 27B EXL3 packs list `<|im_end|>` only in
  `generation_config.json`, so replies ran past their turn and leaked tool calls and think tags. The CUDA engine now
  reads that file too. Thanks to @vcruz305 (#69).
- **pip installs serve 27B EXL3 packs.** The package was missing the 27B's CUDA sources; a test now checks that every
  CUDA source ships. Thanks to @taussoe (#66).
- **A quantized KV cache for Flash Next on CUDA.** `--kv-dtype int8` or `int4` holds about 1.7x or 2.6x the default
  window in the same memory, and drafted replies still equal serial ones. `--mtp-confidence` sets where MTP chains
  stop. Thanks to @vcruz305 (#47).
- **Flash Next on CUDA reaches the first token sooner.** The head runs on a prompt's final chunk only, and the prompt
  kernels load at startup, so the first 2k prompt takes 1.26 s instead of 1.79 s. Thanks to @MovieMaker93 (#40).
- **GLM on two Sparks holds 256k tokens** with a latent attention cache; its next-token loss is within 0.001 nats of
  the per-head cache. Thanks to @taussoe (#54).
- **Mixed-bit Qwen checkpoints** (4-bit with some 5- and 6-bit layers, such as oQ4) load on every lane backend,
  exact, with new row kernels for 2- to 8-bit weights. On an M3 Ultra, oQ4 27B decodes 114-120 tok/s on code and
  59-61 on chat, against mlx_lm's 34-36.
- **Concurrent 27B on CUDA:** `--parallel 16` serves 161.7 tok/s on one Spark in 25.4 GiB, each reply equal to its
  solo run (#38). **Qwen3.6-35B-A3B on CUDA,** exact: decode 1.36-1.49x vLLM with MTP, prompts 1.21-1.37x (#45).
- **Gemma 4 drafts (opt-in):** `--drafter z-lab/gemma-4-26B-A4B-it-DFlash` decodes 1.3-2.1x mlx_lm on an M3 Ultra,
  exact.
- **Tool calls:** `tool_choice: "required"` and a named tool are enforced on both servers (#52), and a complete tool
  call inside an unclosed think block comes back as a tool call (#60).
- **Conversations come back warm on Macs.** `--spill-gib N` writes a conversation pushed out of the prompt cache
  to disk, up to N GiB, and reads it back when the conversation returns. On a 48 GB budget a 35k-token
  conversation came back in 0.27 s on an M5 Max instead of 75 s, with the same reply. Off by default;
  `--checkpoint-slots` sets how many conversations stay in memory. Thanks to @gilby (#68, #55).
- **Memory:** `TENSORFOLD_MEMORY_LIMIT_GB` raises the budget above the default share, and Flash Next's memory check
  counts its host-mapped n-gram tables, so it starts on a 128 GB Mac. Thanks to @Chedrian07 (#49, #50).
- **CUDA server fixes from @nood-co1:** an abandoned request stops within a round (#57), keys and values stay within
  the admitted window (#58), and a failed admission no longer stops the scheduler (#59).
- **M1-M4:** a prompt split into parts attends exactly as it does in one piece at every length; two 8,192-key cases
  rounded differently in 0.3.6.

## 0.3.6.1 (28 Sep 2026)

- **CUDA builds inside NVIDIA's containers again.** Their `TORCH_CUDA_ARCH_LIST` names every architecture back to
  sm_80, so the kernels' thread-block clusters and FP8 MMA failed to compile for GPUs that lack them. Every extension
  now builds for the GPU that is present, and a GPU older than compute capability 9.0 gets a clear message. Thanks to
  @ss-cong for the report and the exact errors (#56).

## 0.3.6 (28 Sep 2026)

- **GLM-5.3-Flash on Macs** with 256 GB, drafted replies equal to serial ones. Prompts process at or above mlx-vlm
  from 2k to 32k tokens on an M3 Ultra, and tool calls parse in both servers. Thanks to @chadhurley25075-png (#9,
  #39) and @jeidbugs404 (#35).
- **Gemma 4 26B-A4B on the lanes,** exact at every width, with prompts at or above mlx_lm from 2k to 64k tokens on
  an M3 Ultra. Thanks to @cshintov (#10).
- **Bigger models on smaller Macs.** `--ple-on-ssd` reads Flash Next's n-gram tables from disk, so a 128 GB Mac holds
  it (#16). `--ssd-experts GIB` streams routed experts from the checkpoint into a GPU pool of that size, so Flash Next
  fits a 64 GB Mac and GLM a 128 GB one. Replies are the resident model's tokens; decode runs at 0.31-0.39x resident
  speed for Flash Next and 0.13-0.17x for GLM (measured on an M3 Ultra; `pip install "tensorfold[ssd]"` first) (#17).
- **EXL3 checkpoints on CUDA (experimental):** Qwen3.8-27B and Flash Next packs from turboderp, exact on the lanes.
  Decode runs 1.6-3.6x vLLM with MTP; prompt processing is about half the MLX checkpoints' speed for now, and the
  fix is next. Thanks to @vcruz305 (#42).
- **Faster prompts.** Flash Next sizes its prompt chunks to the memory it has, Nemotron takes up to 8,192 tokens a
  chunk on M5 GPUs, and the weights stay wired while a server runs.
- **Fixes:** GLM on two Sparks answered "!" past about 2,000 prompt tokens, and EXL3 GLM prompts past 128 tokens
  failed (#53). A reply that isn't a tool call comes back as content, not an HTTP 500 (#51). The server hands MLX's
  freed buffers back when it goes idle (@kingjamez, #44).
- **MLX 0.32.2 or newer** is required on Macs.
- **What's new after an update:** `tensorfold update` prints these notes when it finishes.
- **Known:** replies to prompts longer than one prompt chunk can differ between machines with different memory,
  because the chunk size follows the memory budget. Within one server, drafted replies always equal serial ones and
  resumed prompts equal fresh ones.

## 0.3.5.1 (28 Sep 2026)

- Qwen3.8-27B loads on M1 and M2 Macs again. Kernels there fit Metal's per-kernel thread limit, with the same sums in
  the same order, so drafted output still equals serial output.
- M3, M4 and M5 run 0.3.5's machine code unchanged.
- Thanks to @hichaiuse, @simonmd, @gcarusso, @tonydehnke, @Cyb3r-Monk and @tinyapps for the reports and the repro.

## 0.3.5 (27 Sep 2026)

- **Concurrent requests share each verification round.** `--parallel auto` is on by default, and every stream's reply
  equals the same request served alone, on Metal and on CUDA.
- **Follow-up turns resume at the start of their newest messages,** with output identical to a fresh prompt. A 12-turn
  agent session with the 27B spent 14.9 s on first tokens instead of 31.9 s.
- **Memory that fits.** The whole process stays inside 70% of RAM, an omitted `--context` defaults to the window the
  machine can hold, and a prompt past it gets a clear 400.
- **Flash Next prefill** with sparse prompt attention, 1.1-1.4x faster than 0.3.4.1 on an M3 Ultra (@quigles1977, #29).
- **2- to 8-bit weights** on the lanes, so mixed-precision 27B checkpoints decode fully (@jasontitus, #34).
- **CUDA:** FP8 prefill and shared expert kernels, concurrent streams for the 27B and Flash Next, and admission from
  available memory before loading.
- **API:** raw `/v1/completions` prompts, `ignore_eos` and `stop`, request `reasoning_effort` and typed tool arguments
  (@chris247474, #28), `developer` messages, `parallel_tool_calls: false`, and cancellation when a client disconnects.

## 0.3.4.1 (27 Sep 2026)

- Prompt processing is back to MLX's speed on every model. Prompts prefill through MLX's own forward on a fixed
  2,048-token grid, so a resumed conversation still equals a fresh one byte for byte.
- Flash Next's peak memory stays within 20 GB of its weights up to a 196k-token prompt.

## 0.3.4 (26 Sep 2026, pre-release)

- Every model runs on the lane engine, and the serial engine is gone. Nemotron drafts on M1 to M4 with row-exact
  kernels.
- Qwen3.8-27B on M1 to M4 decodes at 1.9 to 4x serial, through a new 4-bit matmul on the simdgroup matrix units.
- CUDA: every default path verifies at least two rows a round.

## 0.3.3 (26 Sep 2026)

- Qwen3.8-27B verifies drafted tokens together on every M1 to M5 GPU, with output byte-identical to serial decoding.
- Streamed `/v1/completions` send plain text.

## 0.3.2 (26 Sep 2026)

- `tensorfold update` installs the newest release.
- GLM-5.3-Flash reads Brandon M. Music's EXL3/TR3 weights (re-hosted by Mia-AiLab) on two DGX Sparks (experimental).

## 0.3.1 (26 Sep 2026)

- `tensorfold info` shows how a checkpoint stores its weights and which backends read them. `serve` and `pull` refuse
  checkpoints no engine reads yet, before anything downloads.

## 0.3.0 (26 Sep 2026)

- NVIDIA GPUs: `tensorfold serve` picks CUDA on Linux and runs on one GPU or two (one rank per DGX Spark).
- A new family, GLM-5.3-Flash, on two Sparks.

## 0.2.0 (25 Sep 2026)

- A rewrite: `tensorfold serve`, `pull`, `models` and `info` for Nemotron 3.5 Lightning, Qwen3.8-27B and Qwen3.8
  Flash Next on Apple Silicon, with drafts that never change the output.
