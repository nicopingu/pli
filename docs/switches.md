# PLI settings (`PLI_*`)

Every PLI addition to TensorFold 0.6.6 is switched by a `PLI_*` environment variable, off by default in the code except where the
"Code default" column says otherwise.
PLI 4.1.0 reads 162 of its own, listed here in four parts and "Other", plus the inherited TensorFold settings (renamed, see the end). `--profile NAME` (or `PLI_PROFILE=NAME`) loads a machine profile
from `tensorfold/pli/profiles/NAME/` before any kernel is imported; a variable already set in the environment always wins.

Terms used below:

- **same bits**: the new path gives bit-identical results to the path it replaces, so the text does not change;
- **text can change**: the arithmetic differs, so a reply can differ from the one the replaced path gives; each such change
  was judged on whole conversations before it went into production (see [CHANGELOG.md](CHANGELOG.md)).

The "Code" column names the file where the variable is first read.

## 1. Stable: in production, set by the `m1-ultra` profile

These are the production settings of the validated machine (Mac Studio M1 Ultra, 64 GB). Most apply to the Qwen3.8 dense
family only (`qwen3_5.env`); `common.env` applies to every model. Three are on by default in the code and need no profile line
(`PLI_SPILL_PROMPT`, `PLI_FORK_MIN`, `PLI_CONTROL_AS_TEXT`).

| Variable | Profile value (file) | Code default | What it does | Code |
| --- | --- | --- | --- | --- |
| `PLI_ADMIT_SMALL` | `8000` (qwen3_5.env) | `0` | A request of at most N tokens is admitted ahead of a request held back at admission (small requests first). | `tensorfold/server/scheduler.py` |
| `PLI_ANE_BLOCKS` | `2` (qwen3_5.env) | `1` | Neural Engine share split into two programs pinned on two Neural Engine blocks (Ultra chips only). | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_FFN` | `1` (qwen3_5.env) | `0` | Part of each dense FFN layer runs on the Neural Engine while the GPU reads the prompt. | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_FFN_UNITS` | `18` (qwen3_5.env) | `12` | Units of 512 FFN channels given to the Neural Engine (used when `PLI_ANE_STEPS` is empty). | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_KEEPAWAKE` | `1` (qwen3_5.env) | `0` | Keeps the Neural Engine awake between two shared layers with tiny calls (it slows down after about 2 ms idle). | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_STEPS` | `256:14,512:16,1536:18` (qwen3_5.env) | `empty` | Neural Engine units by chunk size, as `rows:units,...`; one program set per step, each for chunks of at least that many rows. | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_REQUANT_CPU` | `1` (qwen3_5.env) | `0` | Neural Engine weights requantized on the CPU (NEON) during the previous layer instead of on the GPU; same bits. | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_REQUANT_THREADS` | `8` (qwen3_5.env) | `8` | CPU threads used for that requantization. | `tensorfold/pli/ane/requant_cpu.py` |
| `PLI_ARBITER` | `1` (common.env) | `0` | Cost arbiter: learns the real cost of a round online and picks the copy width and the number of drafts; never changes tokens. | `tensorfold/pli/arbiter.py` |
| `PLI_ATTENTION_ATTEC` | `1` (qwen3_5.env) | `0` | Decode attention for 2 to 16 rows (1,100 keys and more) in one launch at equal bits; on the 27B only a fallback since `attm`. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ATTENTION_ATTM` | `1` (qwen3_5.env) | `0` | PLI's own decode attention (`attm`, 8 x 8 simdgroup tiles) for every decode row of the 27B, 1 to 32 rows. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ATTENTION_F16` | `1` (qwen3_5.env) | `0` | Prompt-reading attention of the 27B with its products in f16 (bits change). | `tensorfold/kernels/qwen/dense/v1/prompt_attention.py` |
| `PLI_ATTENTION_GEMM` | `1` (qwen3_5.env) | `0` | Prompt-reading attention products through MLX's `steel_gemm_fused` kernel with tiles measured on M1 Ultra; same bits as MLX's unfused path. | `tensorfold/kernels/qwen/dense/v1/prompt_attention.py` |
| `PLI_ATTENTION_PER_ROW` | `15000` (qwen3_5.env)<br>`15000` (qwen3_5_moe.env) | `0` | From N context keys, one MLX attention call per row instead of grouped calls; same bits. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ATTENTION_REDUCED` | `10` (qwen3_5.env) | `0` | Reduced decode attention beyond `PLI_ATTENTION_REDUCED_THRESHOLD` keys: reads the N% of 32-key pages with the highest score plus a 1,024-token tail (text can change). | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ATTM_SHORT` | `12288` (qwen3_5.env) | `0` | `attm` works in 128-key slices for calls of at most N keys (512-key slices above), so short contexts keep the GPU busy. | `tensorfold/pli/attention_attm.py` |
| `PLI_HEARTBEAT_MS` | `500` (qwen3_5.env) | `0` | Heartbeat between requests: a tiny GPU job every N ms keeps the wired weights resident (Metal releases them after about 2 s idle). 0 = off. | `tensorfold/server/scheduler.py` |
| `PLI_INTERLEAVE_DENSE` | `1` (qwen3_5.env) | `0` | Allows the crossed split of prompt reading on the dense model (see `PLI_CHUNK_LAYERS`). | `tensorfold/families/qwen3_5/family.py` |
| `PLI_MERGE_TAIL` | `16` (qwen3_5.env) | `0` | When the last checkpoint is at most N tokens before the end of the prompt, those tokens are read in the same pass as the chunk before (text can change). | `tensorfold/pli/merge_tail.py` |
| `PLI_HOST_GUARD` | `1` (common.env) | `unset (off)` | Host/Origin guard: 403 for a non-local Host on a local address, or an Origin whose host is not admitted. | `tensorfold/server/host_guard.py` |
| `PLI_KEEP_REPLY` | `1` (common.env) | `0` | Keeps the reply for the next agent turn: the chat template keeps the empty think block, the engine keeps the decoded state (text can change). | `tensorfold/server/prompt_fill.py` |
| `PLI_LOG_TIMES` | `1` (common.env) | `0` | Arrival and end time added at the end of each request's `done` log line. | `tensorfold/server/app.py` |
| `PLI_PREFILL_UNPACK` | `f16` (qwen3_5.env) | `0` | 4-bit weights unpacked before the product in large prompt-reading calls (`f16` for projections fed by the normed state, bf16 for the others). | `tensorfold/pli/unpack.py` |
| `PLI_PREFILL_UNPACK_MLP` | `t` (qwen3_5.env) | `0` | With `f16` unpacking, the dense MLP computed transposed; same bits as the f16 path. | `tensorfold/pli/unpack.py` |
| `PLI_PREFILL_UNPACK_NN` | `1` (qwen3_5.env) | `0` | Unpacked bf16 projections (`out_proj`, `o_proj`) and the draft's `fc`, `k_proj`, `v_proj` computed in nn form; same bits. | `tensorfold/pli/unpack.py` |
| `PLI_PREFILL_UNPACK_KERNEL` | `1` (qwen3_5.env) | `0` | One-pass Metal kernel instead of `mx.dequantize` for the unpacking; same values. | `tensorfold/pli/unpack.py` |
| `PLI_CHUNK_LAYERS` | `4` (common.env) | `0` | While another request decodes, the reading of a prompt chunk yields every N layers (crossed split); same operations, same order. | `tensorfold/engine/family_prefill.py` |
| `PLI_MTP_DRAFTS` | `7` (qwen3_5.env) | `4` | Most MTP drafts per round when the 27B drafts with its MTP head (`PLI_MTP_HEAD`). | `tensorfold/families/qwen3_5/__init__.py` |
| `PLI_MTP_WINDOW` | `2048` (qwen3_5.env) | `0` | The MTP head keeps only its last N rows (sliding window), which bounds its memory. | `tensorfold/families/qwen3_5_moe/mtp.py` |
| `PLI_TILES_PER_GROUP` | `1` (qwen3_5.env) | `2` | 8-row tiles per threadgroup in the exact row kernel (1 instead of TensorFold's 2); same bits. | `tensorfold/kernels/qwen/dense/v1/simd_qmm.py` |
| `PLI_RELAYOUT` | `1` (qwen3_5.env) | `0` | 27B weights and scales re-laid out for the decode kernel; each matrix is checked at load (same bits required, else it keeps its layout). | `tensorfold/pli/relayout.py` |
| `PLI_RECALL` | `1` (qwen3_5.env) | `unset (off)` | Recall drafts: continuations copied from the text of finished requests kept in memory; the model verifies them, same text. | `tensorfold/pli/recall.py` |
| `PLI_REDUCED_ANCHOR` | `32` (qwen3_5.env) | `0` | Reduced attention anchored to positions: a row reads the pages chosen at its anchor `max(N * (q // N), request start)`; text no longer depends on drafts. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_REDUCED_ROUNDS` | `4` (qwen3_5.env) | `1` | A layer keeps its reduced-attention page choice for K rounds; keys added since are read in full. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ROUNDS_BETWEEN_PREFILLS` | `100000` (common.env) | `32` | Most decode rounds between two prompt chunks (TensorFold: 32). | `tensorfold/server/scheduler.py` |
| `PLI_XF_BF16` | `1` (qwen3_5.env) | `0` | Input fragments of the 2 to 8-row kernel stored as bf16 pairs instead of fp32 pairs; same bits, half the bytes read. | `tensorfold/kernels/qwen/dense/v1/simd_qmm.py` |
| `PLI_V8` | `1` (qwen3_5.env) | `0` | Kapitsa's 8-row verify GEMM for 4-bit, group-64 weights, for every call of at most 128 rows; row-exact (text can change against `simd_qmm`). | `tensorfold/kernels/qwen/dense/v1/qmm8.py` |
| `PLI_KV_SHARING` | `1` (qwen3_5.env) | `0` | Checkpoints keep the conversation's attention layers and a length instead of a copy of the KV cache (same text). | `tensorfold/pli/kv_sharing.py` |
| `PLI_HYBRID_MESSAGES` | `3` (qwen3_5.env) | `0` | With `PLI_HYBRID`, a prompt of at most N messages (counted by `PLI_HYBRID_IM_START`, 248045) drafts with DFlash2 alone. | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_KFLASH` | `1` (qwen3_5.env) | `0` | Draft trees in KFlash's manner (method from Kapitsa): best-first tree over the draft's probabilities, calibrated per depth and learned online. Needs `PLI_TREES_ATTM=1` (and `PLI_HYBRID_TREES=1` with the hybrid). The verify stays exact; the tree only decides how many tokens a round keeps. | `tensorfold/pli/kflash.py` |
| `PLI_TREES_ATTM` | `1` (qwen3_5.env) | `0` | Lets the draft propose trees, verified through `attm` with an ancestor mask; chains keep their path and their bits. `PLI_TREE_EXACT` (off) gives every node the bits of the same row in a chain. | `tensorfold/kernels/qwen/dense/v1/row_forward.py` |
| `PLI_HYBRID_TREES` | `1` (qwen3_5.env) | `0` | Lets DFlash2 propose trees inside the hybrid (needs `PLI_TREES_ATTM` or `PLI_ROW_ATTENTION`). | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_TREE_DELTANET` | `3` (qwen3_5.env) | `0` | DeltaNet recurrence kernel for draft trees of at most 8 nodes. `1`: the nodes' states in threadgroup memory, same arithmetic in the same order as the original tree path (same bits); `3`: the same, but only the nodes with a child further than the next node keep their state, in at most 4 slots (less threadgroup memory). | `tensorfold/kernels/qwen/dense/v1/row_glue.py` |
| `PLI_TREES_SHORT_ONLY` | `1` (qwen3_5.env) | `0` | With `PLI_HYBRID_TREES`, DFlash2's trees only for short questions (at most `PLI_HYBRID_MESSAGES` messages); agent sessions keep the chain. | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_TREES_TOKEN_THRESHOLD` | `4.5` (qwen3_5.env) | `0` (off) | On a short question, the chain stays while the moving average of the tokens kept per round (about ten rounds) reaches N; below N, the tree. | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_HEAD_BEFORE_COPY` | `3` (qwen3_5.env) | `0` | Copy of the context or DFlash2's queued chain: `1`: never a copy when the head has a chain queued; `2`: the copy only if its first two tokens are those of the chain; `3`: learned per stream, both scored afterwards on the written text, the copy goes first while the moving average of copy minus head stays above `-PLI_HEAD_COPY_THRESHOLD`. With the MTP head, the former behaviour. The model verifies every draft. | `tensorfold/engine/lane_family.py` |
| `PLI_EXTENDED_DRAFT_LIST` | `extended-list-2048.txt` (qwen3_5.env) | empty (off) | File of token ids (one per line) added once, at load, to the reduced heads of DFlash2 and of the MTP head, so the drafts can propose them; a bare file name is looked up next to the package, which ships `extended-list-2048.txt`. No extra product per round; the verify stays exact (same text). | `tensorfold/pli/extended_list.py` |
| `PLI_DRAFT_ATTN` | `1` (qwen3_5.env) | `0` | Attention of the DFlash2 draft in a PLI kernel (sliding window, non-causal block of 1 to 8 rows, two passes), reading the cache and the block without concatenating them. Bits other than MLX's; the model verifies the drafts. | `tensorfold/pli/attention_draft.py` |
| `PLI_DRAFT_STACK` | `1` (qwen3_5.env) | `0` | DFlash2 draft projections stacked: [q \| k \| v] of the block and [k \| v] of the context in one 8-row product call each, `fc` relaid out, the draft head in two calls instead of three (2 to 16 rows); same bits, checked at install. | `tensorfold/pli/draft_stack.py` |
| `PLI_DRAFT_QKR` | `1` (qwen3_5.env) | `0` | With `PLI_DRAFT_STACK`, the draft's q and k norms, RoPE and reordering (heads, rows) in one kernel per layer instead of about eight MLX operations; same computation as MLX 0.32.3, same bits. | `tensorfold/pli/draft_qkr.py` |
| `PLI_DRAFT_TOPK` | `1` (qwen3_5.env) | `0` | The k best candidates of each row of the draft's lattice in two passes (16 slices per row, then one simdgroup per row); same indices and values, in the same order, as the original kernel. | `tensorfold/pli/topk_fast.py` |
| `PLI_RELAYOUT_DRAFT` | `1` (qwen3_5.env) | `0` | Also relays out the draft's projections that read only the 8 rows of the block (`q_proj`, `o_proj`, gate, up, down, `kernel_projection`); each matrix checked at the same bits. Same drafts, same text. | `tensorfold/pli/relayout.py` |
| `PLI_ATTM_LOAD` | `1` (qwen3_5.env) | `0` | `attm` lays out the block's keys and values without permutation (stride of 264 half-words) and loads the 8 x 8 fragments with `simdgroup_load`; same products in the same order: same bits. | `tensorfold/pli/attention_attm.py` |
| `PLI_V8_PREP2` | `1` (qwen3_5.env) | `0` | The 8-row tables of `PLI_V8` prepared over ceil(G / 8) threadgroups per row instead of one; same computation: same bits. | `tensorfold/kernels/qwen/dense/v1/qmm8.py` |
| `PLI_REREAD_DISK` | `1` (qwen3_5.env) | `0` | Looks at the disk again at a request's first chunk: a checkpoint evicted (spilled to disk) since the request opened is read back instead of reading the whole prompt again. | `tensorfold/server/prompt_fill.py` |
| `PLI_S3B` | `1` (qwen3_5.env) | `0` | S3b: prompt chunks of at most `PLI_S3B_ROWS` rows past `PLI_S3B_THRESHOLD` positions go through the anchored reduced attention (`PLI_ATTENTION_REDUCED`, `PLI_REDUCED_ANCHOR`); anchors bounded at the start of the request's prefill, full layers read whole (text can change). | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_S3B_THRESHOLD` | `65536` (qwen3_5.env) | `65536` | Positions past which S3b applies (at least 4,096). | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_SPILL_PROMPT` | unset | `1` | An evicted prompt checkpoint goes to disk even when a reply end extends it (a reply end no longer counts as a continuation; a longer prompt still does); `0`: the former behaviour. | `tensorfold/server/checkpoints.py` |
| `PLI_FORK_MIN` | unset | `4096` | When a prompt forks from a kept one inside its last message (the same document sent again with another question), the state at the fork point is kept, past N tokens read since the resume; without a resume, the fork is looked for in the kept prompts. 0: off. Same text (a resume gives what a full prefill gives). | `tensorfold/server/checkpoints.py` |
| `PLI_LAST_MESSAGE_CHECKPOINT` | `1024` (qwen3_5.env) | `0` | Also keeps the state at the start of the prompt's last message (the turn marker that opens the reply, as in ChatML, looked for backwards in the history) when at least N tokens are read there since the resume: the next request that keeps the history but not that message (a question replaced, a reply thrown away, a sub-agent on the same history) starts from there instead of from the previous request's checkpoint. 0: off. Same text (a resume gives what a full prefill gives). | `tensorfold/server/checkpoints.py` |
| `PLI_CHECKPOINT_BITS` | `4` (qwen3_5.env) | `0` | Checkpoints evicted from memory or written to disk are kept in 4 bits (`4`: Hadamard rotation, groups of `PLI_CHECKPOINT_GROUP`) or 8 bits (`8`) and restored to 16 bits on resume; the checkpoint key carries the format, so a file written in another format is not reused (text can change, resumed turns only). | `tensorfold/pli/checkpoint_bits.py` |
| `PLI_A1_PREFILL_ONLY` | `1` (qwen3_5.env) | `0` | With `PLI_A1_LSA_MAP`, A1 reads prompt chunks only; decode and trees keep the attention of production. Without the map, does nothing. | `tensorfold/pli/a1_lsa.py` |
| `PLI_CONTROL_AS_TEXT` | unset | `1` | Control tokens written in the text of a request (messages, tools, tool results) stay ordinary text; `0`: the former behaviour. | `tensorfold/server/text.py` |

## 2. Machine settings: paths and hosts, set in the environment

They name files, directories or hosts of one machine, so no profile sets them. `PLI_HYBRID` is a stable feature, in
production, but it needs the path in `PLI_MTP_HEAD`, so both go in the environment. The two MLX settings are read by MLX itself
when it starts, before any profile, so they go in the environment too (they are not `PLI_*` settings and are not counted
above). `PLI_A1_LSA_MAP` names a file too, and the two S3b values that go with it are set beside it. The FALCON pair is experimental.

| Variable | Example | What it does | Code |
| --- | --- | --- | --- |
| `PLI_HYBRID` | `1` | DFlash2 and the MTP head loaded together; each conversation drafts with the one that gives it more tokens per millisecond. Needs `PLI_MTP_HEAD` and `--drafter`. Stable, in production. | `tensorfold/families/qwen3_5/__init__.py` |
| `PLI_MTP_HEAD` | `~/models/qwen3.8-27b-mtp-4bit.safetensors` | Path of the 27B's native MTP head (`mtp.*` tensors, 4-bit safetensors). Alone, it replaces DFlash2; with `PLI_HYBRID=1`, both are loaded. | `tensorfold/families/qwen3_5/__init__.py` |
| `PLI_A1_LSA_MAP` | `~/pli/models/qwen38_27b.npz` | A1: distilled map of [Latent Sparse Attention](https://github.com/Prefrontal-AI/latent-sparse-attention) (`<layer>.W_Q_idx`, `<layer>.W_K_idx`, release `d-lsa-weights-v1`, `qwen38_27b.pt`, converted by `tools/convert_lsa_map.py`). Past `PLI_S3B_THRESHOLD`, each row of a chunk reads the 10% best scored tiles of 32 keys plus the last 1,024 positions, on the layers the map covers (text can change). Stable with `PLI_A1_PREFILL_ONLY=1`, `PLI_S3B_THRESHOLD=16384` and `PLI_S3B_ROWS=4096`, all three in the environment. | `tensorfold/pli/a1_lsa.py` |
| `PLI_ALLOWED_HOSTS` | `mac.local,192.0.2.10` | Extra host names admitted by the Host/Origin guard (for example the machine's LAN name). | `tensorfold/server/host_guard.py` |
| `PLI_EVENTS` | `~/pli/logs/events.jsonl` | Path of the event journal: one JSON line per scheduler action, with the conversation, no message content. | `tensorfold/pli/events.py` |
| `PLI_TRACE_DIR` | `~/pli/logs/trace` | Directory of the step trace, off until switched on with `python -m tensorfold.pli.trace on [--duration 24h]` (no restart). | `tensorfold/pli/trace.py` |
| `MLX_MAX_MB_PER_BUFFER` | `500` | Most MB of an MLX command buffer. MLX reads it when it starts, before any profile is applied, so it goes in the server's environment; `pli setup` sets it with `--profile m1-ultra`. Stable, in production. | `tensorfold/install/commands.py` (read by MLX) |
| `MLX_MAX_OPS_PER_BUFFER` | `100` | Most operations of an MLX command buffer; in the environment for the same reason. Stable, in production. | `tensorfold/install/commands.py` (read by MLX) |
| `PLI_FALCON` | `~/pli/falcon/capture` | Experimental. Directory where FALCON captures training examples for the MTP head (hidden state, chosen token, top logits). Off when unset. | `tensorfold/pli/falcon/capture.py` |
| `PLI_FALCON_MAX_GB` | `50` | Experimental. Disk cap of the FALCON capture in GB; the oldest files go first. Code default 2. | `tensorfold/pli/falcon/capture.py` |

## 3. Experimental: off by default

Present in the code, not in production, not set by any profile. Some are kept for benches or diagnostics. FALCON
(capture of training examples for the MTP head) is listed here with its settings: it is off unless `PLI_FALCON` is set.

| Variable | Code default | What it does | Code |
| --- | --- | --- | --- |
| `PLI_ADMIT_BY_WORK` | `0` | Admission counts the prompt-reading work measured by `PromptMemory` instead of the `prefill_bytes` line. | `tensorfold/server/scheduler.py` |
| `PLI_ANCHOR_LOG` | `unset` | Logs the reasons of the anchored reduced attention. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ANE_STATS` | `unset` | Writes cumulative Neural Engine time and CPU requantization time to a file every 0.5 s (load survey). | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_REQUANT_AFTER` | `0` | Starts the CPU requantization of the next layer after the Neural Engine call instead of during it. | `tensorfold/pli/ane/ffn.py` |
| `PLI_ARBITER_LOG` | `unset` | Log file of the arbiter. | `tensorfold/pli/arbiter.py` |
| `PLI_ARBITER_MIN_ROWS` | `0` | A request decoding alone keeps at least N - 1 drafts (N rows) when the draft offers them. | `tensorfold/pli/arbiter.py` |
| `PLI_ARBITER_TABLE` | `0` | Draft-count costs taken from a table observed in service instead of the arbiter's formula. | `tensorfold/pli/arbiter.py` |
| `PLI_TREE_EDGE` | `unset (0.6)` | Draft tree setting: weight of the pairwise score. | `tensorfold/drafters/dflash_proposer.py` |
| `PLI_TREE_NOISE` | `unset (0.7)` | Draft tree setting: Gumbel noise weight. | `tensorfold/drafters/dflash_proposer.py` |
| `PLI_TREE_CORRECT` | `0` | Corrects the odds of tree nodes by the observed ratio of accepted to expected tokens. | `tensorfold/engine/family_depth.py` |
| `PLI_TREE_CTX_MAX` | `0` | Draft trees only below N context positions (0 = everywhere). | `tensorfold/families/qwen3_5/dflash_head.py` |
| `PLI_TREE_CHILDREN` | `unset (4)` | Draft tree setting: candidates expanded under each node. | `tensorfold/drafters/dflash_proposer.py` |
| `PLI_TREE_EXACT` | `0` | Tree attention with the chain's bits for every node. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_TREE_NODES` | `0` | Caps the nodes of a draft tree (7 = 8 rows verified per round). | `tensorfold/drafters/dflash_proposer.py` |
| `PLI_TREE_ALONE` | `0` | Trees only for a request decoding alone; with others, each tree is cut to its main branch, run as a chain. | `tensorfold/engine/family_depth.py` |
| `PLI_TREE_TAU` | `unset (1.5)` | Draft tree setting: temperature of the node scores. | `tensorfold/drafters/dflash_proposer.py` |
| `PLI_ATTENTION` | `unset` | `gemm` or `m1`: windows of 1 to 128 rows through a PLI attention where a row has the same bits alone or in a window. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ATTENTION_GROUPED` | `0` | Prompt reading: query heads sharing a key head in one call with an explicit causal mask (more than 16 rows). | `tensorfold/kernels/qwen/dense/v1/prompt_attention.py` |
| `PLI_ATTENTION_TUNING_FILE` | `unset` | Benches only: file that changes the reduced-attention share at run time. | `tensorfold/cli.py` |
| `PLI_ATTM_SMALL_KEYS` | `0` | `attm` stores the first N keys in 128-key slices. | `tensorfold/pli/attention_attm.py` |
| `PLI_ATTN_SLICES` | `512x0` | `PLI_ATTENTION=m1`: key slices as `size x count,...`. | `tensorfold/pli/attention_m1.py` |
| `PLI_ATTN_NS` | `4` | `PLI_ATTENTION=m1`: simdgroups per 8-query tile. | `tensorfold/pli/attention_m1.py` |
| `PLI_ATTN_ROWS` | `16` | `PLI_ATTENTION=gemm`: rows per pass. | `tensorfold/pli/attention_gemm.py` |
| `PLI_ATTN_RT` | `1` | `PLI_ATTENTION=m1`: 8-query tiles per simdgroup. | `tensorfold/pli/attention_m1.py` |
| `PLI_ATTN_TK` | `64` | `PLI_ATTENTION=m1`: keys per softmax step. | `tensorfold/pli/attention_m1.py` |
| `PLI_HEARTBEAT_ANE` | `0` | A Neural Engine keep-awake call at the heartbeat's pace. | `tensorfold/server/scheduler.py` |
| `PLI_F16_CLAMP` | `0` | Clamps Q, K and V to +-65280 before the f16 conversion. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_f16.py` |
| `PLI_FALCON_TARGETS_EVERY` | `2` | FALCON: top-logit targets captured one round in N. | `tensorfold/pli/falcon/capture.py` |
| `PLI_FALCON_CONTEXT8` | `1` | FALCON: prompt rows captured in 8 bits per row (scale = max abs / 127) instead of 16. | `tensorfold/pli/falcon/capture.py` |
| `PLI_FALCON_STREAM` | `1` | FALCON: capture on its own GPU stream, so the next round's drafts do not wait for it. | `tensorfold/pli/falcon/capture.py` |
| `PLI_FALCON_K` | `20` | FALCON: top logits kept per row. | `tensorfold/pli/falcon/capture.py` |
| `PLI_FALCON_BATCH` | `4096` | FALCON: rows per capture file. | `tensorfold/pli/falcon/capture.py` |
| `PLI_FALCON_PROMPT` | `2048` | FALCON: last prompt rows kept per request. | `tensorfold/pli/falcon/capture.py` |
| `PLI_FALCON_HEAD` | `unset` | FALCON: JSON file naming a published MTP head; checked every 30 s, the head is swapped between two rounds. | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_HYBRID_FORCE` | `unset` | `dflash` or `mtp`: freezes the hybrid's choice (benches). | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_HYBRID_LOG` | `unset` | File logging each switch of the hybrid draft. | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_HYBRID_DEPTH` | `0` | In MTP mode, the hybrid picks its depth like the MTP head alone (acceptance by depth, cost of one step). | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_A1_MAX_TILES` | `0` | With `PLI_A1_LSA_MAP`, most tiles of 32 keys a row reads (0 = no cap). Faster at long context (first token at 146k -22 to -27% instead of -15.6%), but the NLL rises (+0.0046 at 160 tiles, 97k). | `tensorfold/pli/a1_lsa.py` |
| `PLI_PREFILL_CPU` | `0` | Share of prompt rows (0 to 0.9) computed on the CPU in 32 bits (text can change). | `tensorfold/pli/relayout.py` |
| `PLI_PREFILL_CPU_MIN` | `256` | Chunk rows below which nothing goes to the CPU. | `tensorfold/families/qwen3_5/prefill_cpu.py` |
| `PLI_PREFILL_CPU_MODE` | `decale` | `decale` or `couche`: how CPU and GPU rows are scheduled. | `tensorfold/families/qwen3_5/prefill_cpu.py` |
| `PLI_PREFILL_CPU_PRIORITY` | `1` | Priority of the CPU work: 0 high, 1 medium, 2 background. | `tensorfold/families/qwen3_5/prefill_cpu.py` |
| `PLI_PREFILL_CPU_TRACE` | `0` | One log line per chunk with the CPU share's times. | `tensorfold/families/qwen3_5/prefill_cpu.py` |
| `PLI_PREFILL_UNPACK_CHECK` | `0` | Counts infinite or NaN f16 outputs (forces an evaluation: tests only). | `tensorfold/pli/unpack.py` |
| `PLI_PREFILL_UNPACK_STACK` | `0` | Stacked projections (`in_proj_qkv` + `in_proj_z`, `q` + `k` + `v`) in one nn product. | `tensorfold/pli/unpack.py` |
| `PLI_MLP_INT` | `0` | MLP path in integers (reads the original weight layout, so `PLI_RELAYOUT` is not applied). | `tensorfold/pli/relayout.py` |
| `PLI_CHUNK_TOKENS` | `0` | While another request decodes, a prompt chunk is read in passes of at most N tokens. | `tensorfold/engine/family_prefill.py` |
| `PLI_TRACE` | `unset` | Step trace file, on from the start (benches, tests). | `tensorfold/pli/trace.py` |
| `PLI_TRACE_LAYERS` | `0` | With the trace on, one line per chunk with the attention, DeltaNet and FFN times (slows reading). | `tensorfold/pli/trace_layers.py` |
| `PLI_TREE_EXACT2` | `1` | With `PLI_TREE_EXACT`, the per-node tails in one kernel call (same bits); `0` keeps the loop over nodes. | `tensorfold/pli/attention_attm.py` |
| `PLI_GPU_CHAIN` | `0` | DFlash2's greedy chain drawn on the GPU from its lattice; no gain measured. | `tensorfold/families/qwen3_5/dflash_head.py` |
| `PLI_CHECKPOINT_LOG` | `0` | Writes `checkpoint_kept`, `checkpoint_evicted` and `checkpoint_refused` events to the `PLI_EVENTS` log. | `tensorfold/server/checkpoints.py` |
| `PLI_TREE_GAIN_MIN` | empty | A KFlash tree is kept only if its expected tokens exceed those of DFlash2's greedy chain by at least N; otherwise the chain is verified alone. Empty: always the tree. | `tensorfold/pli/kflash.py` |
| `PLI_ROUND_PROFILE` | `0` | Where the time of a decode round goes: one `[PLI] round profile` line per kind of round with the mean time of each step (verification, head, draw, next draft...). Each step is evaluated at its end, so rounds are a little longer; same operations, same order, the text does not change. | `tensorfold/pli/round_profile.py` |
| `PLI_ROUND_PROFILE_N` | `200` | `PLI_ROUND_PROFILE`: rounds between two lines, per kind. | `tensorfold/pli/round_profile.py` |
| `PLI_ROUND_PROFILE_BUCKET` | `0` | `PLI_ROUND_PROFILE`: kinds of round also split by context bucket of N tokens (0 = not split). | `tensorfold/pli/round_profile.py` |
| `PLI_COPY_LOG` | empty | jsonl file: each round writes its kind and the tokens it gave; a copy round also writes the copy, the head's queued chain and the match length (to choose offline when the copy should go first). The choice itself does not change. | `tensorfold/engine/lane_family.py` |
| `PLI_LATTICE_LOG` | empty | Folder of the log of the DFlash2 lattices served by the head, to replay offline the chain, the trees and the candidates' ceiling on the same text. Nothing changes in the computation or in the text. | `tensorfold/pli/lattice_log.py` |

## 4. Internal tuning

Fine settings of a stable feature. Their code default is the value used in production; the profile does not set them.
Change them only to measure.

| Variable | Code default | What it tunes | Code |
| --- | --- | --- | --- |
| `PLI_ANE_BLOCK_IDS` | `1,3` | Neural Engine blocks the two programs are pinned on (`PLI_ANE_BLOCKS=2`). | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_FFN_MIN_ROWS` | `512` | Smallest chunk (rows) given a Neural Engine share when `PLI_ANE_STEPS` is empty; floor 128. | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_KEEPAWAKE_MS` | `500` | The keep-awake thread stops this many ms after the last shared layer. | `tensorfold/pli/ane/ffn.py` |
| `PLI_ANE_REQUANT_MODE` | `1` | Rounding of the CPU requantization: 1 = one rounding after FMA, as the Metal kernel; 0 = two bf16 roundings; +2 forces the scalar path. | `tensorfold/pli/ane/requant_cpu.py` |
| `PLI_ATTENTION_F16_HEADS` | `2` | Key heads converted to f16 per group in the f16 prompt attention. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_f16.py` |
| `PLI_ATTENTION_F16_MIN` | `16384` | Key threshold of the f16 prompt attention path. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_f16.py` |
| `PLI_ATTENTION_F16_PV` | `32,64,32,2,2` | Tile (BM, BN, BK, WM, WN) of the P.V product, f16 path. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_f16.py` |
| `PLI_ATTENTION_F16_QK` | `64,64,16,2,2` | Tile of the Q.Kt product, f16 path. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_f16.py` |
| `PLI_ATTENTION_GEMM_MIN` | `16384` | Key threshold of the tuned GEMM prompt attention path. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_gemm.py` |
| `PLI_ATTENTION_GEMM_PV` | `32,32,32,4,1` | Tile of the P.V product up to `PLI_ATTENTION_GEMM_PV_THRESHOLD` visible keys. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_gemm.py` |
| `PLI_ATTENTION_GEMM_PV_LONG` | `16,32,64,2,2` | Tile of the P.V product beyond that threshold. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_gemm.py` |
| `PLI_ATTENTION_GEMM_PV_THRESHOLD` | `65536` | Visible keys where the P.V tile changes. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_gemm.py` |
| `PLI_ATTENTION_GEMM_QK` | `32,64,16,1,2` | Tile of the Q.Kt product, GEMM path. | `tensorfold/kernels/qwen/dense/v1/prompt_attention_gemm.py` |
| `PLI_ATTENTION_REDUCED_ATTM` | `1` | Reduced layers go through `attm` (0 = older path without `attm`). | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ATTENTION_REDUCED_MIN_ROWS` | `2` | Fewest rows in a round for the reduction to apply (without the anchor). | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ATTENTION_REDUCED_FULL_LAYERS` | `3,7,59` | Layers always read in full by the reduced attention. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_ATTENTION_REDUCED_THRESHOLD` | `20000` | Context keys beyond which the reduced attention applies (at least 4,096). | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_HEARTBEAT_WINDOW_S` | `600` | The heartbeat runs for this many seconds after the last activity. | `tensorfold/server/scheduler.py` |
| `PLI_EVENTS_MAX_MB` | `50` | The event journal rotates to `<file>.1` beyond this size in MB. | `tensorfold/pli/events.py` |
| `PLI_HYBRID_GLOBAL_ROUNDS` | `12` | Rounds of each draft the engine measures before new conversations start on the better one. | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_SNAPSHOT_SETTINGS` | `1` | A fingerprint of the `PLI_*` settings enters the snapshot key, so a conversation saved under other settings is not resumed. | `tensorfold/cli.py` |
| `PLI_PREFILL_UNPACK_ROWS` | `512` | Rows from which prompt-reading calls unpack the weights. | `tensorfold/pli/unpack.py` |
| `PLI_RECALL_TOKENS` | `1000000` | Size bound of the Recall registry in tokens; the oldest go first. | `tensorfold/pli/recall.py` |
| `PLI_TRACE_MAX_MB` | `500` | The step trace rotates to `<file>.1` beyond this size in MB. | `tensorfold/pli/trace.py` |
| `PLI_ANE_EVAL_HINT` | `2` | Neural Engine evaluation hint: 0 none, 1 the pinned block, 2 the pinned block and variant 1. Read by the native client. | `tensorfold/pli/ane/native/ane/Program.mm` |
| `PLI_HEAD_COPY_WEIGHT` | `0.2` | `PLI_HEAD_BEFORE_COPY=3`: weight of the moving average of copy minus head. | `tensorfold/engine/lane_family.py` |
| `PLI_HEAD_COPY_THRESHOLD` | `1` | `PLI_HEAD_BEFORE_COPY=3`: the copy goes first while that average stays above minus N tokens. | `tensorfold/engine/lane_family.py` |
| `PLI_HYBRID_IM_START` | `248045` | Message start token counted by `PLI_HYBRID_MESSAGES` (Qwen3.8: `<|im_start|>`, 248045). | `tensorfold/families/qwen3_5_moe/hybrid.py` |
| `PLI_S3B_ROWS` | `2047` | Largest prompt chunk (rows) that S3b reads through the anchored reduced attention. | `tensorfold/kernels/qwen/dense/v1/exact_attention.py` |
| `PLI_CHECKPOINT_GROUP` | `64` | `PLI_CHECKPOINT_BITS`: values per quantization group. | `tensorfold/pli/checkpoint_bits.py` |
| `PLI_A1_SLICE` | `2048` | A1: keys scored per kernel call. | `tensorfold/pli/a1_lsa.py` |
| `PLI_A1_SMALL_SLICE` | `128` | A1: rows below which a chunk is scored in one call. | `tensorfold/pli/a1_lsa.py` |

## Other

| Variable | Code default | What it does | Code |
| --- | --- | --- | --- |
| `PLI_PROFILE` | unset | Machine profile to load, as `--profile` (for example `m1-ultra`). | `tensorfold/cli.py` |

## Names before 4.0.0

The names before 3.7.0, the 23 names that production used until 4.0.0, `PLI_DEBUT_DERNIER`, used until 4.0.1, and the 7 production names of the settings added in 4.1.0. Every name below is still accepted; when both are set, the new name wins. Inherited TensorFold settings (`TF_*`,
`TENSORFOLD_*`) were renamed to `PLI_*` too; they keep TensorFold's meaning and are described in TensorFold's
documentation.

| Old name | New name |
| --- | --- |
| `PLI_A1_GARDES_MAX` | `PLI_A1_MAX_TILES` |
| `PLI_A1_PETITE` | `PLI_A1_SMALL_SLICE` |
| `PLI_A1_TRANCHE` | `PLI_A1_SLICE` |
| `PLI_ADMISSION_PETITES` | `PLI_ADMIT_SMALL` |
| `PLI_ADMISSION_TRAVAIL` | `PLI_ADMIT_BY_WORK` |
| `PLI_ANCRE1_LECTURE_SEULE` | `PLI_A1_PREFILL_ONLY` |
| `PLI_ANCRE1_LSA` | `PLI_A1_LSA_MAP` |
| `PLI_ANCRE_JOURNAL` | `PLI_ANCHOR_LOG` |
| `PLI_ANE_BLOCS_IDS` | `PLI_ANE_BLOCK_IDS` |
| `PLI_ANE_BLOCS` | `PLI_ANE_BLOCKS` |
| `PLI_ANE_FFN_LIGNES` | `PLI_ANE_FFN_MIN_ROWS` |
| `PLI_ANE_FFN_UNITES` | `PLI_ANE_FFN_UNITS` |
| `PLI_ANE_GARDE_MS` | `PLI_ANE_KEEPAWAKE_MS` |
| `PLI_ANE_GARDE` | `PLI_ANE_KEEPAWAKE` |
| `PLI_ANE_PALIERS` | `PLI_ANE_STEPS` |
| `PLI_ANE_RELEVE` | `PLI_ANE_STATS` |
| `PLI_ANE_REQUANT_FILS` | `PLI_ANE_REQUANT_THREADS` |
| `PLI_ANE_RQ_APRES` | `PLI_ANE_REQUANT_AFTER` |
| `PLI_ARBITRE_JOURNAL` | `PLI_ARBITER_LOG` |
| `PLI_ARBITRE_PALIER` | `PLI_ARBITER_MIN_ROWS` |
| `PLI_ARBITRE_TABLE` | `PLI_ARBITER_TABLE` |
| `PLI_ARBITRE` | `PLI_ARBITER` |
| `PLI_ARBRES_ATTM` | `PLI_TREES_ATTM` |
| `PLI_ARBRES_COURTES` | `PLI_TREES_SHORT_ONLY` |
| `PLI_ARBRES_SEUIL_JETONS` | `PLI_TREES_TOKEN_THRESHOLD` |
| `PLI_ARBRE_ARETE` | `PLI_TREE_EDGE` |
| `PLI_ARBRE_BRUIT` | `PLI_TREE_NOISE` |
| `PLI_ARBRE_CORR` | `PLI_TREE_CORRECT` |
| `PLI_ARBRE_CTX_MAX` | `PLI_TREE_CTX_MAX` |
| `PLI_ARBRE_DN` | `PLI_TREE_DELTANET` |
| `PLI_ARBRE_ENFANTS` | `PLI_TREE_CHILDREN` |
| `PLI_ARBRE_EXACT2` | `PLI_TREE_EXACT2` |
| `PLI_ARBRE_EXACT` | `PLI_TREE_EXACT` |
| `PLI_ARBRE_GAIN_MIN` | `PLI_TREE_GAIN_MIN` |
| `PLI_ARBRE_NOEUDS` | `PLI_TREE_NODES` |
| `PLI_ARBRE_SEUL` | `PLI_TREE_ALONE` |
| `PLI_ARBRE_TAU` | `PLI_TREE_TAU` |
| `PLI_ATTENTION_F16_HC` | `PLI_ATTENTION_F16_HEADS` |
| `PLI_ATTENTION_GEMM_PVLONG` | `PLI_ATTENTION_GEMM_PV_LONG` |
| `PLI_ATTENTION_GEMM_PVSEUIL` | `PLI_ATTENTION_GEMM_PV_THRESHOLD` |
| `PLI_ATTENTION_GROUPEE` | `PLI_ATTENTION_GROUPED` |
| `PLI_ATTENTION_LIGNE` | `PLI_ATTENTION_PER_ROW` |
| `PLI_ATTENTION_REDUITE_ATTM` | `PLI_ATTENTION_REDUCED_ATTM` |
| `PLI_ATTENTION_REDUITE_MIN` | `PLI_ATTENTION_REDUCED_MIN_ROWS` |
| `PLI_ATTENTION_REDUITE_PLEINES` | `PLI_ATTENTION_REDUCED_FULL_LAYERS` |
| `PLI_ATTENTION_REDUITE_SEUIL` | `PLI_ATTENTION_REDUCED_THRESHOLD` |
| `PLI_ATTENTION_REDUITE` | `PLI_ATTENTION_REDUCED` |
| `PLI_ATTENTION_REGLAGE` | `PLI_ATTENTION_TUNING_FILE` |
| `PLI_ATTM_CHARGE` | `PLI_ATTM_LOAD` |
| `PLI_ATTM_COURT` | `PLI_ATTM_SHORT` |
| `PLI_ATTM_PETITES` | `PLI_ATTM_SMALL_KEYS` |
| `PLI_ATTN_MORCEAUX` | `PLI_ATTN_SLICES` |
| `PLI_BATTEMENT_ANE` | `PLI_HEARTBEAT_ANE` |
| `PLI_BATTEMENT_FENETRE_S` | `PLI_HEARTBEAT_WINDOW_S` |
| `PLI_BATTEMENT_MS` | `PLI_HEARTBEAT_MS` |
| `PLI_BROUILLON_ATTN` | `PLI_DRAFT_ATTN` |
| `PLI_BROUILLON_PILE` | `PLI_DRAFT_STACK` |
| `PLI_BROUILLON_QKR` | `PLI_DRAFT_QKR` |
| `PLI_BROUILLON_TOPK` | `PLI_DRAFT_TOPK` |
| `PLI_CHAINE_GPU` | `PLI_GPU_CHAIN` |
| `PLI_CONTROLE_TEXTE` | `PLI_CONTROL_AS_TEXT` |
| `PLI_COPIE_JOURNAL` | `PLI_COPY_LOG` |
| `PLI_CROISE_DENSE` | `PLI_INTERLEAVE_DENSE` |
| `PLI_DEBUT_DERNIER` | `PLI_LAST_MESSAGE_CHECKPOINT` |
| `PLI_DEVERSE_INVITE` | `PLI_SPILL_PROMPT` |
| `PLI_EVENEMENTS_MAX_MO` | `PLI_EVENTS_MAX_MB` |
| `PLI_EVENEMENTS` | `PLI_EVENTS` |
| `PLI_F16_ECRETE` | `PLI_F16_CLAMP` |
| `PLI_FALCON_CIBLES_UN_SUR` | `PLI_FALCON_TARGETS_EVERY` |
| `PLI_FALCON_CONTEXTE8` | `PLI_FALCON_CONTEXT8` |
| `PLI_FALCON_FLOT` | `PLI_FALCON_STREAM` |
| `PLI_FALCON_LOT` | `PLI_FALCON_BATCH` |
| `PLI_FALCON_MAX_GO` | `PLI_FALCON_MAX_GB` |
| `PLI_FALCON_TETE` | `PLI_FALCON_HEAD` |
| `PLI_FONDRE_FIN` | `PLI_MERGE_TAIL` |
| `PLI_FOURCHE_MIN` | `PLI_FORK_MIN` |
| `PLI_GARDE_HOTE` | `PLI_HOST_GUARD` |
| `PLI_GARDE_REPONSE` | `PLI_KEEP_REPLY` |
| `PLI_HEURES` | `PLI_LOG_TIMES` |
| `PLI_HOTES_ADMIS` | `PLI_ALLOWED_HOSTS` |
| `PLI_HYBRIDE_ARBRES` | `PLI_HYBRID_TREES` |
| `PLI_HYBRIDE_FORCE` | `PLI_HYBRID_FORCE` |
| `PLI_HYBRIDE_GLOBAL_TOURS` | `PLI_HYBRID_GLOBAL_ROUNDS` |
| `PLI_HYBRIDE_IM_START` | `PLI_HYBRID_IM_START` |
| `PLI_HYBRIDE_JOURNAL` | `PLI_HYBRID_LOG` |
| `PLI_HYBRIDE_MESSAGES` | `PLI_HYBRID_MESSAGES` |
| `PLI_HYBRIDE_PROFONDEUR` | `PLI_HYBRID_DEPTH` |
| `PLI_HYBRIDE` | `PLI_HYBRID` |
| `PLI_INSTANTANE_REGLAGES` | `PLI_SNAPSHOT_SETTINGS` |
| `PLI_KV_PARTAGE` | `PLI_KV_SHARING` |
| `PLI_LECTURE_CPU_MIN` | `PLI_PREFILL_CPU_MIN` |
| `PLI_LECTURE_CPU_MODE` | `PLI_PREFILL_CPU_MODE` |
| `PLI_LECTURE_CPU_PRIORITE` | `PLI_PREFILL_CPU_PRIORITY` |
| `PLI_LECTURE_CPU_TRACE` | `PLI_PREFILL_CPU_TRACE` |
| `PLI_LECTURE_CPU` | `PLI_PREFILL_CPU` |
| `PLI_LECTURE_DEPLIE_CONTROLE` | `PLI_PREFILL_UNPACK_CHECK` |
| `PLI_LECTURE_DEPLIE_LIGNES` | `PLI_PREFILL_UNPACK_ROWS` |
| `PLI_LECTURE_DEPLIE_MLP` | `PLI_PREFILL_UNPACK_MLP` |
| `PLI_LECTURE_DEPLIE_NN` | `PLI_PREFILL_UNPACK_NN` |
| `PLI_LECTURE_DEPLIE_NOYAU` | `PLI_PREFILL_UNPACK_KERNEL` |
| `PLI_LECTURE_DEPLIE_PILE` | `PLI_PREFILL_UNPACK_STACK` |
| `PLI_LECTURE_DEPLIE` | `PLI_PREFILL_UNPACK` |
| `PLI_LISTE_PLUS` | `PLI_EXTENDED_DRAFT_LIST` |
| `PLI_MLP_ENT` | `PLI_MLP_INT` |
| `PLI_MORCEAU_COUCHES` | `PLI_CHUNK_LAYERS` |
| `PLI_MORCEAU_JETONS` | `PLI_CHUNK_TOKENS` |
| `PLI_MTP27_BROUILLONS` | `PLI_MTP_DRAFTS` |
| `PLI_MTP27_FENETRE` | `PLI_MTP_WINDOW` |
| `PLI_MTP27` | `PLI_MTP_HEAD` |
| `PLI_PAVES_PAR_GROUPE` | `PLI_TILES_PER_GROUP` |
| `PLI_POINTS_BITS` | `PLI_CHECKPOINT_BITS` |
| `PLI_POINTS_GROUPE` | `PLI_CHECKPOINT_GROUP` |
| `PLI_POINTS_JOURNAL` | `PLI_CHECKPOINT_LOG` |
| `PLI_PROFIL_TOUR_N` | `PLI_ROUND_PROFILE_N` |
| `PLI_PROFIL_TOUR_TRANCHE` | `PLI_ROUND_PROFILE_BUCKET` |
| `PLI_PROFIL_TOUR` | `PLI_ROUND_PROFILE` |
| `PLI_PROFIL` | `PLI_PROFILE` |
| `PLI_RANGE_BROUILLON` | `PLI_RELAYOUT_DRAFT` |
| `PLI_RANGE` | `PLI_RELAYOUT` |
| `PLI_RAPPEL_JETONS` | `PLI_RECALL_TOKENS` |
| `PLI_RAPPEL` | `PLI_RECALL` |
| `PLI_REDUITE_ANCRE` | `PLI_REDUCED_ANCHOR` |
| `PLI_REDUITE_TOURS` | `PLI_REDUCED_ROUNDS` |
| `PLI_RELIRE_DISQUE` | `PLI_REREAD_DISK` |
| `PLI_RQ_NC_MAX` | `PLI_REQUANT_NC_MAX` |
| `PLI_S3B_LIGNES` | `PLI_S3B_ROWS` |
| `PLI_S3B_SEUIL` | `PLI_S3B_THRESHOLD` |
| `PLI_TETE_AVANT_COPIE` | `PLI_HEAD_BEFORE_COPY` |
| `PLI_TETE_COPIE_POIDS` | `PLI_HEAD_COPY_WEIGHT` |
| `PLI_TETE_COPIE_SEUIL` | `PLI_HEAD_COPY_THRESHOLD` |
| `PLI_TOURS_ENTRE_LECTURES` | `PLI_ROUNDS_BETWEEN_PREFILLS` |
| `PLI_TRACE_COUCHES` | `PLI_TRACE_LAYERS` |
| `PLI_TRACE_DOSSIER` | `PLI_TRACE_DIR` |
| `PLI_TRACE_MAX_MO` | `PLI_TRACE_MAX_MB` |
| `PLI_TREILLIS_JOURNAL` | `PLI_LATTICE_LOG` |
| `TENSORFOLD_API_KEY` | `PLI_API_KEY` |
| `TENSORFOLD_MEMORY_LIMIT_GB` | `PLI_MEMORY_LIMIT_GB` |
| `TENSORFOLD_NO_LIVE` | `PLI_NO_LIVE` |
| `TENSORFOLD_NO_UPDATE_CHECK` | `PLI_NO_UPDATE_CHECK` |
| `TENSORFOLD_REQUEST_LOG` | `PLI_REQUEST_LOG` |
| `TENSORFOLD_SEED_SALT` | `PLI_SEED_SALT` |
| `TENSORFOLD_TEST_LAUNCHD` | `PLI_TEST_LAUNCHD` |
| `TENSORFOLD_VIDEO_TOKENS` | `PLI_VIDEO_TOKENS` |
| `TF_ATTN_DIRECT_P` | `PLI_ATTN_DIRECT_P` |
| `TF_ATTN_TILES` | `PLI_ATTN_TILES` |
| `TF_COPY_MATCH` | `PLI_COPY_MATCH` |
| `TF_COPY_ROWS` | `PLI_COPY_ROWS` |
| `TF_DRAFT_CAPTURE` | `PLI_DRAFT_CAPTURE` |
| `TF_DRAFT_LEAN` | `PLI_DRAFT_LEAN` |
| `TF_DRAFT_LOG` | `PLI_DRAFT_LOG` |
| `TF_DRAFT_PROFILE` | `PLI_DRAFT_PROFILE` |
| `TF_DRAFT_VOCAB` | `PLI_DRAFT_VOCAB` |
| `TF_FAMILY_PROFILE` | `PLI_FAMILY_PROFILE` |
| `TF_FAMILY_ROUND_LOG` | `PLI_FAMILY_ROUND_LOG` |
| `TF_FLASH_DENSE` | `PLI_FLASH_DENSE` |
| `TF_FLASH_DRAFT_VOCAB` | `PLI_FLASH_DRAFT_VOCAB` |
| `TF_FLASH_FUSED` | `PLI_FLASH_FUSED` |
| `TF_FLASH_MTP` | `PLI_FLASH_MTP` |
| `TF_FLASH_PREFILL` | `PLI_FLASH_PREFILL` |
| `TF_FLASH_QUEUED` | `PLI_FLASH_QUEUED` |
| `TF_GLM_CLEAR_THINKING` | `PLI_GLM_CLEAR_THINKING` |
| `TF_GLM_DENSE` | `PLI_GLM_DENSE` |
| `TF_LANE_TILE` | `PLI_LANE_TILE` |
| `TF_MOE_ROWS` | `PLI_MOE_ROWS` |
| `TF_MTP_ROUNDS` | `PLI_MTP_ROUNDS` |
| `TF_NEMOTRON_MTP` | `PLI_NEMOTRON_MTP` |
| `TF_NGRAM_HOST` | `PLI_NGRAM_HOST` |
| `TF_NGRAM_WEIGHT` | `PLI_NGRAM_WEIGHT` |
| `TF_QWEN36_DRAFT_VOCAB` | `PLI_QWEN36_DRAFT_VOCAB` |
| `TF_QWEN36_MTP` | `PLI_QWEN36_MTP` |
| `TF_REQUIRE_KERNELS` | `PLI_REQUIRE_KERNELS` |
| `TF_ROW_ATTENTION` | `PLI_ROW_ATTENTION` |
| `TF_TREE_TRACE` | `PLI_TREE_TRACE` |