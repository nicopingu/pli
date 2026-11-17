# PLI features

Every feature PLI adds, stable (in production on the validated machine) or experimental (off by default),
with its settings and whether it changes the text. Every setting, with its default: [switches.md](switches.md).

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
| Generation in child processes: voice, transcription, image, video; memory accounting, pause while the LLM writes, cancellation (499), prompt guard | `PLI_GEN` (env), `PLI_GEN_CONFIG` (env), `PLI_GEN_*` | same text for the LLM | 5.0 to 5.4 |
| Background tasks: snapshot pruning, drafter swapped hot with a guard, drafter tuning with the served model | `PLI_TASKS` (env), `PLI_TASKS_*` | same text (the drafter only changes speed) | 5.6 |
| DFlash2 trees at the chains' bits: the draft does not change the text at temperature 0 | `PLI_TREE_EXACT` | same text | 5.6.2 |
| The LLM capped while a generation child runs (cache emptied, memory limit frozen, prompt budget lowered), given back when the last child stops; freeing waits up to 60 s for the LLM to be idle | `PLI_GEN_ENGINE_WAIT_S` | same text | 5.6.3 |
| Startup measurements kept on disk: a 27B start in 9.0 s instead of 71.1 s, same figures and same text | `--startup-cache on` | same text | 5.6.3 |
| Room asked of an outside scheduler when a generation child lacks memory: a request file, read by the scheduler, which pauses its own jobs; up to 120 s of wait | `place_fichier`, `place_attente_s` (`PLI_GEN_CONFIG`) | same text | 5.6.4 |
| `/health`, `gen`: per family, the model served by the last request and the requests in progress | none | same text | 5.6.4 |
| Video LoRA applied in memory by h3, up to 4 at once, no fused copy on disk (byte-identical video to the fused one) | mode `en_memoire`, `lora_natif` (video catalog); `tools/pli-lora-natif`, `tools/h3c-lora.patch` | same video as the fusion | 5.6.5 |

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

The full list, with each variable's code default, is in [docs/switches.md](switches.md). Every setting starts with
`PLI_`; the names used before 3.7.0 and 4.0.0 (French `PLI_*` names, `TF_*`, `TENSORFOLD_*`) are still accepted.
