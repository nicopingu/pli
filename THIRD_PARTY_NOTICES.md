# Third-party notices

PLI is a fork of [TensorFold](https://github.com/ashhart/TensorFold) 0.6.6, Apache License 2.0, Copyright the
TensorFold contributors (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).

PLI uses [MLX](https://github.com/ml-explore/mlx) and
[mlx-lm](https://github.com/ml-explore/mlx-lm), MIT License, Copyright © 2023 Apple Inc.
They are installed as dependencies.

## MLX and mlx-lm adaptations

The DeltaNet implementations in `src/tensorfold/kernels/qwen/dense/v1/lane_gdn.py` and
`lane_tree.py` adapt mlx-lm's `qwen3_5` and `gated_delta` model math and kernels under its MIT License.

## Qwen Flash Next

The n-gram ID helpers in `src/tensorfold/families/qwen4_exp/model.py` translates
Hugging Face transformers' `models/qwen4_exp/modeling_qwen4_exp.py` into MLX and NumPy with renamed
identifiers. Copyright 2026 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
See [the license text](LICENSES/Apache-2.0.txt). The same helpers appear in mlx-vlm's
`models/qwen4_exp/language.py`, MIT License, Copyright © 2025 Prince Canuma.

## GLM-5.3-Flash on Apple Silicon

The MLX engine of `glm5_next` (`src/tensorfold/families/glm5_next/`: the forward pass in `model.py`, `kda.py`,
`mla.py` and `mlp.py`, the draft head in `mtp.py`, `runtime.py`) and its Metal kernels
(`src/tensorfold/kernels/glm/flash/v1/`) were written for TensorFold. What they follow or port:

- The forward pass follows, op for op on its prefill path, the GLM-5.3-Flash (`glm5_next`)
  implementation added to [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) by PR #2030 (by Lazarus-931; MIT License,
  Copyright (c) 2025 Prince Canuma), as vendored by [oMLX](https://github.com/jundot/omlx) (Apache-2.0). Nothing is
  imported from either at runtime.
- `kernels/glm/flash/v1/kda.py` is ported from mlx-vlm PR #2105 ("glm5_next: fuse the KDA decode chain into one
  Metal kernel", by avlp12; `mlx_vlm/models/glm5_next/fused_kda.py`; closed without merging, MIT License,
  Copyright (c) 2025 Prince Canuma): the whole KDA decode step in one Metal kernel. TensorFold runs a window of
  rows in order inside the launch, folds the 4-bit `f_b` / `g_b` projections in with MLX's one-row `qmv_quad`
  arithmetic, and keeps its own rounding points. Its precision rules (precise exp, uncontracted sums of squares)
  are also used in `fused.py`, `moe.py` and `hc.py`.
- `kernels/glm/flash/v1/sparse_attention.py` is mlx-vlm's `indexed_sparse_attention` kernel
  (`mlx_vlm/models/sparse_attention.py`) as extended by mlx-vlm PR #2245 ("Fix GLM-5.3 cached decode batch
  invariance", by raullenchai; closed without merging, MIT License, Copyright (c) 2025 Prince Canuma), adapted to
  TensorFold's single latent cache.
- mlx-vlm PR #2107 (the sparse indexer's incremental decode and a stale-pool fix, by avlp12) needed no code:
  TensorFold's cache already pools once per completed block. Its stale-pool case is pinned by
  `tests/test_glm5_ported_kernels.py`.
- The hyper-connection kernel `_HC_SPLIT` in `kernels/glm/flash/v1/kernels.py`, and the sinkhorn and collapse in
  `hc.py`, repeat the `hc_sinkhorn_collapse` kernel of mlx-vlm's `mlx_vlm/models/deepseek_v4/hyper_connection.py`
  (MIT License, Copyright (c) 2026 Apple Inc.), with its output type set to the input's.
- The 4-bit matvec `_QMV_ROWS` in `kernels.py` is Flash Next's `qmv_rows` with MLX's group-64 scale indexing, and
  the expert kernels (`_EXPERT_GROUP`, `_EXPERT_QMV`) follow Flash Next's `expert_group` / `grouped_gateup`. The
  row kernels in `kernels.py`, `moe.py` and `hc.py` repeat the arithmetic and partitions of MLX 0.32's own kernels (MIT
  License, Copyright © 2023 Apple Inc.): `qmv_fast`, `qmv_quad` and `gather_qmv_fast` (`quantized.h`), `GEMVKernel`
  and `GEMVTKernel` (`gemv.h`) and the `rms_norm` kernels, one row per grid slice with the tiling MLX picks for one
  row, so each row keeps MLX's one-row bits.

## Added by PLI

- **Neural Engine client** (`src/tensorfold/pli/ane/native/`): `ane/Program.mm`, `ane/Program.hpp`,
  `ane/ProgramInstrumentation.hpp`, `AwakeClock.hpp` and `Checked.hpp` are taken from
  [Splash](https://github.com/incoai/splash) as ported to M1/M2 GPUs by
  [paperniuk/splash](https://github.com/paperniuk/splash) (version 1.3.2-m1, branch `apple7-m1-1.3.2`, commit 5c484b2),
  Apache License 2.0; the license text is in `src/tensorfold/pli/ane/native/LICENSE-splash`. `ProgramInstrumentation.hpp`,
  `AwakeClock.hpp` and `Checked.hpp` are unmodified. PLI modifies `ane/Program.mm` and `ane/Program.hpp` to pin a
  program to one Neural Engine block (`setNextBlock`, load and evaluation hints, a block suffix in the program key);
  every change is marked `PLI` in the code. `pli_ane.mm` and
  `metal/MetalBackend.hpp` (which replaces Splash's Metal backend) are written for PLI. The MIL program of the shared FFN
  in `src/tensorfold/pli/ane/client.py` and the split in `ffn.py` are rewritten in Python from Splash's
  `runtime/ops/AneFfn.cpp` (`AneFfn::function`), Apache License 2.0.
- **8-row verify GEMM** (`src/tensorfold/kernels/qwen/dense/v1/qmm8.py`, `PLI_V8`): the X8 input table (`x8_prep`) and the
  dense 4-bit GEMM `qmm8snb_b4_nf2` are ported from [Kapitsa](https://github.com/nace-ai/kapitsa) v0.2.23,
  `src/kernels/verify8.metal`, with Kapitsa's M1 tiling (`src/model/qwen35_v8.cpp`, `v8_plan`). Copyright 2026 Nace AI,
  Apache License 2.0; see [the license text](LICENSES/Apache-2.0.txt). PLI turns them into MLX `metal_kernel` bodies, adds a
  variant that reads PLI's relaid-out weights, row tiles of 8 on grid y for up to 128 rows and bf16 outputs.
- **Kapitsa's methods** reimplemented without copying code: draft trees with calibrated probabilities (KFlash,
  `src/tensorfold/pli/kflash.py`, from Kapitsa's `docs/kflash.md`), checkpoints that keep a length instead of a copy of the
  KV cache (`src/tensorfold/pli/kv_sharing.py`), the tree attention tail in one kernel call (`sdpa_tree_exact2` in
  `src/tensorfold/pli/attention_attm.py`) and the draft chain drawn on the GPU (`DFlashHead._gpu_chain`).
- **Recall drafts** (`src/tensorfold/pli/recall.py`) implement the idea of [TensorFold 1.0.12](https://github.com/ashhart/TensorFold/releases/tag/v1.0.12)'s "Recall"
  (Apache License 2.0); no code is copied (TensorFold 1.0 is written in Zig).
- **Checkpoint at the fork point** (`src/tensorfold/server/checkpoints.py`, `PLI_FORK_MIN`): the rule of TensorFold
  pull request [#1291](https://github.com/ashhart/TensorFold/pull/1291) ("prompt cache: a miss keeps the state where the
  next prompt forks", by paragontasx, for issue [#1262](https://github.com/ashhart/TensorFold/issues/1262); closed without
  merging; TensorFold is under the Apache License 2.0). The pull request is written in Zig; PLI rewrites the rule in
  Python and copies no code.
- **Control tokens kept as text** (`src/tensorfold/server/text.py`, `PLI_CONTROL_AS_TEXT`): the behaviour of
  [TensorFold 1.1.2](https://github.com/ashhart/TensorFold/releases/tag/v1.1.2) for issue
  [#1117](https://github.com/ashhart/TensorFold/issues/1117) (Apache License 2.0), rewritten in Python; no code is copied.
- **A1 scored by the LSA index** (`src/tensorfold/pli/a1_lsa.py`, `PLI_A1_LSA_MAP`): the index queries and keys of
  [Latent Sparse Attention](https://github.com/Prefrontal-AI/latent-sparse-attention) (Prefrontal-AI, commit 906be19,
  Apache License 2.0) score PLI's tiles of keys; the rule is rewritten in Python and Metal for PLI, no code is copied.
  The distilled map (release `d-lsa-weights-v1`, Apache License 2.0) is not shipped: `pli setup` downloads it from
  that release after asking, and `src/tensorfold/install/lsa_map.py` (written for PLI) converts it to NumPy.
- Everything else under `src/tensorfold/pli/` and the PLI changes in the kernels are written for PLI.

## Vendored code and weights

`src/tensorfold/drafters/vendor/z_lab_dflash/model_mlx.py` is the unmodified `dflash/model_mlx.py` from
[z-lab/dflash](https://github.com/z-lab/dflash), MIT License, Copyright © 2026 Z Lab.

`src/tensorfold/families/deepseek_v4/vendor/encoding_dsv4.py` is the unmodified `encoding/encoding_dsv4.py` of
[deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (revision 60d8d70), and
`tests/fixtures/deepseek_v4/` holds two of its test cases, MIT License, Copyright (c) 2023 DeepSeek.
The MTP layer PLI drafts with comes from that checkpoint's last shard (MIT), converted by
`families/deepseek_v4/convert.py`.

PLI ships no model weights. The model cards of `z-lab/Qwen3.8-27B-DFlash2`, `EigenLabs/Qwen3.8-27B-MTP-bf16`,
`mlx-community/Qwen3.8-27B-4bit` and `sabicap/sabi-noema` state Apache-2.0.
The optional `incoai/GLM-5.3-Flash-DFlash2` model card states CC BY-NC-ND 4.0, for non-commercial use
without derivatives. Each checkpoint keeps its own license.

## MIT License text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

