# Model families

The CLI discovers family packages by `MODEL_TYPES`, matching the checkpoint configuration.
Every family runs on the MLX lane engine (Apple Silicon only).

| Package | Model | Drafting |
| --- | --- | --- |
| `nemotron_h/` | Nemotron 3.5 Lightning | MTP and context copies |
| `qwen3_5/` | Qwen3.8-27B (PLI's main target) | DFlash2 and context copies |
| `qwen3_5_moe/` | Qwen3.6-35B-A3B | MTP and context copies |
| `qwen4_exp/` | Qwen3.8 Flash Next | MTP and context copies |
| `glm5_next/` | GLM-5.3-Flash (256 GB, M3 Ultra; its kernels need 1,024 threads per threadgroup) | MTP |
| `gemma4/` | Gemma 4 26B-A4B | Context copies |
| `bonsai/` | Ternary Bonsai 2 27B | DFlash2 and context copies |
| `deepseek_v4/` | DeepSeek-V4-Flash (256 GB) | DSpark and MTP |

Load-time checks determine the usable window width and shared-forward support. Each stream has
independent state; shared execution must reproduce its solo output. Backend support is declared by
the package, not inferred from a model's name.

See [the recipe book](../../../docs/recipes/README.md) for checkpoints and limits,
[the family interface](../../../docs/recipes/adding-a-family.md) and
[the kernel map](../kernels/README.md).
