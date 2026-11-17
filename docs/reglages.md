# PLI Settings

This document lists all PLI settings and describes their organization into families (decision of 11/15/2026).
The inventory (section 3) is produced by `tools/pli-reglages-inventaire`: it reruns on every code change
and loads no model.

## 1. Comment lire l'inventaire

A setting comes from one of three sources:

- an **option** of `pli serve` (`--context`, `--parallel`...), read in `src/pli/cli_args.py`;
- an environment **variable** `PLI_*` (and the two `MLX_*` read by MLX), read in `src/`;
- a **JSON key** from generation files (`PLI_GEN_CONFIG` and the files it names).

Colonnes :

- **Key**: the stable name `family.setting` (see the design, section 2).
- **Option**: the existing option kept, then the `--family.setting` form.
- **Variable**: the name read by the code, then the name derived from the key (`PLI_FAMILY_SETTING`) when it differs, then in
  parentheses the old names still accepted (table in `src/pli/additions/env_names.py`).
- **Type**: deduced from what the code does with the value (`int()`, `float()`, comparison to `"1"`...). flag: 0 or 1.
- **Default**: the code value when nothing is set; "computed" when the code computes it.
- **Read at**: `file:line` under `src/` (the first two locations, then the count of others).
- **Production**: value in the machine's production files (common = `pli.env`, 27b = `pli-27b.env`, 35b =
  `pli-35b.env`, options in `config.yaml`). Paths, hosts, and keys are not copied.
- **m1-ultra profile**: value set by `src/pli/additions/profiles/m1-ultra/*.env`.
- **Aide** : reprise de `docs/switches.md` (statut entre crochets) ou de l'aide de l'option.

Rerun: `tools/pli-reglages-inventaire` (with the PLI venv Python). `--verifier` says if
`src/pli/config/table.json` (the table that `pli config` will read) is up to date; `--sans-prod` ignores production.

## 2. Design (written before the code)

### 2.1 But

A setting has a single readable name, `family.setting`, and is set the same way everywhere: in a `pli.toml` file
organized by families, as an option, or as a variable. Commands in the style of `occ` (Nextcloud) list it, read it,
write it, verify it, and explain where its value comes from. Nothing changes for those who do not use it: same
`PLI_*` variables, same options, same `.env` files.

### 2.2 The `pli.toml` file

One section per family, one line per setting. `#` comments are allowed.

```toml
[cache]
context = 229376
spill_gib = 120.0
checkpoint_bits = 4

[brouillons]
hybrid = true
mtp_head = "/chemin/vers/mtp-4bit.safetensors"
mtp_head_drafts = 7

[ane]
ffn = true
steps = "256:14,512:16,1536:18"

[gen]
actif = true
config = "/chemin/vers/pli-gen.json"
```

Types: `flag` is written `true`/`false` (becomes `1`/`0` in the environment), `integer`, `float`, `text`, `list`
(for `--alias`, `--api-key`). A value that a type does not render identically (for example `"015"`) remains written
as text in quotes: the string passed to the code is always exactly the one from before.

The file is only read if it is named: `pli serve --config FILE` or `PLI_CONFIG=FILE`. No default location
is read silently (an accidental `pli config:set` must not change production on the next
startup).

### 2.3 Noms

- **Key**: `family.setting`. The family is one of the 17 in the inventory. The setting is the current English name without
  `PLI_` and without the prefix that repeats the family (`ane_`, `attention_`, `prefill_`, `tree_`/`trees_`, `draft_`,
  `gen_`). Exemples : `PLI_ANE_FFN` donne `ane.ffn`, `PLI_TREE_DELTANET` donne `arbres.deltanet`,
  `PLI_CHECKPOINT_BITS` gives `cache.checkpoint_bits`, `--spill-gib` gives `cache.spill_gib`. Exceptions are
  written manually in the script (`CLES_FORCEES`): `PLI_MTP_DRAFTS` becomes `brouillons.mtp_head_drafts` because
  `--mtp-drafts` (`brouillons.mtp_drafts`) is a different setting. Keys are fixed in
  `src/pli/config/table.json`; a test verifies that the table follows the code.
- **Option**: the existing option remains (`--context`); `--family.setting VALUE` (or `=VALUE`) is accepted for
  all settings, including the 230 variables. A flag alone (`--ane.ffn`) is true.
- **Variable**: the current English name remains the one read by the code (no read location changes). The name derived
  from the key, `PLI_FAMILY_SETTING` (for example `PLI_TREES_DELTANET`), is accepted in addition when it differs. It
  is not created if it collides with another variable or with the old name of another (the script checks this). For an
  option, the variable derived from the key (`PLI_CACHE_CONTEXT`) provides its default value.
- **Old names**: those from `env_names.py` remain accepted in the environment and in profiles (a
  `pli.toml` uses only keys). At `pli serve` startup, a single warning line lists the old names
  set, with the name and key to use. Today no option has an old name.

### 2.4 Priority order

option > variable > `pli.toml` > machine profile > code default.

Implementation, in `cli.main`, before the profile and before any kernel import (at the same time as the profile
today, so with the same effect as a variable set in the shell):

1. `pli/__init__.py` (unchanged): an old name set gives its value to the new name.
2. The `--family.setting` from the command line are removed from argv: for a variable, `os.environ[VAR] = value`
   (the option wins); for an option, they are rewritten as the existing option.
3. The names derived from keys (`PLI_FAMILY_SETTING`) set the variable read by the code if it is not already set,
   then are removed from the environment.
4. `pli.toml`: `os.environ.setdefault(VAR, value)` for variables; for options, the value becomes the
   parser default (a command-line option wins; a `PLI_CACHE_CONTEXT` variable wins over the file).
5. The profile (unchanged): `setdefault`, so after everything else.

`PLI_CONFIG`, the names derived from keys and option variables are removed from the environment once read: they
do not enter the fingerprint of the checkpoint key settings (`code_key.settings_fingerprint` reads
all `PLI_*`), and child processes only see the usual names.

### 2.5 Commandes

| Command | Role |
| --- | --- |
| `pli config:list [family] [--deployments]` | the settings of a family (or all) with their value and source |
| `pli config:get key` | the value only (for a script) |
| `pli config:set key value` | writes the value to `pli.toml` (checked first); `config:set key --remove` removes the line |
| `pli config:check` | validates types and values of the file and environment; reports unknown and legacy names |
| `pli config:explain key` | help, type, default, read location, accepted names, value and where it comes from |
| `pli config:import-env A.env [B.env]` | translates `.env` files (read in order, last one wins) to `pli.toml` |

Target file: `--file F`, otherwise `PLI_CONFIG`, otherwise `./pli.toml`. The commands load no model and
do not import MLX. The machine profile counts in `list`, `get` and `explain` if named (`--profile NAME`
`--famille-modele qwen3_5`).

### 2.6 Assumptions

- H1. The code reads its variables at module import or later. `import pli.cli` imports neither MLX nor kernel
  (verified on 11/15/2026): setting the environment in `cli.main` is equivalent to setting it in the shell, including for
  `MLX_MAX_MB_PER_BUFFER` and `MLX_MAX_OPS_PER_BUFFER`, read by MLX at its import in `cmd_serve` (test
  `test_mlx_n_est_pas_importe_avant_la_preparation`).
- H2. The checkpoint key depends only on the final environment and hashed files. The new code is
  in `src/pli/config/`, `cli.py` and `cli_args.py`, outside `src/pli/additions` and the engine files
  hashed: `|code=` does not change (verified by test), and the same environment gives the same fingerprint.
- H3. The types inferred by the script are correct for all production and profile values (verified). A
  wrong type elsewhere does not block the service: only `config:check` complains, and `set` says so.
- H4. The JSON generation files remain what they are in this batch: `pli.toml` names them (`gen.config`), and
  the inventory lists their keys.

### 2.7 Alternatives and trade-offs

- **A (chosen): a translation layer.** The key is translated to the name read by the code. No read location changes,
  nor does the checkpoint key. Trade-off: two variable names live side by side (the current English one
  and `PLI_FAMILY_SETTING`), and the short name of a key is not always the variable name.
- **B: rename everywhere to `PLI_FAMILY_SETTING`** (like the English rename of 10/27). A single name. But about
  230 settings and hundreds of read locations affected, hashed files modified (all saved conversations
  lost at deployment), all bench scripts to redo or covered by 230 more legacy names.
- **C: the code reads `reglages.get("cache.checkpoint_bits")`** instead of `os.environ`. Cleanest long-term (types and
  defaults in one place), same cost as B; possible later family by family, above A.

### 2.8 Risks and fallback

- R1. A mistranslated value changes a calculation: variable-by-variable equivalence test on the production configuration
  (the two production `.env` files yield the same environment as the generated `pli.toml`), plus
  `config:check`.
- R2. One extra warning line at startup in production while `pli.env` keeps its old names
  (about 60): no effect on calculation; running `pli config:import-env` makes it disappear.
- R3. A name derived from a key might one day collide with a new variable: the script refuses collisions and
  the `--verifier` test fails if the table is not regenerated.
- R4. Remove `PLI_CONFIG` and the names derived from environment keys: only `cli.main` reads them.

Fallback: as long as production keeps its `.env` and uses neither `--config` nor `PLI_CONFIG`, the new code does
nothing (tested: without file, without `--famille.reglage` and without `PLI_FAMILLE_REGLAGE`, arguments and environment
identical). After a deployment with `pli.toml`, removing `PLI_CONFIG` from the launch is enough to revert to `.env`; otherwise
revert to the previous commit.

### 2.9 To be decided

1. Variable names: A (current English + `PLI_FAMILLE_REGLAGE` accepted) or B (rename for good).
2. Reading `pli.toml`: only if named (retained) or also a default location.
3. Switch production to `pli.toml` (`pli config:import-env`) or keep the `.env`.
4. Merge the generation JSONs into `pli.toml` (`[gen.image]`...) or keep them separate.
5. The dead settings found by the inventory: remove them from `env_names.py` and the docs, or keep them.

### 2.10 Statuses (added on 11/15/2026)

Each setting has a status, calculated by the inventory script and stored in the table:

- **recommends**: safe, documented, and measured or used in production. Only settings deployed by the
  reference production (`tests/fixtures/reglages`, cleaned copy of the 11/15/2026 production) or by the
  `m1-ultra` profile (measured), those classified stable by `docs/switches.md`, and documented options of `pli serve`;
- **experimental**: research, effect not guaranteed (`docs/switches.md` "experimental" or "fine-tuning", variables
  inherited from TensorFold not documented, fine-tuning options);
- **obsolete**: kept for compatibility (`--backend`, `--no-update-check`, `PLI_NO_UPDATE_CHECK`). Old
  variable names are reported separately by `config:check`.

A production setting whose docs say otherwise is ranked "recommends" but marked "(?)": the list of cases
douteux is in the inventory. The status is visible in `pli config:list` (filter `--statut`), `pli config:explain`,
and `pli config:check` warns when an experimental or obsolete setting is deployed.

### 2.11 `PLI_ATTENTION_PER_ROW` split into two settings (decision of 11/15/2026)

The code reads `PLI_ATTENTION_PER_ROW` in two places (`exact_attention.py:22` and `:26`): "1" disables query grouping,
a number N > 1 is a threshold. Both reads mean the same thing: one call per line as soon as N keys are present,
and N = 1 means from the first key, so always. The separation is done without touching the kernel:

- `attention.per_row` (`PLI_ATTENTION_PER_ROW`) keeps the threshold meaning (production: 15000);
- `attention.per_row_always` (`PLI_ATTENTION_PER_ROW_ALWAYS`, new flag) is translated by the settings layer into
  `PLI_ATTENTION_PER_ROW=1`, then removed from the environment. If set with another value of
  `PLI_ATTENTION_PER_ROW` (variable or command line), it is refused; if coming from `pli.toml`, it yields to the variable.

Why not in the kernel: `families.kernel_source_version` hashes the sources of the kernel package
(`kernels/qwen/dense/v1`) in the checkpoint key. Changing `exact_attention.py` would lose all
conversations guarded for the 27B at deployment. The flag can move into the kernel during a change that already changes
this key (the renaming from TensorFold to PLI, for example).

## 3. Inventaire

<!-- debut inventaire (generated by tools/pli-reglages-inventaire, do not edit by hand) -->

### Count by family

| Family | Options | Variables | JSON Keys | In production |
| --- | --: | --: | --: | --: |
| modele | 15 | 7 | 0 | 0 |
| brouillons | 4 | 43 | 0 | 0 |
| cache | 7 | 12 | 0 | 0 |
| memoire | 4 | 5 | 0 | 0 |
| parallele | 2 | 7 | 0 | 0 |
| lecture | 1 | 18 | 0 | 0 |
| attention | 0 | 43 | 0 | 0 |
| arbres | 0 | 17 | 0 | 0 |
| ane | 0 | 13 | 0 | 0 |
| gen | 0 | 17 | 8 | 0 |
| gen.voix | 0 | 4 | 10 | 0 |
| gen.image | 0 | 0 | 16 | 0 |
| gen.video | 0 | 2 | 20 | 0 |
| gen.lora | 0 | 0 | 16 | 0 |
| taches | 0 | 28 | 0 | 0 |
| reseau | 6 | 6 | 0 | 0 |
| journal | 0 | 26 | 0 | 0 |
| expert | 1 | 13 | 0 | 0 |
| **total** | **40** | **261** | **70** | **0** |

Production files missing on this machine: « production » columns empty.

### modele

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `modele.alias` | recommande | `--alias`, `--modele.alias` | `PLI_MODELE_ALIAS` | liste | `[]` | `pli/cli_args.py:40` |  |  | another model id to answer to |
| `modele.backend` | obsolete | `--backend`, `--modele.backend` | `PLI_MODELE_BACKEND` | choix : auto/mlx | `auto` | `pli/cli_args.py:128` |  |  | MLX on Apple Silicon (the only backend PLI has; kept for compatibility) |
| `modele.control_as_text` | recommande | `--modele.control_as_text` | `PLI_CONTROL_AS_TEXT` or `PLI_MODELE_CONTROL_AS_TEXT` (`PLI_CONTROLE_TEXTE`) | drapeau | `1` | `pli/server/text.py:128` |  |  | Control tokens written in the text of a request (messages, tools, tool results) stay ordinary text; 0: the former behaviour. *[switches.md : stable]* |
| `modele.glm_clear_thinking` | experimental | `--modele.glm_clear_thinking` | `PLI_GLM_CLEAR_THINKING` or `PLI_MODELE_GLM_CLEAR_THINKING` (`TF_GLM_CLEAR_THINKING`) | texte | `0` | `pli/families/glm5_next/prompts.py:24` |  |  |  |
| `modele.max_tokens` | recommande | `--max-tokens`, `--modele.max_tokens` | `PLI_MODELE_MAX_TOKENS` | entier | `4096` | `pli/cli_args.py:55` |  |  | reply tokens when a request does not say |
| `modele.min_p` | recommande | `--min-p`, `--modele.min_p` | `PLI_MODELE_MIN_P` | reel |  | `pli/cli_args.py:61` |  |  | keep tokens at least this share of the likeliest one's probability (default: the model's generation config, else 0: off) |
| `modele.name` | recommande | `--name`, `--modele.name` | `PLI_MODELE_NAME` | texte | vide | `pli/cli_args.py:39` |  |  | model id clients ask for (default: the model's name) |
| `modele.profile` | recommande | `--profile`, `--modele.profile` | `PLI_MODELE_PROFILE` | texte |  | `pli/cli_args.py:122` |  |  | machine profile of PLI_* settings (e.g. m1-ultra; also PLI_PROFILE); variables already in the environment win |
| `modele.profile_env` | recommande | `--modele.profile_env` | `PLI_PROFILE` or `PLI_MODELE_PROFILE_ENV` (`PLI_PROFIL`) | texte | vide | `pli/cli.py:87` |  |  | Machine profile to load, as --profile (for example m1-ultra). *[switches.md : autre]* |
| `modele.reasoning_effort` | recommande | `--reasoning-effort`, `--modele.reasoning_effort` | `PLI_MODELE_REASONING_EFFORT` | choix : low/medium/high/xhigh |  | `pli/cli_args.py:66` |  |  | default effort when a request omits one. An unnamed level maps to the nearest level the template names, and a tie takes the higher one. This flag uses that rule. xhigh stays xhigh, so GLM-5.3 renders it as Max |
| `modele.require_kernels` | experimental | `--modele.require_kernels` | `PLI_REQUIRE_KERNELS` or `PLI_MODELE_REQUIRE_KERNELS` (`TF_REQUIRE_KERNELS`) | drapeau |  | `pli/kernels/qwen/flash_next/v1/prefill_mm.py:262` |  |  |  |
| `modele.seed_salt` | experimental | `--modele.seed_salt` | `PLI_SEED_SALT` or `PLI_MODELE_SEED_SALT` (`TENSORFOLD_SEED_SALT`) | texte | vide | `pli/engine/exact_sampling.py:39` |  |  |  |
| `modele.temperature` | recommande | `--temperature`, `--modele.temperature` | `PLI_MODELE_TEMPERATURE` | reel |  | `pli/cli_args.py:57` |  |  | 0 decodes greedily (default: the model's generation_config.json, else 0) |
| `modele.thinking` | recommande | `--thinking`, `--no-thinking`, `--modele.thinking` | `PLI_MODELE_THINKING` | drapeau | `True` | `pli/cli_args.py:64` |  |  | open a think block when the chat template supports it |
| `modele.thinking_budget` | recommande | `--thinking-budget`, `--modele.thinking_budget` | `PLI_MODELE_THINKING_BUDGET` | entier | `0` | `pli/cli_args.py:70` |  |  | most thinking tokens before the server closes the think block (0: no limit) |
| `modele.tool_grammar` | recommande (?) | `--modele.tool_grammar` | `PLI_TOOL_GRAMMAR` or `PLI_MODELE_TOOL_GRAMMAR` | drapeau | `0` | `pli/engine/grammar.py:27` |  |  | 1: a request that offers tools, without response_format or guided_*, gets its model's tool-call grammar (xgrammar builtin structural tag, Qwen XML for qwen3_5 and qwen3_5_moe): free text, and in each call an offered tool's name then its parameters in the schema's order, values valid against their schema. Needs pli[grammar]. A tools grammar that does not compile leaves the request free (logged). *[switches.md : reglage-fin]* |
| `modele.top_k` | recommande | `--top-k`, `--modele.top_k` | `PLI_MODELE_TOP_K` | entier |  | `pli/cli_args.py:60` |  |  | (default: the model's generation config) |
| `modele.top_p` | recommande | `--top-p`, `--modele.top_p` | `PLI_MODELE_TOP_P` | reel |  | `pli/cli_args.py:59` |  |  | (default: the model's generation config) |
| `modele.video_tokens` | experimental | `--modele.video_tokens` | `PLI_VIDEO_TOKENS` or `PLI_MODELE_VIDEO_TOKENS` (`TENSORFOLD_VIDEO_TOKENS`) | entier |  | `pli/vision/qwen_processing.py:263` |  |  |  |
| `modele.vision` | recommande | `--vision`, `--modele.vision` | `PLI_MODELE_VISION` | drapeau | `False` | `pli/cli_args.py:44` |  |  | enable image input for supported GLM and Qwen vision checkpoints |
| `modele.vision_max_images` | recommande | `--vision-max-images`, `--modele.vision_max_images` | `PLI_MODELE_VISION_MAX_IMAGES` | entier |  | `pli/cli_args.py:48` |  |  | with --vision, maximum images across the full request history (default: 4); byte, pixel and visual-token limits still apply |
| `modele.vision_urls` | recommande | `--vision-urls`, `--modele.vision_urls` | `PLI_MODELE_VISION_URLS` | drapeau | `False` | `pli/cli_args.py:46` |  |  | with --vision, accept public HTTP(S) image URLs (default: data URLs only) |

### brouillons

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `brouillons.attn` | recommande | `--brouillons.attn` | `PLI_DRAFT_ATTN` or `PLI_BROUILLONS_ATTN` (`PLI_BROUILLON_ATTN`) | drapeau | `0` | `pli/additions/attention_draft.py:20` |  | m1-ultra/qwen3_5 `1` | Attention of the DFlash2 draft in a PLI kernel (sliding window, non-causal block of 1 to 8 rows, two passes), reading the cache and the block without concatenating them. Bits other than MLX's; the model verifies the drafts. *[switches.md : stable]* |
| `brouillons.copy_match` | experimental | `--brouillons.copy_match` | `PLI_COPY_MATCH` or `PLI_BROUILLONS_COPY_MATCH` (`TF_COPY_MATCH`) | entier | `8` | `pli/drafters/dflash_proposer.py:231` |  |  |  |
| `brouillons.copy_rows` | experimental | `--brouillons.copy_rows` | `PLI_COPY_ROWS` or `PLI_BROUILLONS_COPY_ROWS` (`TF_COPY_ROWS`) | texte | vide | `pli/families/qwen3_5/__init__.py:29` |  |  |  |
| `brouillons.drafter` | recommande | `--drafter`, `--brouillons.drafter` | `PLI_BROUILLONS_DRAFTER` | texte | `auto` | `pli/cli_args.py:76` |  |  | a draft model (repo id or directory); auto: the family's draft model when it has been pulled; none: no draft model |
| `brouillons.drafter_bits` | recommande | `--drafter-bits`, `--brouillons.drafter_bits` | `PLI_BROUILLONS_DRAFTER_BITS` | entier | `4` | `pli/cli_args.py:79` |  |  | quantize the draft model's linears (0: bf16) |
| `brouillons.extended_draft_list` | recommande | `--brouillons.extended_draft_list` | `PLI_EXTENDED_DRAFT_LIST` or `PLI_BROUILLONS_EXTENDED_DRAFT_LIST` (`PLI_LISTE_PLUS`) | texte | vide | `pli/additions/extended_list.py:19` |  | m1-ultra/qwen3_5 `extended-list-2048.txt` | File of token ids (one per line) added once, at load, to the reduced heads of DFlash2 and of the MTP head, so the drafts can propose them; a bare file name is looked up next to the package, which ships extended-list-2048.txt. No extra product per round; the verify stays exact (same text). *[switches.md : stable]* |
| `brouillons.falcon` | experimental | `--brouillons.falcon` | `PLI_FALCON` or `PLI_BROUILLONS_FALCON` | texte | vide | `pli/additions/falcon/capture.py:29` |  |  | Experimental. Directory where FALCON captures training examples for the MTP head (hidden state, chosen token, top logits). Off when unset. *[switches.md : machine]* |
| `brouillons.falcon_batch` | experimental | `--brouillons.falcon_batch` | `PLI_FALCON_BATCH` or `PLI_BROUILLONS_FALCON_BATCH` (`PLI_FALCON_LOT`) | entier | `4096` | `pli/additions/falcon/capture.py:31` |  |  | FALCON: rows per capture file. *[switches.md : experimental]* |
| `brouillons.falcon_context8` | experimental | `--brouillons.falcon_context8` | `PLI_FALCON_CONTEXT8` or `PLI_BROUILLONS_FALCON_CONTEXT8` (`PLI_FALCON_CONTEXTE8`) | drapeau | `1` | `pli/additions/falcon/capture.py:39` |  |  | FALCON: prompt rows captured in 8 bits per row (scale = max abs / 127) instead of 16. *[switches.md : experimental]* |
| `brouillons.falcon_head` | experimental | `--brouillons.falcon_head` | `PLI_FALCON_HEAD` or `PLI_BROUILLONS_FALCON_HEAD` (`PLI_FALCON_TETE`) | texte | vide | `pli/families/qwen3_5_moe/hybrid.py:76` |  |  | FALCON: JSON file naming a published MTP head; checked every 30 s, the head is swapped between two rounds. *[switches.md : experimental]* |
| `brouillons.falcon_k` | experimental | `--brouillons.falcon_k` | `PLI_FALCON_K` or `PLI_BROUILLONS_FALCON_K` | entier | `20` | `pli/additions/falcon/capture.py:30` |  |  | FALCON: top logits kept per row. *[switches.md : experimental]* |
| `brouillons.falcon_max_gb` | experimental | `--brouillons.falcon_max_gb` | `PLI_FALCON_MAX_GB` or `PLI_BROUILLONS_FALCON_MAX_GB` (`PLI_FALCON_MAX_GO`) | reel | `2` | `pli/additions/falcon/capture.py:32` |  |  | Experimental. Disk cap of the FALCON capture in GB; the oldest files go first. Code default 2. *[switches.md : machine]* |
| `brouillons.falcon_prompt` | experimental | `--brouillons.falcon_prompt` | `PLI_FALCON_PROMPT` or `PLI_BROUILLONS_FALCON_PROMPT` | entier | `2048` | `pli/additions/falcon/capture.py:36` |  |  | FALCON: last prompt rows kept per request. *[switches.md : experimental]* |
| `brouillons.falcon_stream` | experimental | `--brouillons.falcon_stream` | `PLI_FALCON_STREAM` or `PLI_BROUILLONS_FALCON_STREAM` (`PLI_FALCON_FLOT`) | drapeau | `1` | `pli/additions/falcon/capture.py:233` |  |  | FALCON: capture on its own GPU stream, so the next round's drafts do not wait for it. *[switches.md : experimental]* |
| `brouillons.falcon_targets_every` | experimental | `--brouillons.falcon_targets_every` | `PLI_FALCON_TARGETS_EVERY` or `PLI_BROUILLONS_FALCON_TARGETS_EVERY` (`PLI_FALCON_CIBLES_UN_SUR`) | entier | `2` | `pli/additions/falcon/capture.py:35` |  |  | FALCON: top-logit targets captured one round in N. *[switches.md : experimental]* |
| `brouillons.falcon_tokens` | recommande (?) | `--brouillons.falcon_tokens` | `PLI_FALCON_TOKENS` or `PLI_BROUILLONS_FALCON_TOKENS` | texte | vide | `pli/additions/falcon/tokens.py:27`<br>`pli/tasks/falcon.py:259` |  |  | FALCON B: directory where the prompt and reply tokens of each request are kept (one file per request, mode 700), for the night job that recomputes the 27B states and adjusts the DFlash2 draft. Off when unset. *[switches.md : experimental]* |
| `brouillons.falcon_tokens_days` | recommande (?) | `--brouillons.falcon_tokens_days` | `PLI_FALCON_TOKENS_DAYS` or `PLI_BROUILLONS_FALCON_TOKENS_DAYS` | reel | `7` | `pli/additions/falcon/tokens.py:28` |  |  | FALCON B: days a token file is kept; older files are deleted. *[switches.md : experimental]* |
| `brouillons.flash_draft_vocab` | experimental | `--brouillons.flash_draft_vocab` | `PLI_FLASH_DRAFT_VOCAB` or `PLI_BROUILLONS_FLASH_DRAFT_VOCAB` (`TF_FLASH_DRAFT_VOCAB`) | drapeau | `1` | `pli/families/qwen4_exp/runtime.py:61` |  |  |  |
| `brouillons.flash_mtp` | experimental | `--brouillons.flash_mtp` | `PLI_FLASH_MTP` or `PLI_BROUILLONS_FLASH_MTP` (`TF_FLASH_MTP`) | entier | `3` | `pli/families/qwen4_exp/runtime.py:436` |  |  |  |
| `brouillons.gpu_chain` | experimental | `--brouillons.gpu_chain` | `PLI_GPU_CHAIN` or `PLI_BROUILLONS_GPU_CHAIN` (`PLI_CHAINE_GPU`) | drapeau | `0` | `pli/families/qwen3_5/dflash_head.py:24` |  |  | DFlash2's greedy chain drawn on the GPU from its lattice; no gain measured. *[switches.md : experimental]* |
| `brouillons.head_before_copy` | recommande | `--brouillons.head_before_copy` | `PLI_HEAD_BEFORE_COPY` or `PLI_BROUILLONS_HEAD_BEFORE_COPY` (`PLI_TETE_AVANT_COPIE`) | texte | `0` | `pli/engine/lane_family.py:22` |  | m1-ultra/qwen3_5 `3` | Copy of the context or DFlash2's queued chain: 1: never a copy when the head has a chain queued; 2: the copy only if its first two tokens are those of the chain; 3: learned per stream, both scored afterwards on the written text, the copy goes first while the moving average of copy minus head stays above -PLI_HEAD_COPY_THRESHOLD. With the MTP head, the former behaviour. The model verifies every draft. *[switches.md : stable]* |
| `brouillons.head_copy_threshold` | experimental | `--brouillons.head_copy_threshold` | `PLI_HEAD_COPY_THRESHOLD` or `PLI_BROUILLONS_HEAD_COPY_THRESHOLD` (`PLI_TETE_COPIE_SEUIL`) | reel | `1` | `pli/engine/lane_family.py:34` |  |  | PLI_HEAD_BEFORE_COPY=3: the copy goes first while that average stays above minus N tokens. *[switches.md : reglage-fin]* |
| `brouillons.head_copy_weight` | experimental | `--brouillons.head_copy_weight` | `PLI_HEAD_COPY_WEIGHT` or `PLI_BROUILLONS_HEAD_COPY_WEIGHT` (`PLI_TETE_COPIE_POIDS`) | reel | `0.2` | `pli/engine/lane_family.py:33` |  |  | PLI_HEAD_BEFORE_COPY=3: weight of the moving average of copy minus head. *[switches.md : reglage-fin]* |
| `brouillons.hybrid` | recommande | `--brouillons.hybrid` | `PLI_HYBRID` or `PLI_BROUILLONS_HYBRID` (`PLI_HYBRIDE`) | drapeau | `0` | `pli/families/qwen3_5/__init__.py:169`<br>`pli/families/qwen3_5/__init__.py:182` |  |  | DFlash2 and the MTP head loaded together; each conversation drafts with the one that gives it more tokens per millisecond. Needs PLI_MTP_HEAD and --drafter. Stable, in production. *[switches.md : machine]* |
| `brouillons.hybrid_depth` | experimental | `--brouillons.hybrid_depth` | `PLI_HYBRID_DEPTH` or `PLI_BROUILLONS_HYBRID_DEPTH` (`PLI_HYBRIDE_PROFONDEUR`) | drapeau | `0` | `pli/families/qwen3_5_moe/hybrid.py:317` |  |  | In MTP mode, the hybrid picks its depth like the MTP head alone (acceptance by depth, cost of one step). *[switches.md : experimental]* |
| `brouillons.hybrid_force` | experimental | `--brouillons.hybrid_force` | `PLI_HYBRID_FORCE` or `PLI_BROUILLONS_HYBRID_FORCE` (`PLI_HYBRIDE_FORCE`) | texte | vide | `pli/families/qwen3_5_moe/hybrid.py:72` |  |  | dflash or mtp: freezes the hybrid's choice (benches). *[switches.md : experimental]* |
| `brouillons.hybrid_global_rounds` | experimental | `--brouillons.hybrid_global_rounds` | `PLI_HYBRID_GLOBAL_ROUNDS` or `PLI_BROUILLONS_HYBRID_GLOBAL_ROUNDS` (`PLI_HYBRIDE_GLOBAL_TOURS`) | entier | `12` | `pli/families/qwen3_5_moe/hybrid.py:37` |  |  | Rounds of each draft the engine measures before new conversations start on the better one. *[switches.md : reglage-fin]* |
| `brouillons.hybrid_im_start` | experimental | `--brouillons.hybrid_im_start` | `PLI_HYBRID_IM_START` or `PLI_BROUILLONS_HYBRID_IM_START` (`PLI_HYBRIDE_IM_START`) | entier | `248045` | `pli/families/qwen3_5_moe/hybrid.py:35` |  |  | Message start token counted by PLI_HYBRID_MESSAGES (Qwen3.8: < *[switches.md : reglage-fin]* |
| `brouillons.hybrid_messages` | recommande | `--brouillons.hybrid_messages` | `PLI_HYBRID_MESSAGES` or `PLI_BROUILLONS_HYBRID_MESSAGES` (`PLI_HYBRIDE_MESSAGES`) | entier | `0` | `pli/families/qwen3_5_moe/hybrid.py:34` |  | m1-ultra/qwen3_5 `3` | With PLI_HYBRID, a prompt of at most N messages (counted by PLI_HYBRID_IM_START, 248045) drafts with DFlash2 alone. *[switches.md : stable]* |
| `brouillons.lean` | experimental | `--brouillons.lean` | `PLI_DRAFT_LEAN` or `PLI_BROUILLONS_LEAN` (`TF_DRAFT_LEAN`) | texte | `1` | `pli/drafters/dflash_proposer.py:21` |  |  |  |
| `brouillons.mtp_drafts` | recommande | `--mtp-drafts`, `--brouillons.mtp_drafts` | `PLI_BROUILLONS_MTP_DRAFTS` | entier |  | `pli/cli_args.py:80` |  |  | most MTP drafts a round (Qwen3.8 Flash Next: 3; Qwen3.6 MoE: 4, each round's depth, plain included, from measured costs); 0: no MTP drafts |
| `brouillons.mtp_head` | recommande | `--brouillons.mtp_head` | `PLI_MTP_HEAD` or `PLI_BROUILLONS_MTP_HEAD` (`PLI_MTP27`) | texte | vide | `pli/families/qwen3_5/__init__.py:157` |  |  | Path of the 27B's native MTP head (mtp.* tensors, 4-bit safetensors). Alone, it replaces DFlash2; with PLI_HYBRID=1, both are loaded. *[switches.md : machine]* |
| `brouillons.mtp_head_drafts` | recommande | `--brouillons.mtp_head_drafts` | `PLI_MTP_DRAFTS` or `PLI_BROUILLONS_MTP_HEAD_DRAFTS` (`PLI_MTP27_BROUILLONS`) | entier | `(calculated: DRAFTS)` | `pli/families/qwen3_5/__init__.py:173`<br>`pli/families/qwen3_5/__init__.py:176` |  | m1-ultra/qwen3_5 `7` | Most MTP drafts per round when the 27B drafts with its MTP head (PLI_MTP_HEAD). *[switches.md : stable]* |
| `brouillons.mtp_head_window` | recommande | `--brouillons.mtp_head_window` | `PLI_MTP_WINDOW` or `PLI_BROUILLONS_MTP_HEAD_WINDOW` (`PLI_MTP27_FENETRE`) | entier | `0` | `pli/engine/family_prefill.py:60`<br>`pli/families/qwen3_5_moe/mtp.py:17` |  | m1-ultra/qwen3_5 `2048` | The MTP head keeps only its last N rows (sliding window), which bounds its memory. *[switches.md : stable]* |
| `brouillons.mtp_rounds` | experimental | `--brouillons.mtp_rounds` | `PLI_MTP_ROUNDS` or `PLI_BROUILLONS_MTP_ROUNDS` (`TF_MTP_ROUNDS`) | drapeau | `1` | `pli/families/nemotron_h/model.py:593` |  |  |  |
| `brouillons.nemotron_mtp` | experimental | `--brouillons.nemotron_mtp` | `PLI_NEMOTRON_MTP` or `PLI_BROUILLONS_NEMOTRON_MTP` (`TF_NEMOTRON_MTP`) | texte | vide | `pli/families/nemotron_h/model.py:592` |  |  |  |
| `brouillons.ngram_host` | experimental | `--brouillons.ngram_host` | `PLI_NGRAM_HOST` or `PLI_BROUILLONS_NGRAM_HOST` (`TF_NGRAM_HOST`) | texte | vide | `pli/families/qwen4_exp/host_table.py:25` |  |  |  |
| `brouillons.ngram_weight` | experimental | `--brouillons.ngram_weight` | `PLI_NGRAM_WEIGHT` or `PLI_BROUILLONS_NGRAM_WEIGHT` (`TF_NGRAM_WEIGHT`) | reel | `0.1` | `pli/drafters/dflash_proposer.py:237` |  |  |  |
| `brouillons.no_drafts` | recommande | `--no-drafts`, `--brouillons.no_drafts` | `PLI_BROUILLONS_NO_DRAFTS` | drapeau | `False` | `pli/cli_args.py:74` |  |  | one token a round: the serial reference (same output, slower) |
| `brouillons.qkr` | recommande | `--brouillons.qkr` | `PLI_DRAFT_QKR` or `PLI_BROUILLONS_QKR` (`PLI_BROUILLON_QKR`) | drapeau | `0` | `pli/additions/draft_qkr.py:22` |  | m1-ultra/qwen3_5 `1` | With PLI_DRAFT_STACK, the draft's q and k norms, RoPE and reordering (heads, rows) in one kernel per layer instead of about eight MLX operations; same computation as MLX 0.32.3, same bits. *[switches.md : stable]* |
| `brouillons.qwen36_draft_vocab` | experimental | `--brouillons.qwen36_draft_vocab` | `PLI_QWEN36_DRAFT_VOCAB` or `PLI_BROUILLONS_QWEN36_DRAFT_VOCAB` (`TF_QWEN36_DRAFT_VOCAB`) | drapeau | `1` | `pli/families/qwen3_5_moe/family.py:57` |  |  |  |
| `brouillons.qwen36_mtp` | experimental | `--brouillons.qwen36_mtp` | `PLI_QWEN36_MTP` or `PLI_BROUILLONS_QWEN36_MTP` (`TF_QWEN36_MTP`) | texte | vide | `pli/families/qwen3_5_moe/__init__.py:42` |  |  |  |
| `brouillons.recall` | recommande | `--brouillons.recall` | `PLI_RECALL` or `PLI_BROUILLONS_RECALL` (`PLI_RAPPEL`) | choix : 1/on/oui/yes | vide | `pli/additions/recall.py:143` |  | m1-ultra/qwen3_5 `1` | Recall drafts: continuations copied from the text of finished requests kept in memory; the model verifies them, same text. *[switches.md : stable]* |
| `brouillons.recall_tokens` | experimental | `--brouillons.recall_tokens` | `PLI_RECALL_TOKENS` or `PLI_BROUILLONS_RECALL_TOKENS` (`PLI_RAPPEL_JETONS`) | entier | `1000000` | `pli/additions/recall.py:147` |  |  | Size bound of the Recall registry in tokens; the oldest go first. *[switches.md : reglage-fin]* |
| `brouillons.stack` | recommande | `--brouillons.stack` | `PLI_DRAFT_STACK` or `PLI_BROUILLONS_STACK` (`PLI_BROUILLON_PILE`) | drapeau | `0` | `pli/additions/draft_stack.py:19` |  | m1-ultra/qwen3_5 `1` | DFlash2 draft projections stacked: [q \ *[switches.md : stable]* |
| `brouillons.topk` | recommande | `--brouillons.topk` | `PLI_DRAFT_TOPK` or `PLI_BROUILLONS_TOPK` (`PLI_BROUILLON_TOPK`) | drapeau | `0` | `pli/additions/topk_fast.py:20` |  | m1-ultra/qwen3_5 `1` | The k best candidates of each row of the draft's lattice in two passes (16 slices per row, then one simdgroup per row); same indices and values, in the same order, as the original kernel. *[switches.md : stable]* |
| `brouillons.vocab` | experimental | `--brouillons.vocab` | `PLI_DRAFT_VOCAB` or `PLI_BROUILLONS_VOCAB` (`TF_DRAFT_VOCAB`) | choix : full | vide | `pli/drafters/dflash_drafter.py:173`<br>`pli/drafters/dflash_drafter.py:196` |  |  |  |

### cache

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cache.checkpoint_bits` | recommande | `--cache.checkpoint_bits` | `PLI_CHECKPOINT_BITS` or `PLI_CACHE_CHECKPOINT_BITS` (`PLI_POINTS_BITS`) | entier | `0` | `pli/additions/checkpoint_bits.py:22` |  | m1-ultra/qwen3_5 `4` | Checkpoints evicted from memory or written to disk are kept in 4 bits (4: Hadamard rotation, groups of PLI_CHECKPOINT_GROUP) or 8 bits (8) and restored to 16 bits on resume; the checkpoint key carries the format, so a file written in another format is not reused (text can change, resumed turns only). *[switches.md : stable]* |
| `cache.checkpoint_group` | experimental | `--cache.checkpoint_group` | `PLI_CHECKPOINT_GROUP` or `PLI_CACHE_CHECKPOINT_GROUP` (`PLI_POINTS_GROUPE`) | entier | `64` | `pli/additions/checkpoint_bits.py:23` |  |  | PLI_CHECKPOINT_BITS: values per quantization group. *[switches.md : reglage-fin]* |
| `cache.checkpoint_slots` | recommande | `--checkpoint-slots`, `--cache.checkpoint_slots` | `PLI_CACHE_CHECKPOINT_SLOTS` | entier |  | `pli/cli_args.py:88` |  |  | cached conversation prefixes kept in memory (default: 3 per parallel lane, at least 8); with long conversations this, not --prompt-cache-gib, is usually the limit |
| `cache.code_key` | experimental | `--cache.code_key` | `PLI_CODE_KEY` or `PLI_CACHE_CODE_KEY` | drapeau | `1` | `pli/additions/code_key.py:24` |  |  | A fingerprint of PLI's own sources (pli/additions, the prompt path of pli/engine) enters the snapshot key, so a checkpoint computed by other code is not resumed; with it on, the release number leaves the key (a new release that does not change the code keeps its checkpoints; python -m pli.additions.rekey moves older checkpoints to the new key); 0 gives the previous key, release number included. *[switches.md : reglage-fin]* |
| `cache.context` | recommande | `--context`, `--cache.context` | `PLI_CACHE_CONTEXT` | entier |  | `pli/cli_args.py:53` |  |  | prompt plus reply window (default: model config; 0: remove the metadata cap) |
| `cache.disk_reply_end` | recommande (?) | `--cache.disk_reply_end` | `PLI_DISK_REPLY_END` or `PLI_CACHE_DISK_REPLY_END` (`PLI_DISQUE_FIN_REPONSE`) | drapeau | `0` | `pli/server/prompt_fill.py:25` |  |  | 1 with PLI_KEEP_REPLY: the disk lookup takes the checkpoint lengths memory takes, so a spilled reply-end checkpoint is read back instead of the prompt being read again. Same state, same text. *[switches.md : reglage-fin]* |
| `cache.fork_min` | recommande | `--cache.fork_min` | `PLI_FORK_MIN` or `PLI_CACHE_FORK_MIN` (`PLI_FOURCHE_MIN`) | entier | `4096` | `pli/server/checkpoints.py:49` |  |  | When a prompt forks from a kept one inside its last message (the same document sent again with another question), the state at the fork point is kept, past N tokens read since the resume; without a resume, the fork is looked for in the kept prompts. 0: off. Same text (a resume gives what a full prefill gives). *[switches.md : stable]* |
| `cache.kv_sharing` | recommande | `--cache.kv_sharing` | `PLI_KV_SHARING` or `PLI_CACHE_KV_SHARING` (`PLI_KV_PARTAGE`) | drapeau | `0` | `pli/additions/kv_sharing.py:23` |  | m1-ultra/qwen3_5 `1` | Checkpoints keep the conversation's attention layers and a length instead of a copy of the KV cache (same text). *[switches.md : stable]* |
| `cache.last_message_checkpoint` | recommande | `--cache.last_message_checkpoint` | `PLI_LAST_MESSAGE_CHECKPOINT` or `PLI_CACHE_LAST_MESSAGE_CHECKPOINT` (`PLI_DEBUT_DERNIER`) | entier | `0` | `pli/server/checkpoints.py:57` |  | m1-ultra/qwen3_5 `1024` | Also keeps the state at the start of the prompt's last message (the turn marker that opens the reply, as in ChatML, looked for backwards in the history) when at least N tokens are read there since the resume: the next request that keeps the history but not that message (a question replaced, a reply thrown away, a sub-agent on the same history) starts from there instead of from the previous request's checkpoint. 0: off. Same text (a resume gives what a full prefill gives). *[switches.md : stable]* |
| `cache.max_snapshots` | recommande | `--max-snapshots`, `--cache.max_snapshots` | `PLI_CACHE_MAX_SNAPSHOTS` | entier | `3` | `pli/cli_args.py:99` |  |  | system-block snapshots loaded at start |
| `cache.prompt_cache_gib` | recommande | `--prompt-cache-gib`, `--cache.prompt_cache_gib` | `PLI_CACHE_PROMPT_CACHE_GIB` | reel |  | `pli/cli_args.py:85` |  |  | memory for cached conversation prefixes (0: off; default on a Mac: what the weights, a whole-window request and a shared round leave idle, at least an eighth of RAM up to 16) |
| `cache.reread_disk` | recommande | `--cache.reread_disk` | `PLI_REREAD_DISK` or `PLI_CACHE_REREAD_DISK` (`PLI_RELIRE_DISQUE`) | drapeau | `0` | `pli/server/prompt_fill.py:20` |  | m1-ultra/qwen3_5 `1` | Looks at the disk again at a request's first chunk: a checkpoint evicted (spilled to disk) since the request opened is read back instead of reading the whole prompt again. *[switches.md : stable]* |
| `cache.snapshot_dir` | recommande | `--snapshot-dir`, `--cache.snapshot_dir` | `PLI_CACHE_SNAPSHOT_DIR` | texte | `~/.cache/tensorfold/prefix-snapshots` | `pli/cli_args.py:94` |  |  | where system-block and conversation snapshots are kept ('none': in memory only) |
| `cache.snapshot_settings` | experimental | `--cache.snapshot_settings` | `PLI_SNAPSHOT_SETTINGS` or `PLI_CACHE_SNAPSHOT_SETTINGS` (`PLI_INSTANTANE_REGLAGES`) | drapeau | `1` | `pli/cli.py:448` |  |  | A fingerprint of the PLI_* settings enters the snapshot key, so a conversation saved under other settings is not resumed. *[switches.md : reglage-fin]* |
| `cache.spill_background` | recommande (?) | `--cache.spill_background` | `PLI_SPILL_BACKGROUND` or `PLI_CACHE_SPILL_BACKGROUND` | drapeau | `0` | `pli/additions/spill_writer.py:35` |  |  | 1: an evicted conversation is prepared on the engine thread (reduction, export, evaluated arrays) and written to disk by a thread of its own; the engine waits only for a write a prompt could resume, past PLI_SPILL_BACKGROUND_MAX_GIB of bytes waiting, or at shutdown. Same file, same text. *[switches.md : reglage-fin]* |
| `cache.spill_background_max_gib` | experimental | `--cache.spill_background_max_gib` | `PLI_SPILL_BACKGROUND_MAX_GIB` or `PLI_CACHE_SPILL_BACKGROUND_MAX_GIB` | reel | `8` | `pli/additions/spill_writer.py:36` |  |  | PLI_SPILL_BACKGROUND: bytes of spills held in memory while they wait to be written. *[switches.md : reglage-fin]* |
| `cache.spill_gib` | recommande | `--spill-gib`, `--cache.spill_gib` | `PLI_CACHE_SPILL_GIB` | reel | `0.0` | `pli/cli_args.py:91` |  |  | write evicted conversation prefixes to disk, up to this many GiB, and read them back on demand instead of prefilling again (0: off; needs --snapshot-dir) |
| `cache.spill_prompt` | recommande | `--cache.spill_prompt` | `PLI_SPILL_PROMPT` or `PLI_CACHE_SPILL_PROMPT` (`PLI_DEVERSE_INVITE`) | drapeau | `1` | `pli/server/checkpoints.py:50` |  |  | An evicted prompt checkpoint goes to disk even when a reply end extends it (a reply end no longer counts as a continuation; a longer prompt still does); 0: the former behaviour. *[switches.md : stable]* |
| `cache.startup_cache` | recommande | `--startup-cache`, `--cache.startup_cache` | `PLI_CACHE_STARTUP_CACHE` | choix : off/on/remeasure | `off` | `pli/cli_args.py:96` |  |  | keep the startup measurements (window check, memory probes) on disk and read them back at the next start with the same weights, code and settings ('remeasure': measure and rewrite) |

### memoire

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `memoire.heartbeat_ms` | recommande | `--memoire.heartbeat_ms` | `PLI_HEARTBEAT_MS` or `PLI_MEMOIRE_HEARTBEAT_MS` (`PLI_BATTEMENT_MS`) | reel | `0` | `pli/server/scheduler.py:29` |  | m1-ultra/qwen3_5 `500` | Heartbeat between requests: a tiny GPU job every N ms keeps the wired weights resident (Metal releases them after about 2 s idle). 0 = off. *[switches.md : stable]* |
| `memoire.heartbeat_window_s` | experimental | `--memoire.heartbeat_window_s` | `PLI_HEARTBEAT_WINDOW_S` or `PLI_MEMOIRE_HEARTBEAT_WINDOW_S` (`PLI_BATTEMENT_FENETRE_S`) | reel | `600` | `pli/server/scheduler.py:30` |  |  | The heartbeat runs for this many seconds after the last activity. *[switches.md : reglage-fin]* |
| `memoire.memory_limit_gb` | experimental | `--memoire.memory_limit_gb` | `PLI_MEMORY_LIMIT_GB` or `PLI_MEMOIRE_MEMORY_LIMIT_GB` (`TENSORFOLD_MEMORY_LIMIT_GB`) | texte |  | `pli/server/memory_budget.py:76` |  |  |  |
| `memoire.mlx_cache_gib` | recommande | `--mlx-cache-gib`, `--memoire.mlx_cache_gib` | `PLI_MEMOIRE_MLX_CACHE_GIB` | reel | `8.0` | `pli/cli_args.py:110` |  |  | MLX's cache of freed buffers |
| `memoire.mlx_max_mb_per_buffer` | recommande | `--memoire.mlx_max_mb_per_buffer` | `MLX_MAX_MB_PER_BUFFER` or `PLI_MEMOIRE_MLX_MAX_MB_PER_BUFFER` | entier |  | `pli/families/deepseek_v4/__init__.py:24`<br>`pli/families/glm5_next/__init__.py:25` (+1) |  |  | Most MB of an MLX command buffer. MLX reads it when it starts, before any profile is applied, so it goes in the server's environment; pli setup sets it with --profile m1-ultra. Stable, in production. *[switches.md : machine]* **To check : never read directly in src/ (read by MLX).** |
| `memoire.mlx_max_ops_per_buffer` | recommande | `--memoire.mlx_max_ops_per_buffer` | `MLX_MAX_OPS_PER_BUFFER` or `PLI_MEMOIRE_MLX_MAX_OPS_PER_BUFFER` | entier |  | `pli/families/deepseek_v4/__init__.py:24`<br>`pli/families/glm5_next/__init__.py:25` (+1) |  |  | Most operations of an MLX command buffer; in the environment for the same reason. Stable, in production. *[switches.md : machine]* **To check : never read directly in src/ (read by MLX).** |
| `memoire.pass_cache_gib` | recommande | `--pass-cache-gib`, `--memoire.pass_cache_gib` | `PLI_MEMOIRE_PASS_CACHE_GIB` | reel | `16.0` | `pli/cli_args.py:108` |  |  | Mac: MLX's cache of freed buffers during such a pass, where the memory budget has room (at most --mlx-cache-gib: no change) |
| `memoire.ple_on_ssd` | recommande | `--ple-on-ssd`, `--memoire.ple_on_ssd` | `PLI_MEMOIRE_PLE_ON_SSD` | drapeau | `False` | `pli/cli_args.py:114` |  |  | Flash Next: read the n-gram (PLE) tables from the checkpoint on SSD at each lookup instead of holding them in memory. A trade: a few percent of decode speed for about 40 GiB less at peak (the tables are 29.8 GiB); a 128 GB Mac needs it |
| `memoire.ssd_experts` | recommande | `--ssd-experts`, `--memoire.ssd_experts` | `PLI_MEMOIRE_SSD_EXPERTS` | reel |  | `pli/cli_args.py:111` |  |  | stream routed experts from the checkpoint into a GPU pool of this many GiB, for models past the memory budget (the rest stays resident; output is the resident model's) |

### parallele

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `parallele.admit_by_work` | recommande (?) | `--parallele.admit_by_work` | `PLI_ADMIT_BY_WORK` or `PLI_PARALLELE_ADMIT_BY_WORK` (`PLI_ADMISSION_TRAVAIL`) | drapeau | `0` | `pli/server/scheduler.py:211` |  |  | Admission counts the prompt-reading work measured by PromptMemory instead of the prefill_bytes line. *[switches.md : experimental]* |
| `parallele.admit_small` | recommande | `--parallele.admit_small` | `PLI_ADMIT_SMALL` or `PLI_PARALLELE_ADMIT_SMALL` (`PLI_ADMISSION_PETITES`) | entier | `0` | `pli/server/scheduler.py:209` |  | m1-ultra/qwen3_5 `8000` | A request of at most N tokens is admitted ahead of a request held back at admission (small requests first). *[switches.md : stable]* |
| `parallele.arbiter` | recommande | `--parallele.arbiter` | `PLI_ARBITER` or `PLI_PARALLELE_ARBITER` (`PLI_ARBITRE`) | drapeau | `0` | `pli/additions/arbiter.py:29` |  | m1-ultra/common `1` | Cost arbiter: learns the real cost of a round online and picks the copy width and the number of drafts; never changes tokens. *[switches.md : stable]* |
| `parallele.arbiter_min_rows` | experimental | `--parallele.arbiter_min_rows` | `PLI_ARBITER_MIN_ROWS` or `PLI_PARALLELE_ARBITER_MIN_ROWS` (`PLI_ARBITRE_PALIER`) | entier | `0` | `pli/additions/arbiter.py:36` |  |  | A request decoding alone keeps at least N - 1 drafts (N rows) when the draft offers them. *[switches.md : experimental]* |
| `parallele.arbiter_table` | experimental | `--parallele.arbiter_table` | `PLI_ARBITER_TABLE` or `PLI_PARALLELE_ARBITER_TABLE` (`PLI_ARBITRE_TABLE`) | drapeau | `0` | `pli/additions/arbiter.py:31` |  |  | Draft-count costs taken from a table observed in service instead of the arbiter's formula. *[switches.md : experimental]* |
| `parallele.decode_share` | recommande | `--decode-share`, `--parallele.decode_share` | `PLI_PARALLELE_DECODE_SHARE` | reel |  | `pli/cli_args.py:103` |  |  | Mac: while prompts prefill, running replies keep moving for this share of each chunk's time, and a new prompt starts at the next chunk (default 0.25; 0: whole prompts first, in order, as 0.3.6.2) |
| `parallele.flash_queued` | experimental | `--parallele.flash_queued` | `PLI_FLASH_QUEUED` or `PLI_PARALLELE_FLASH_QUEUED` (`TF_FLASH_QUEUED`) | drapeau | `1` | `pli/families/qwen4_exp/runtime.py:67` |  |  |  |
| `parallele.lane_tile` | experimental | `--parallele.lane_tile` | `PLI_LANE_TILE` or `PLI_PARALLELE_LANE_TILE` (`TF_LANE_TILE`) | drapeau | `1` | `pli/families/qwen3_5/__init__.py:310`<br>`pli/families/qwen3_5/__init__.py:254` |  |  |  |
| `parallele.parallel` | recommande | `--parallel`, `--parallele.parallel` | `PLI_PARALLELE_PARALLEL` | texte | `auto` | `pli/cli_args.py:100` |  |  | requests decoded together, their windows sharing each round's forward: a number, or auto (up to 8, each started only while the projected memory fits the budget) |

### lecture

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lecture.chunk_layers` | recommande | `--lecture.chunk_layers` | `PLI_CHUNK_LAYERS` or `PLI_LECTURE_CHUNK_LAYERS` (`PLI_MORCEAU_COUCHES`) | entier | `0` | `pli/engine/family_prefill.py:46` |  | m1-ultra/common `4` | While another request decodes, the reading of a prompt chunk yields every N layers (crossed split); same operations, same order. *[switches.md : stable]* |
| `lecture.chunk_tokens` | experimental | `--lecture.chunk_tokens` | `PLI_CHUNK_TOKENS` or `PLI_LECTURE_CHUNK_TOKENS` (`PLI_MORCEAU_JETONS`) | entier | `0` | `pli/engine/family_prefill.py:31` |  |  | While another request decodes, a prompt chunk is read in passes of at most N tokens. *[switches.md : experimental]* |
| `lecture.cpu` | experimental | `--lecture.cpu` | `PLI_PREFILL_CPU` (`PLI_LECTURE_CPU`) | reel | `0` | `pli/additions/relayout.py:114`<br>`pli/families/qwen3_5/prefill_cpu.py:33` |  |  | Share of prompt rows (0 to 0.9) computed on the CPU in 32 bits (text can change). *[switches.md : experimental]* |
| `lecture.cpu_min` | experimental | `--lecture.cpu_min` | `PLI_PREFILL_CPU_MIN` (`PLI_LECTURE_CPU_MIN`) | entier | `256` | `pli/families/qwen3_5/prefill_cpu.py:70` |  |  | Chunk rows below which nothing goes to the CPU. *[switches.md : experimental]* |
| `lecture.cpu_mode` | experimental | `--lecture.cpu_mode` | `PLI_PREFILL_CPU_MODE` (`PLI_LECTURE_CPU_MODE`) | texte | `shifted` | `pli/families/qwen3_5/prefill_cpu.py:60` |  |  | decale or couche: how CPU and GPU rows are scheduled. *[switches.md : experimental]* |
| `lecture.cpu_priority` | experimental | `--lecture.cpu_priority` | `PLI_PREFILL_CPU_PRIORITY` or `PLI_LECTURE_CPU_PRIORITY` (`PLI_LECTURE_CPU_PRIORITE`) | entier | `1` | `pli/families/qwen3_5/prefill_cpu.py:76` |  |  | Priority of the CPU work: 0 high, 1 medium, 2 background. *[switches.md : experimental]* |
| `lecture.flash_prefill` | experimental | `--lecture.flash_prefill` | `PLI_FLASH_PREFILL` or `PLI_LECTURE_FLASH_PREFILL` (`TF_FLASH_PREFILL`) | drapeau | `1` | `pli/kernels/qwen/flash_next/v1/prefill_mm.py:188` |  |  |  |
| `lecture.interleave_dense` | recommande | `--lecture.interleave_dense` | `PLI_INTERLEAVE_DENSE` or `PLI_LECTURE_INTERLEAVE_DENSE` (`PLI_CROISE_DENSE`) | drapeau | `0` | `pli/families/qwen3_5/family.py:163` |  | m1-ultra/qwen3_5 `1` | Allows the crossed split of prompt reading on the dense model (see PLI_CHUNK_LAYERS). *[switches.md : stable]* |
| `lecture.merge_tail` | recommande | `--lecture.merge_tail` | `PLI_MERGE_TAIL` or `PLI_LECTURE_MERGE_TAIL` (`PLI_FONDRE_FIN`) | entier | `0` | `pli/additions/merge_tail.py:25` |  | m1-ultra/qwen3_5 `16` | When the last checkpoint is at most N tokens before the end of the prompt, those tokens are read in the same pass as the chunk before (text can change). *[switches.md : stable]* |
| `lecture.mlp_int` | experimental | `--lecture.mlp_int` | `PLI_MLP_INT` or `PLI_LECTURE_MLP_INT` (`PLI_MLP_ENT`) | drapeau | `0` | `pli/additions/relayout.py:118` |  |  | MLP path in integers (reads the original weight layout, so PLI_RELAYOUT is not applied). *[switches.md : experimental]* |
| `lecture.prefill_pass` | recommande | `--prefill-pass`, `--lecture.prefill_pass` | `PLI_LECTURE_PREFILL_PASS` | entier | `8` | `pli/cli_args.py:106` |  |  | Mac: prompt chunks one forward takes while a prompt fills alone, for models with a prompt pass (1: one chunk a forward, as 0.5.0) |
| `lecture.rounds_between_prefills` | recommande | `--lecture.rounds_between_prefills` | `PLI_ROUNDS_BETWEEN_PREFILLS` or `PLI_LECTURE_ROUNDS_BETWEEN_PREFILLS` (`PLI_TOURS_ENTRE_LECTURES`) | entier | `32` | `pli/server/scheduler.py:196` |  | m1-ultra/common `100000` | Most decode rounds between two prompt chunks (TensorFold: 32). *[switches.md : stable]* |
| `lecture.unpack` | recommande | `--lecture.unpack` | `PLI_PREFILL_UNPACK` or `PLI_LECTURE_UNPACK` (`PLI_LECTURE_DEPLIE`) | texte | `0` | `pli/additions/unpack.py:45` |  | m1-ultra/qwen3_5 `f16` | 4-bit weights unpacked before the product in large prompt-reading calls (f16 for projections fed by the normed state, bf16 for the others). *[switches.md : stable]* |
| `lecture.unpack_check` | experimental | `--lecture.unpack_check` | `PLI_PREFILL_UNPACK_CHECK` or `PLI_LECTURE_UNPACK_CHECK` (`PLI_LECTURE_DEPLIE_CONTROLE`) | drapeau | `0` | `pli/additions/unpack.py:47` |  |  | Counts infinite or NaN f16 outputs (forces an evaluation: tests only). *[switches.md : experimental]* |
| `lecture.unpack_kernel` | recommande | `--lecture.unpack_kernel` | `PLI_PREFILL_UNPACK_KERNEL` or `PLI_LECTURE_UNPACK_KERNEL` (`PLI_LECTURE_DEPLIE_NOYAU`) | drapeau | `0` | `pli/additions/unpack.py:49` |  | m1-ultra/qwen3_5 `1` | One-pass Metal kernel instead of mx.dequantize for the unpacking; same values. *[switches.md : stable]* |
| `lecture.unpack_mlp` | recommande | `--lecture.unpack_mlp` | `PLI_PREFILL_UNPACK_MLP` or `PLI_LECTURE_UNPACK_MLP` (`PLI_LECTURE_DEPLIE_MLP`) | choix : t | `0` | `pli/additions/unpack.py:48` |  | m1-ultra/qwen3_5 `t` | With f16 unpacking, the dense MLP computed transposed; same bits as the f16 path. *[switches.md : stable]* |
| `lecture.unpack_nn` | recommande | `--lecture.unpack_nn` | `PLI_PREFILL_UNPACK_NN` or `PLI_LECTURE_UNPACK_NN` (`PLI_LECTURE_DEPLIE_NN`) | drapeau | `0` | `pli/additions/unpack.py:50` |  | m1-ultra/qwen3_5 `1` | Unpacked bf16 projections (out_proj, o_proj) and the draft's fc, k_proj, v_proj computed in nn form; same bits. *[switches.md : stable]* |
| `lecture.unpack_rows` | experimental | `--lecture.unpack_rows` | `PLI_PREFILL_UNPACK_ROWS` or `PLI_LECTURE_UNPACK_ROWS` (`PLI_LECTURE_DEPLIE_LIGNES`) | entier | `512` | `pli/additions/unpack.py:46` |  |  | Rows from which prompt-reading calls unpack the weights. *[switches.md : reglage-fin]* |
| `lecture.unpack_stack` | experimental | `--lecture.unpack_stack` | `PLI_PREFILL_UNPACK_STACK` or `PLI_LECTURE_UNPACK_STACK` (`PLI_LECTURE_DEPLIE_PILE`) | drapeau | `0` | `pli/additions/unpack.py:51` |  |  | Stacked projections (in_proj_qkv + in_proj_z, q + k + v) in one nn product. *[switches.md : experimental]* |

### attention

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `attention.a1_lsa_map` | recommande | `--attention.a1_lsa_map` | `PLI_A1_LSA_MAP` or `PLI_ATTENTION_A1_LSA_MAP` (`PLI_ANCRE1_LSA`) | texte | vide | `pli/additions/a1_lsa.py:45` |  |  | A1: distilled map of [Latent Sparse Attention](https://github.com/Prefrontal-AI/latent-sparse-attention) (<layer>.W_Q_idx, <layer>.W_K_idx, release d-lsa-weights-v1, file qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt, downloaded and converted by pli setup, or by tools/convert_lsa_map.py). Past PLI_S3B_THRESHOLD, each row of a chunk reads the 10% best scored tiles of 32 keys plus the last 1,024 positions, on the layers the map covers (text can change). Stable with PLI_A1_PREFILL_ONLY=1, PLI_S3B_THRESHOLD=16384 and PLI_S3B_ROWS=4096, all three in the environment (pli setup writes them in the service). *[switches.md : machine]* |
| `attention.a1_max_tiles` | experimental | `--attention.a1_max_tiles` | `PLI_A1_MAX_TILES` or `PLI_ATTENTION_A1_MAX_TILES` (`PLI_A1_GARDES_MAX`) | entier | `0` | `pli/additions/a1_lsa.py:116` |  |  | With PLI_A1_LSA_MAP, most tiles of 32 keys a row reads (0 = no cap). Faster at long context (first token at 146k -22 to -27% instead of -15.6%), but the NLL rises (+0.0046 at 160 tiles, 97k). *[switches.md : experimental]* |
| `attention.a1_prefill_only` | recommande | `--attention.a1_prefill_only` | `PLI_A1_PREFILL_ONLY` or `PLI_ATTENTION_A1_PREFILL_ONLY` (`PLI_ANCRE1_LECTURE_SEULE`) | drapeau | `0` | `pli/additions/a1_lsa.py:49` |  | m1-ultra/qwen3_5 `1` | With PLI_A1_LSA_MAP, A1 reads prompt chunks only; decode and trees keep the attention of production. Without the map, does nothing. *[switches.md : stable]* |
| `attention.a1_slice` | experimental | `--attention.a1_slice` | `PLI_A1_SLICE` or `PLI_ATTENTION_A1_SLICE` (`PLI_A1_TRANCHE`) | entier | `2048` | `pli/additions/a1_lsa.py:61` |  |  | A1: keys scored per kernel call. *[switches.md : reglage-fin]* |
| `attention.a1_small_slice` | experimental | `--attention.a1_small_slice` | `PLI_A1_SMALL_SLICE` or `PLI_ATTENTION_A1_SMALL_SLICE` (`PLI_A1_PETITE`) | entier | `128` | `pli/additions/a1_lsa.py:62` |  |  | A1: rows below which a chunk is scored in one call. *[switches.md : reglage-fin]* |
| `attention.attec` | recommande (?) | `--attention.attec` | `PLI_ATTENTION_ATTEC` | drapeau | `0` | `pli/kernels/qwen/dense/v1/exact_attention.py:16` |  | m1-ultra/qwen3_5 `1` | Decode attention for 2 to 16 rows (1,100 keys and more) in one launch at equal bits; on the 27B only a fallback since attm. *[switches.md : stable]* |
| `attention.attm` | recommande | `--attention.attm` | `PLI_ATTENTION_ATTM` | drapeau | `0` | `pli/kernels/qwen/dense/v1/exact_attention.py:19` |  | m1-ultra/qwen3_5 `1` | PLI's own decode attention (attm, 8 x 8 simdgroup tiles) for every decode row of the 27B, 1 to 32 rows. *[switches.md : stable]* |
| `attention.attm_load` | recommande | `--attention.attm_load` | `PLI_ATTM_LOAD` or `PLI_ATTENTION_ATTM_LOAD` (`PLI_ATTM_CHARGE`) | drapeau | `0` | `pli/additions/attention_attm.py:48` |  | m1-ultra/qwen3_5 `1` | attm lays out the block's keys and values without permutation (stride of 264 half-words) and loads the 8 x 8 fragments with simdgroup_load; same products in the same order: same bits. *[switches.md : stable]* |
| `attention.attm_short` | recommande | `--attention.attm_short` | `PLI_ATTM_SHORT` or `PLI_ATTENTION_ATTM_SHORT` (`PLI_ATTM_COURT`) | entier | `0` | `pli/additions/attention_attm.py:41` |  | m1-ultra/qwen3_5 `12288` | attm works in 128-key slices for calls of at most N keys (512-key slices above), so short contexts keep the GPU busy. *[switches.md : stable]* |
| `attention.attm_small_keys` | experimental | `--attention.attm_small_keys` | `PLI_ATTM_SMALL_KEYS` or `PLI_ATTENTION_ATTM_SMALL_KEYS` (`PLI_ATTM_PETITES`) | entier | `0` | `pli/additions/attention_attm.py:36` |  |  | attm stores the first N keys in 128-key slices. *[switches.md : experimental]* |
| `attention.attn_direct_p` | experimental | `--attention.attn_direct_p` | `PLI_ATTN_DIRECT_P` or `PLI_ATTENTION_ATTN_DIRECT_P` (`TF_ATTN_DIRECT_P`) | drapeau | `1` | `pli/kernels/qwen/dense/v1/lane_attention.py:128` |  |  |  |
| `attention.attn_ns` | experimental | `--attention.attn_ns` | `PLI_ATTN_NS` or `PLI_ATTENTION_ATTN_NS` | entier | `4` | `pli/additions/attention_m1.py:37` |  |  | PLI_ATTENTION=m1: simdgroups per 8-query tile. *[switches.md : experimental]* |
| `attention.attn_rows` | experimental | `--attention.attn_rows` | `PLI_ATTN_ROWS` or `PLI_ATTENTION_ATTN_ROWS` | entier | `16` | `pli/additions/attention_gemm.py:27` |  |  | PLI_ATTENTION=gemm: rows per pass. *[switches.md : experimental]* |
| `attention.attn_rt` | experimental | `--attention.attn_rt` | `PLI_ATTN_RT` or `PLI_ATTENTION_ATTN_RT` | entier | `1` | `pli/additions/attention_m1.py:39` |  |  | PLI_ATTENTION=m1: 8-query tiles per simdgroup. *[switches.md : experimental]* |
| `attention.attn_slices` | experimental | `--attention.attn_slices` | `PLI_ATTN_SLICES` or `PLI_ATTENTION_ATTN_SLICES` (`PLI_ATTN_MORCEAUX`) | texte | `512x0` | `pli/additions/attention_m1.py:31` |  |  | PLI_ATTENTION=m1: key slices as size x count,.... *[switches.md : experimental]* |
| `attention.attn_tiles` | experimental | `--attention.attn_tiles` | `PLI_ATTN_TILES` or `PLI_ATTENTION_ATTN_TILES` (`TF_ATTN_TILES`) | entier | `16` | `pli/kernels/qwen/dense/v1/lane_attention.py:11` |  |  |  |
| `attention.attn_tk` | experimental | `--attention.attn_tk` | `PLI_ATTN_TK` or `PLI_ATTENTION_ATTN_TK` | entier | `64` | `pli/additions/attention_m1.py:33` |  |  | PLI_ATTENTION=m1: keys per softmax step. *[switches.md : experimental]* |
| `attention.f16` | recommande | `--attention.f16` | `PLI_ATTENTION_F16` | drapeau | `0` | `pli/kernels/qwen/dense/v1/prompt_attention.py:33` |  | m1-ultra/qwen3_5 `1` | Prompt-reading attention of the 27B with its products in f16 (bits change). *[switches.md : stable]* |
| `attention.f16_heads` | experimental | `--attention.f16_heads` | `PLI_ATTENTION_F16_HEADS` (`PLI_ATTENTION_F16_HC`) | entier | `2` | `pli/kernels/qwen/dense/v1/prompt_attention_f16.py:27` |  |  | Key heads converted to f16 per group in the f16 prompt attention. *[switches.md : reglage-fin]* |
| `attention.f16_min` | experimental | `--attention.f16_min` | `PLI_ATTENTION_F16_MIN` | entier | `16384` | `pli/kernels/qwen/dense/v1/prompt_attention_f16.py:30` |  |  | Key threshold of the f16 prompt attention path. *[switches.md : reglage-fin]* |
| `attention.f16_pv` | experimental | `--attention.f16_pv` | `PLI_ATTENTION_F16_PV` | texte | `32,64,32,2,2` | `pli/kernels/qwen/dense/v1/prompt_attention_f16.py:29` |  |  | Tile (BM, BN, BK, WM, WN) of the P.V product, f16 path. *[switches.md : reglage-fin]* |
| `attention.f16_qk` | experimental | `--attention.f16_qk` | `PLI_ATTENTION_F16_QK` | texte | `64,64,16,2,2` | `pli/kernels/qwen/dense/v1/prompt_attention_f16.py:28` |  |  | Tile of the Q.Kt product, f16 path. *[switches.md : reglage-fin]* |
| `attention.gemm` | recommande | `--attention.gemm` | `PLI_ATTENTION_GEMM` | drapeau | `0` | `pli/kernels/qwen/dense/v1/prompt_attention.py:30` |  | m1-ultra/qwen3_5 `1` | Prompt-reading attention products through MLX's steel_gemm_fused kernel with tiles measured on M1 Ultra; same bits as MLX's unfused path. *[switches.md : stable]* |
| `attention.gemm_min` | experimental | `--attention.gemm_min` | `PLI_ATTENTION_GEMM_MIN` | entier | `16384` | `pli/kernels/qwen/dense/v1/prompt_attention_gemm.py:21` |  |  | Key threshold of the tuned GEMM prompt attention path. *[switches.md : reglage-fin]* |
| `attention.gemm_pv` | experimental | `--attention.gemm_pv` | `PLI_ATTENTION_GEMM_PV` | texte | `32,32,32,4,1` | `pli/kernels/qwen/dense/v1/prompt_attention_gemm.py:17` |  |  | Tile of the P.V product up to PLI_ATTENTION_GEMM_PV_THRESHOLD visible keys. *[switches.md : reglage-fin]* |
| `attention.gemm_pv_long` | experimental | `--attention.gemm_pv_long` | `PLI_ATTENTION_GEMM_PV_LONG` (`PLI_ATTENTION_GEMM_PVLONG`) | texte | `16,32,64,2,2` | `pli/kernels/qwen/dense/v1/prompt_attention_gemm.py:18` |  |  | Tile of the P.V product beyond that threshold. *[switches.md : reglage-fin]* |
| `attention.gemm_pv_threshold` | experimental | `--attention.gemm_pv_threshold` | `PLI_ATTENTION_GEMM_PV_THRESHOLD` (`PLI_ATTENTION_GEMM_PVSEUIL`) | entier | `65536` | `pli/kernels/qwen/dense/v1/prompt_attention_gemm.py:19` |  |  | Visible keys where the P.V tile changes. *[switches.md : reglage-fin]* |
| `attention.gemm_qk` | experimental | `--attention.gemm_qk` | `PLI_ATTENTION_GEMM_QK` | texte | `32,64,16,1,2` | `pli/kernels/qwen/dense/v1/prompt_attention_gemm.py:20` |  |  | Tile of the Q.Kt product, GEMM path. *[switches.md : reglage-fin]* |
| `attention.grouped` | experimental | `--attention.grouped` | `PLI_ATTENTION_GROUPED` (`PLI_ATTENTION_GROUPEE`) | drapeau | `0` | `pli/kernels/qwen/dense/v1/prompt_attention.py:13` |  |  | Prompt reading: query heads sharing a key head in one call with an explicit causal mask (more than 16 rows). *[switches.md : experimental]* |
| `attention.mode` | experimental | `--attention.mode` | `PLI_ATTENTION` or `PLI_ATTENTION_MODE` | texte | vide | `pli/kernels/qwen/dense/v1/exact_attention.py:40` |  |  | gemm or m1: windows of 1 to 128 rows through a PLI attention where a row has the same bits alone or in a window. *[switches.md : experimental]* |
| `attention.per_row` | recommande | `--attention.per_row` | `PLI_ATTENTION_PER_ROW` (`PLI_ATTENTION_LIGNE`) | entier | `0` | `pli/kernels/qwen/dense/v1/exact_attention.py:22`<br>`pli/kernels/qwen/dense/v1/exact_attention.py:26` |  | m1-ultra/qwen3_5 `15000`; m1-ultra/qwen3_5_moe `15000` | From N context keys, one MLX attention call per row instead of grouped calls; same bits. *[switches.md : stable]* *Note : threshold N: one call per line once N context keys; 1 = from the first key, so always (the attention.per_row_always flag says the same thing).* |
| `attention.per_row_always` | recommande | `--attention.per_row_always` | `PLI_ATTENTION_PER_ROW_ALWAYS` (pose `PLI_ATTENTION_PER_ROW=1`) | drapeau | `0` | `pli/config/reglages.py (traduit vers PLI_ATTENTION_PER_ROW)` |  |  | One MLX attention call per row from the first key (PLI_ATTENTION_PER_ROW=1, said as a flag); refused together with another PLI_ATTENTION_PER_ROW value. Same bits. |
| `attention.reduced` | recommande | `--attention.reduced` | `PLI_ATTENTION_REDUCED` (`PLI_ATTENTION_REDUITE`) | entier | `0` | `pli/kernels/qwen/dense/v1/exact_attention.py:59` |  | m1-ultra/qwen3_5 `10` | Reduced decode attention beyond PLI_ATTENTION_REDUCED_THRESHOLD keys: reads the N% of 32-key pages with the highest score plus a 1,024-token tail (text can change). *[switches.md : stable]* |
| `attention.reduced_anchor` | recommande | `--attention.reduced_anchor` | `PLI_REDUCED_ANCHOR` or `PLI_ATTENTION_REDUCED_ANCHOR` (`PLI_REDUITE_ANCRE`) | entier | `0` | `pli/kernels/qwen/dense/v1/exact_attention.py:122` |  | m1-ultra/qwen3_5 `32` | Reduced attention anchored to positions: a row reads the pages chosen at its anchor max(N * (q // N), request start); text no longer depends on drafts. *[switches.md : stable]* |
| `attention.reduced_attm` | experimental | `--attention.reduced_attm` | `PLI_ATTENTION_REDUCED_ATTM` (`PLI_ATTENTION_REDUITE_ATTM`) | drapeau | `1` | `pli/kernels/qwen/dense/v1/exact_attention.py:66` |  |  | Reduced layers go through attm (0 = older path without attm). *[switches.md : reglage-fin]* |
| `attention.reduced_full_layers` | experimental | `--attention.reduced_full_layers` | `PLI_ATTENTION_REDUCED_FULL_LAYERS` (`PLI_ATTENTION_REDUITE_PLEINES`) | texte | `3,7,59` | `pli/kernels/qwen/dense/v1/exact_attention.py:61` |  |  | Layers always read in full by the reduced attention. *[switches.md : reglage-fin]* |
| `attention.reduced_min_rows` | experimental | `--attention.reduced_min_rows` | `PLI_ATTENTION_REDUCED_MIN_ROWS` (`PLI_ATTENTION_REDUITE_MIN`) | entier | `2` | `pli/kernels/qwen/dense/v1/exact_attention.py:64` |  |  | Fewest rows in a round for the reduction to apply (without the anchor). *[switches.md : reglage-fin]* |
| `attention.reduced_rounds` | recommande | `--attention.reduced_rounds` | `PLI_REDUCED_ROUNDS` or `PLI_ATTENTION_REDUCED_ROUNDS` (`PLI_REDUITE_TOURS`) | entier | `1` | `pli/kernels/qwen/dense/v1/exact_attention.py:72` |  | m1-ultra/qwen3_5 `4` | A layer keeps its reduced-attention page choice for K rounds; keys added since are read in full. *[switches.md : stable]* |
| `attention.reduced_threshold` | experimental | `--attention.reduced_threshold` | `PLI_ATTENTION_REDUCED_THRESHOLD` (`PLI_ATTENTION_REDUITE_SEUIL`) | entier | `20000` | `pli/kernels/qwen/dense/v1/exact_attention.py:60` |  |  | Context keys beyond which the reduced attention applies (at least 4,096). *[switches.md : reglage-fin]* |
| `attention.row_attention` | experimental | `--attention.row_attention` | `PLI_ROW_ATTENTION` or `PLI_ATTENTION_ROW_ATTENTION` (`TF_ROW_ATTENTION`) | drapeau | `0` | `pli/kernels/qwen/dense/v1/row_forward.py:22` |  |  |  |
| `attention.s3b` | recommande | `--attention.s3b` | `PLI_S3B` or `PLI_ATTENTION_S3B` | drapeau | `0` | `pli/kernels/qwen/dense/v1/exact_attention.py:126` |  | m1-ultra/qwen3_5 `1` | S3b: prompt chunks of at most PLI_S3B_ROWS rows past PLI_S3B_THRESHOLD positions go through the anchored reduced attention (PLI_ATTENTION_REDUCED, PLI_REDUCED_ANCHOR); anchors bounded at the start of the request's prefill, full layers read whole (text can change). *[switches.md : stable]* |
| `attention.s3b_rows` | recommande (?) | `--attention.s3b_rows` | `PLI_S3B_ROWS` or `PLI_ATTENTION_S3B_ROWS` (`PLI_S3B_LIGNES`) | entier | `2047` | `pli/kernels/qwen/dense/v1/exact_attention.py:127` |  |  | Largest prompt chunk (rows) that S3b reads through the anchored reduced attention. *[switches.md : reglage-fin]* |
| `attention.s3b_threshold` | recommande | `--attention.s3b_threshold` | `PLI_S3B_THRESHOLD` or `PLI_ATTENTION_S3B_THRESHOLD` (`PLI_S3B_SEUIL`) | entier | `65536` | `pli/kernels/qwen/dense/v1/exact_attention.py:128` |  | m1-ultra/qwen3_5 `65536` | Positions past which S3b applies (at least 4,096). *[switches.md : stable]* |
| `attention.tuning_file` | experimental | `--attention.tuning_file` | `PLI_ATTENTION_TUNING_FILE` (`PLI_ATTENTION_REGLAGE`) | texte | vide | `pli/kernels/qwen/dense/v1/exact_attention.py:375` |  |  | Benches only: file that changes the reduced-attention share at run time. *[switches.md : experimental]* |

### arbres

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `arbres.alone` | experimental | `--arbres.alone` | `PLI_TREE_ALONE` or `PLI_ARBRES_ALONE` (`PLI_ARBRE_SEUL`) | drapeau | `0` | `pli/engine/family_depth.py:35` |  |  | Trees only for a request decoding alone; with others, each tree is cut to its main branch, run as a chain. *[switches.md : experimental]* |
| `arbres.attm` | recommande | `--arbres.attm` | `PLI_TREES_ATTM` (`PLI_ARBRES_ATTM`) | drapeau | `0` | `pli/kernels/qwen/dense/v1/row_forward.py:25` |  | m1-ultra/qwen3_5 `1` | Lets the draft propose trees, verified through attm with an ancestor mask; chains keep their path and their bits. PLI_TREE_EXACT gives every node the bits of the same row in a chain. *[switches.md : stable]* |
| `arbres.children` | experimental | `--arbres.children` | `PLI_TREE_CHILDREN` or `PLI_ARBRES_CHILDREN` (`PLI_ARBRE_ENFANTS`) | entier | vide | `pli/drafters/dflash_proposer.py:226` |  |  | Draft tree setting: candidates expanded under each node. *[switches.md : experimental]* |
| `arbres.correct` | experimental | `--arbres.correct` | `PLI_TREE_CORRECT` or `PLI_ARBRES_CORRECT` (`PLI_ARBRE_CORR`) | drapeau | `0` | `pli/engine/family_depth.py:32` |  |  | Corrects the odds of tree nodes by the observed ratio of accepted to expected tokens. *[switches.md : experimental]* |
| `arbres.ctx_max` | experimental | `--arbres.ctx_max` | `PLI_TREE_CTX_MAX` or `PLI_ARBRES_CTX_MAX` (`PLI_ARBRE_CTX_MAX`) | entier | `0` | `pli/families/qwen3_5/dflash_head.py:19` |  |  | Draft trees only below N context positions (0 = everywhere). *[switches.md : experimental]* |
| `arbres.deltanet` | recommande | `--arbres.deltanet` | `PLI_TREE_DELTANET` or `PLI_ARBRES_DELTANET` (`PLI_ARBRE_DN`) | choix : 1/3 | `0` | `pli/kernels/qwen/dense/v1/row_glue.py:227`<br>`pli/kernels/qwen/dense/v1/row_glue.py:230` |  | m1-ultra/qwen3_5 `3` | DeltaNet recurrence kernel for draft trees of at most 8 nodes. 1: the nodes' states in threadgroup memory, same arithmetic in the same order as the original tree path (same bits); 3: the same, but only the nodes with a child further than the next node keep their state, in at most 4 slots (less threadgroup memory). *[switches.md : stable]* |
| `arbres.edge` | experimental | `--arbres.edge` | `PLI_TREE_EDGE` or `PLI_ARBRES_EDGE` (`PLI_ARBRE_ARETE`) | reel | vide | `pli/drafters/dflash_proposer.py:228` |  |  | Draft tree setting: weight of the pairwise score. *[switches.md : experimental]* |
| `arbres.exact` | recommande | `--arbres.exact` | `PLI_TREE_EXACT` or `PLI_ARBRES_EXACT` (`PLI_ARBRE_EXACT`) | drapeau | `0` | `pli/kernels/qwen/dense/v1/exact_attention.py:274`<br>`pli/kernels/qwen/dense/v1/exact_attention.py:311` (+1) |  | m1-ultra/qwen3_5 `1` | Tree attention with the chain's bits for every node: with the draft, the text is the one of decoding one token at a time (5.6.2; 10 answers out of 10 identical, same speed). *[switches.md : stable]* |
| `arbres.exact2` | experimental | `--arbres.exact2` | `PLI_TREE_EXACT2` or `PLI_ARBRES_EXACT2` (`PLI_ARBRE_EXACT2`) | drapeau | `1` | `pli/additions/attention_attm.py:623` |  |  | With PLI_TREE_EXACT, the per-node tails in one kernel call (same bits); 0 keeps the loop over nodes. *[switches.md : experimental]* |
| `arbres.gain_min` | experimental | `--arbres.gain_min` | `PLI_TREE_GAIN_MIN` or `PLI_ARBRES_GAIN_MIN` (`PLI_ARBRE_GAIN_MIN`) | texte | vide | `pli/additions/kflash.py:132` |  |  | A KFlash tree is kept only if its expected tokens exceed those of DFlash2's greedy chain by at least N; otherwise the chain is verified alone. Empty: always the tree. *[switches.md : experimental]* |
| `arbres.hybrid_trees` | recommande | `--arbres.hybrid_trees` | `PLI_HYBRID_TREES` or `PLI_ARBRES_HYBRID_TREES` (`PLI_HYBRIDE_ARBRES`) | drapeau | `0` | `pli/families/qwen3_5_moe/hybrid.py:66` |  | m1-ultra/qwen3_5 `1` | Lets DFlash2 propose trees inside the hybrid (needs PLI_TREES_ATTM or PLI_ROW_ATTENTION). *[switches.md : stable]* |
| `arbres.kflash` | recommande | `--arbres.kflash` | `PLI_KFLASH` or `PLI_ARBRES_KFLASH` | drapeau | `0` | `pli/additions/kflash.py:25` |  | m1-ultra/qwen3_5 `1` | Draft trees in KFlash's manner (method from Kapitsa): best-first tree over the draft's probabilities, calibrated per depth and learned online. Needs PLI_TREES_ATTM=1 (and PLI_HYBRID_TREES=1 with the hybrid). The verify stays exact; the tree only decides how many tokens a round keeps. *[switches.md : stable]* |
| `arbres.nodes` | experimental | `--arbres.nodes` | `PLI_TREE_NODES` or `PLI_ARBRES_NODES` (`PLI_ARBRE_NOEUDS`) | entier | `0` | `pli/drafters/dflash_proposer.py:220`<br>`pli/families/qwen3_5/family.py:37` (+2) |  |  | Caps the nodes of a draft tree (7 = 8 rows verified per round). *[switches.md : experimental]* |
| `arbres.noise` | experimental | `--arbres.noise` | `PLI_TREE_NOISE` or `PLI_ARBRES_NOISE` (`PLI_ARBRE_BRUIT`) | reel | vide | `pli/drafters/dflash_proposer.py:229`<br>`pli/drafters/dflash_proposer.py:229` |  |  | Draft tree setting: Gumbel noise weight. *[switches.md : experimental]* |
| `arbres.short_only` | recommande | `--arbres.short_only` | `PLI_TREES_SHORT_ONLY` or `PLI_ARBRES_SHORT_ONLY` (`PLI_ARBRES_COURTES`) | drapeau | `0` | `pli/families/qwen3_5_moe/hybrid.py:41` |  | m1-ultra/qwen3_5 `1` | With PLI_HYBRID_TREES, DFlash2's trees only for short questions (at most PLI_HYBRID_MESSAGES messages); agent sessions keep the chain. *[switches.md : stable]* |
| `arbres.tau` | experimental | `--arbres.tau` | `PLI_TREE_TAU` or `PLI_ARBRES_TAU` (`PLI_ARBRE_TAU`) | reel | vide | `pli/drafters/dflash_proposer.py:227` |  |  | Draft tree setting: temperature of the node scores. *[switches.md : experimental]* |
| `arbres.token_threshold` | recommande | `--arbres.token_threshold` | `PLI_TREES_TOKEN_THRESHOLD` or `PLI_ARBRES_TOKEN_THRESHOLD` (`PLI_ARBRES_SEUIL_JETONS`) | reel | `0` | `pli/families/qwen3_5_moe/hybrid.py:45` |  | m1-ultra/qwen3_5 `4.5` | On a short question, the chain stays while the moving average of the tokens kept per round (about ten rounds) reaches N; below N, the tree. *[switches.md : stable]* |

### ane

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ane.block_ids` | experimental | `--ane.block_ids` | `PLI_ANE_BLOCK_IDS` (`PLI_ANE_BLOCS_IDS`) | texte | `1,3` | `pli/additions/ane/ffn.py:63` |  |  | Neural Engine blocks the two programs are pinned on (PLI_ANE_BLOCKS=2). *[switches.md : reglage-fin]* |
| `ane.blocks` | recommande | `--ane.blocks` | `PLI_ANE_BLOCKS` (`PLI_ANE_BLOCS`) | entier | `1` | `pli/additions/ane/ffn.py:54` |  | m1-ultra/qwen3_5 `2` | Neural Engine share split into two programs pinned on two Neural Engine blocks (Ultra chips only). *[switches.md : stable]* |
| `ane.ffn` | recommande | `--ane.ffn` | `PLI_ANE_FFN` | drapeau | `0` | `pli/additions/ane/ffn.py:37` |  | m1-ultra/qwen3_5 `1` | Part of each dense FFN layer runs on the Neural Engine while the GPU reads the prompt. *[switches.md : stable]* |
| `ane.ffn_min_rows` | experimental | `--ane.ffn_min_rows` | `PLI_ANE_FFN_MIN_ROWS` (`PLI_ANE_FFN_LIGNES`) | entier | `512` | `pli/additions/ane/ffn.py:39` |  |  | Smallest chunk (rows) given a Neural Engine share when PLI_ANE_STEPS is empty; floor 128. *[switches.md : reglage-fin]* |
| `ane.ffn_units` | recommande | `--ane.ffn_units` | `PLI_ANE_FFN_UNITS` (`PLI_ANE_FFN_UNITES`) | entier | `12` | `pli/additions/ane/ffn.py:38` |  | m1-ultra/qwen3_5 `18` | Units of 512 FFN channels given to the Neural Engine (used when PLI_ANE_STEPS is empty). *[switches.md : stable]* |
| `ane.heartbeat_ane` | experimental | `--ane.heartbeat_ane` | `PLI_HEARTBEAT_ANE` or `PLI_ANE_HEARTBEAT_ANE` (`PLI_BATTEMENT_ANE`) | drapeau | `0` | `pli/server/scheduler.py:31` |  |  | A Neural Engine keep-awake call at the heartbeat's pace. *[switches.md : experimental]* |
| `ane.keepawake` | recommande | `--ane.keepawake` | `PLI_ANE_KEEPAWAKE` (`PLI_ANE_GARDE`) | drapeau | `0` | `pli/additions/ane/ffn.py:70` |  | m1-ultra/qwen3_5 `1` | Keeps the Neural Engine awake between two shared layers with tiny calls (it slows down after about 2 ms idle). *[switches.md : stable]* |
| `ane.keepawake_ms` | experimental | `--ane.keepawake_ms` | `PLI_ANE_KEEPAWAKE_MS` (`PLI_ANE_GARDE_MS`) | reel | `500` | `pli/additions/ane/ffn.py:71` |  |  | The keep-awake thread stops this many ms after the last shared layer. *[switches.md : reglage-fin]* |
| `ane.requant_after` | experimental | `--ane.requant_after` | `PLI_ANE_REQUANT_AFTER` (`PLI_ANE_RQ_APRES`) | drapeau | `0` | `pli/additions/ane/ffn.py:53` |  |  | Starts the CPU requantization of the next layer after the Neural Engine call instead of during it. *[switches.md : experimental]* |
| `ane.requant_cpu` | recommande | `--ane.requant_cpu` | `PLI_ANE_REQUANT_CPU` | drapeau | `0` | `pli/additions/ane/ffn.py:64` |  | m1-ultra/qwen3_5 `1` | Neural Engine weights requantized on the CPU (NEON) during the previous layer instead of on the GPU; same bits. *[switches.md : stable]* |
| `ane.requant_mode` | experimental | `--ane.requant_mode` | `PLI_ANE_REQUANT_MODE` | entier | `1` | `pli/additions/ane/requant_cpu.py:26` |  |  | Rounding of the CPU requantization: 1 = one rounding after FMA, as the Metal kernel; 0 = two bf16 roundings; +2 forces the scalar path. *[switches.md : reglage-fin]* |
| `ane.requant_threads` | recommande | `--ane.requant_threads` | `PLI_ANE_REQUANT_THREADS` (`PLI_ANE_REQUANT_FILS`) | entier | `8` | `pli/additions/ane/requant_cpu.py:27` |  | m1-ultra/qwen3_5 `8` | CPU threads used for that requantization. *[switches.md : stable]* |
| `ane.steps` | recommande | `--ane.steps` | `PLI_ANE_STEPS` (`PLI_ANE_PALIERS`) | texte | vide | `pli/additions/ane/ffn.py:47` |  | m1-ultra/qwen3_5 `256:14,512:16,1536:18` | Neural Engine units by chunk size, as rows:units,...; one program set per step, each for chunks of at least that many rows. *[switches.md : stable]* |

### gen

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gen.actif` | recommande | `--gen.actif` | `PLI_GEN` or `PLI_GEN_ACTIF` | drapeau |  | `pli/gen/config.py:42`<br>`pli/server/http.py:95` |  |  |  |
| `gen.attente_memoire_s` | experimental | `--gen.attente_memoire_s` | `PLI_GEN_ATTENTE_MEMOIRE_S` | reel | `1800` | `pli/gen/fils_video.py:392` |  |  |  |
| `gen.besoin` | experimental | `--gen.besoin` | `PLI_GEN_BESOIN` | texte | vide | `pli/gen/fils_video.py:389` |  |  |  |
| `gen.cache_go` | experimental | `--gen.cache_go` | `PLI_GEN_CACHE_GO` | reel | `2` | `pli/gen/memoire_general.py:214` |  |  |  |
| `gen.catalogue_prive` | experimental | `--gen.catalogue_prive` | `PLI_GEN_CATALOGUE_PRIVE` | texte | vide | `pli/gen/catalogue_prive.py:49`<br>`pli/gen/config.py:142` |  |  | PLI 5: TOML files of private LoRA (separated by :), added to the image and video catalogues; a name that is already public is refused at startup. Give it through the catalogue_prive key of PLI_GEN_CONFIG (child processes only): in the server's environment it would change the checkpoint key. See docs/usage-expert.md, « Private LoRA catalog ». *[switches.md : reglage-fin]* |
| `gen.config` | recommande | `--gen.config` | `PLI_GEN_CONFIG` | texte | vide | `pli/gen/config.py:97` |  |  |  |
| `gen.drapeau` | experimental | `--gen.drapeau` | `PLI_GEN_DRAPEAU` | texte | vide | `pli/gen/fils_audio.py:763`<br>`pli/gen/fils_image.py:816` (+2) |  |  |  |
| `gen.drapeau_autres` | experimental | `--gen.drapeau_autres` | `PLI_GEN_DRAPEAU_AUTRES` | texte | vide | `pli/gen/fils_video.py:384` |  |  |  |
| `gen.engine_wait_s` | experimental | `--gen.engine_wait_s` | `PLI_GEN_ENGINE_WAIT_S` | reel | `60` | `pli/gen/memoire_general.py:216` |  |  | PLI 5.6.3: how long freeing or capping the LLM for a generation child waits for the LLM to be idle (seconds; 5 before 5.6.3). Past it, nothing is freed and the child starts anyway. *[switches.md : reglage-fin]* |
| `gen.garde_journal` | recommande (?) | `--gen.garde_journal` | `PLI_GEN_GARDE_JOURNAL` | texte |  | `pli/gen/garde.py:415` |  |  |  |
| `gen.garde_llm` | experimental | `--gen.garde_llm` | `PLI_GEN_GARDE_LLM` | drapeau |  | `pli/gen/garde.py:416` |  |  |  |
| `gen.garde_llm_delai_s` | experimental | `--gen.garde_llm_delai_s` | `PLI_GEN_GARDE_LLM_DELAI_S` | reel | `(calculated: DELAI_AVIS_S)` | `pli/gen/garde.py:352` |  |  |  |
| `gen.garde_noms` | experimental | `--gen.garde_noms` | `PLI_GEN_GARDE_NOMS` | texte |  | `pli/gen/garde.py:415` |  |  |  |
| `gen.marge_go` | experimental | `--gen.marge_go` | `PLI_GEN_MARGE_GO` | reel | `2` | `pli/gen/memoire_general.py:213` |  |  |  |
| `gen.pids` | experimental | `--gen.pids` | `PLI_GEN_PIDS` | texte |  | `pli/gen/superviseur.py:251` |  |  |  |
| `gen.reserve_go` | experimental | `--gen.reserve_go` | `PLI_GEN_RESERVE_GO` | reel | `(calculated: RESERVE_PAR_DEFAUT_GO)` | `pli/gen/config.py:103`<br>`pli/gen/superviseur.py:225` |  |  |  |
| `gen.sorties` | experimental | `--gen.sorties` | `PLI_GEN_SORTIES` | texte | vide | `pli/gen/fils_image.py:812` |  |  |  |

### gen.voix

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gen.voix.drapeau_audio` | experimental | `--gen.voix.drapeau_audio` | `PLI_GEN_DRAPEAU_AUDIO` or `PLI_GEN_VOIX_DRAPEAU_AUDIO` | texte | vide | `pli/gen/fils_video.py:388` |  |  |  |
| `gen.voix.drapeau_voix` | experimental | `--gen.voix.drapeau_voix` | `PLI_GEN_DRAPEAU_VOIX` or `PLI_GEN_VOIX_DRAPEAU_VOIX` | texte | vide | `pli/gen/fils_video.py:386` |  |  |  |
| `gen.voix.voix_octets` | experimental | `--gen.voix.voix_octets` | `PLI_GEN_VOIX_OCTETS` or `PLI_GEN_VOIX_VOIX_OCTETS` | entier | `0` | `pli/gen/fils_video.py:390` |  |  |  |
| `gen.voix.voix_reserve_octets` | experimental | `--gen.voix.voix_reserve_octets` | `PLI_GEN_VOIX_RESERVE_OCTETS` or `PLI_GEN_VOIX_VOIX_RESERVE_OCTETS` | entier | `0` | `pli/gen/fils_video.py:391` |  |  |  |

### gen.video

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gen.video.drapeau_video` | experimental | `--gen.video.drapeau_video` | `PLI_GEN_DRAPEAU_VIDEO` or `PLI_GEN_VIDEO_DRAPEAU_VIDEO` | texte | vide | `pli/gen/fils_video.py:387` |  |  |  |
| `gen.video.video_avec_general` | experimental | `--gen.video.video_avec_general` | `PLI_GEN_VIDEO_AVEC_GENERAL` or `PLI_GEN_VIDEO_VIDEO_AVEC_GENERAL` | drapeau |  | `pli/gen/routes.py:57` |  |  |  |

### taches

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `taches.actives` | experimental | `--taches.actives` | `PLI_TASKS` or `PLI_TACHES_ACTIVES` | texte | vide | `pli/tasks/__init__.py:235` |  |  | Background tasks PLI runs itself in the quiet moments of service, comma-separated: prune (session snapshots), drafter (DFlash2 drafter changed between two requests, judged in service). Empty: none. *[switches.md : experimental]* |
| `taches.calm_s` | experimental | `--taches.calm_s` | `PLI_TASKS_CALM_S` or `PLI_TACHES_CALM_S` | texte | `30.0` | `pli/tasks/__init__.py:244` |  |  | Seconds without engine activity before a background step runs. *[switches.md : experimental]* |
| `taches.drafter_drop` | experimental | `--taches.drafter_drop` | `PLI_TASKS_DRAFTER_DROP` or `PLI_TACHES_DRAFTER_DROP` | texte | `0.03` | `pli/tasks/drafter.py:89` |  |  | drafter: relative drop of the share of drafted tokens that brings the previous drafter back. *[switches.md : experimental]* |
| `taches.drafter_file` | experimental | `--taches.drafter_file` | `PLI_TASKS_DRAFTER_FILE` or `PLI_TACHES_DRAFTER_FILE` | texte | vide | `pli/tasks/drafter.py:86`<br>`pli/tasks/falcon.py:277` |  |  | drafter: file naming (first line) the drafter folder to serve; written by FALCON B. *[switches.md : experimental]* |
| `taches.drafter_min_days` | experimental | `--taches.drafter_min_days` | `PLI_TASKS_DRAFTER_MIN_DAYS` or `PLI_TACHES_DRAFTER_MIN_DAYS` | texte | `3.0` | `pli/tasks/drafter.py:87` |  |  | drafter: days a newly served drafter is watched before it is judged or replaced. *[switches.md : experimental]* |
| `taches.drafter_min_requests` | experimental | `--taches.drafter_min_requests` | `PLI_TASKS_DRAFTER_MIN_REQUESTS` or `PLI_TACHES_DRAFTER_MIN_REQUESTS` | entier | `50` | `pli/tasks/drafter.py:88` |  |  | drafter: requests per size both drafters need for a size to count in the comparison. *[switches.md : experimental]* |
| `taches.falcon_anchors` | experimental | `--taches.falcon_anchors` | `PLI_TASKS_FALCON_ANCHORS` or `PLI_TACHES_FALCON_ANCHORS` | entier | `256` | `pli/tasks/falcon.py:274` |  |  | falcon: anchors drawn per training conversation. *[switches.md : experimental]* |
| `taches.falcon_anchors_control` | experimental | `--taches.falcon_anchors_control` | `PLI_TASKS_FALCON_ANCHORS_CONTROL` or `PLI_TACHES_FALCON_ANCHORS_CONTROL` | entier | `128` | `pli/tasks/falcon.py:275` |  |  | falcon: anchors drawn per control conversation. *[switches.md : experimental]* |
| `taches.falcon_base_file` | experimental | `--taches.falcon_base_file` | `PLI_TASKS_FALCON_BASE_FILE` or `PLI_TACHES_FALCON_BASE_FILE` | texte | vide | `pli/tasks/falcon.py:278` |  |  | falcon: file naming the drafter a cycle starts from. *[switches.md : experimental]* |
| `taches.falcon_chunk` | experimental | `--taches.falcon_chunk` | `PLI_TASKS_FALCON_CHUNK` or `PLI_TACHES_FALCON_CHUNK` | entier | `512` | `pli/tasks/falcon.py:273` |  |  | falcon: tokens of the 27B read per step. *[switches.md : experimental]* |
| `taches.falcon_control` | experimental | `--taches.falcon_control` | `PLI_TASKS_FALCON_CONTROL` or `PLI_TACHES_FALCON_CONTROL` | entier | `12` | `pli/tasks/falcon.py:272` |  |  | falcon: control conversations per cycle. *[switches.md : experimental]* |
| `taches.falcon_dir` | experimental | `--taches.falcon_dir` | `PLI_TASKS_FALCON_DIR` or `PLI_TACHES_FALCON_DIR` | texte | vide | `pli/tasks/falcon.py:257` |  |  | falcon: folder of courant.txt, historique.jsonl, preparer_falcon.py and the cycles. *[switches.md : experimental]* |
| `taches.falcon_every_h` | experimental | `--taches.falcon_every_h` | `PLI_TASKS_FALCON_EVERY_H` or `PLI_TACHES_FALCON_EVERY_H` | texte | `24.0` | `pli/tasks/falcon.py:269` |  |  | falcon: hours between two cycles. *[switches.md : experimental]* |
| `taches.falcon_hours` | experimental | `--taches.falcon_hours` | `PLI_TASKS_FALCON_HOURS` or `PLI_TACHES_FALCON_HOURS` | texte | `1-7` | `pli/tasks/falcon.py:268` |  |  | falcon: local hours (start-end) where an adjustment cycle may start; it then goes on in any quiet moment. *[switches.md : experimental]* |
| `taches.falcon_sessions` | experimental | `--taches.falcon_sessions` | `PLI_TASKS_FALCON_SESSIONS` or `PLI_TACHES_FALCON_SESSIONS` | entier | `60` | `pli/tasks/falcon.py:271` |  |  | falcon: training conversations at most per cycle. *[switches.md : experimental]* |
| `taches.falcon_shadow` | experimental | `--taches.falcon_shadow` | `PLI_TASKS_FALCON_SHADOW` or `PLI_TACHES_FALCON_SHADOW` | drapeau | `0` | `pli/tasks/falcon.py:276` |  |  | falcon: 1 = beside the night job, without taking its place (own folder, night job's current drafter as base, nothing proposed to drafter), so both gains can be compared. *[switches.md : experimental]* |
| `taches.falcon_since_h` | experimental | `--taches.falcon_since_h` | `PLI_TASKS_FALCON_SINCE_H` or `PLI_TACHES_FALCON_SINCE_H` | texte | `26.0` | `pli/tasks/falcon.py:270` |  |  | falcon: hours of recent requests used for training. *[switches.md : experimental]* |
| `taches.falcon_tokens` | experimental | `--taches.falcon_tokens` | `PLI_TASKS_FALCON_TOKENS` or `PLI_TACHES_FALCON_TOKENS` | texte | vide | `pli/tasks/falcon.py:258` |  |  | falcon: folder of the kept tokens. *[switches.md : experimental]* |
| `taches.falcon_zlab` | experimental | `--taches.falcon_zlab` | `PLI_TASKS_FALCON_ZLAB` or `PLI_TACHES_FALCON_ZLAB` | texte | vide | `pli/tasks/falcon.py:260` |  |  | falcon: reference drafter of the gains. *[switches.md : experimental]* |
| `taches.log` | experimental | `--taches.log` | `PLI_TASKS_LOG` or `PLI_TACHES_LOG` | texte | vide | `pli/tasks/__init__.py:244` |  |  | Journal of the background tasks, one JSON line per event. *[switches.md : experimental]* |
| `taches.max_errors` | experimental | `--taches.max_errors` | `PLI_TASKS_MAX_ERRORS` or `PLI_TACHES_MAX_ERRORS` | entier | `3` | `pli/tasks/__init__.py:247` |  |  | Errors in a row that switch a background task off until restart. *[switches.md : experimental]* |
| `taches.memory_fraction` | experimental | `--taches.memory_fraction` | `PLI_TASKS_MEMORY_FRACTION` or `PLI_TACHES_MEMORY_FRACTION` | texte | `0.9` | `pli/tasks/__init__.py:246` |  |  | A step that needs memory waits while MLX's active memory plus its need passes this share of the recommended working set. *[switches.md : experimental]* |
| `taches.prune_cap_gb` | experimental | `--taches.prune_cap_gb` | `PLI_TASKS_PRUNE_CAP_GB` or `PLI_TACHES_PRUNE_CAP_GB` | texte | `60.0` | `pli/tasks/prune.py:66`<br>`pli/tasks/prune.py:68` |  |  | prune: past this size for the whole folder, the least recently used snapshots go. *[switches.md : experimental]* **To check : different defaults depending on the location: '', '60.0'.** |
| `taches.prune_cap_percent` | experimental | `--taches.prune_cap_percent` | `PLI_TASKS_PRUNE_CAP_PERCENT` or `PLI_TACHES_PRUNE_CAP_PERCENT` | texte | `0.0` | `pli/tasks/prune.py:67` |  |  | prune: the cap as a share (%) of the space the folder may use: disk free space plus the folder's size, read at each pass; with PLI_TASKS_PRUNE_CAP_GB set too, the smaller. *[switches.md : experimental]* |
| `taches.prune_days` | experimental | `--taches.prune_days` | `PLI_TASKS_PRUNE_DAYS` or `PLI_TACHES_PRUNE_DAYS` | texte | `7.0` | `pli/tasks/prune.py:65` |  |  | prune: a snapshot unused for this many days is deleted. *[switches.md : experimental]* |
| `taches.prune_dir` | experimental | `--taches.prune_dir` | `PLI_TASKS_PRUNE_DIR` or `PLI_TACHES_PRUNE_DIR` | texte | vide | `pli/tasks/prune.py:61` |  |  | Folder pruned by the prune task (default: the session-snapshot folder beside --snapshot-dir). *[switches.md : experimental]* |
| `taches.prune_every_s` | experimental | `--taches.prune_every_s` | `PLI_TASKS_PRUNE_EVERY_S` or `PLI_TACHES_PRUNE_EVERY_S` | texte | `86400.0` | `pli/tasks/prune.py:69` |  |  | prune: seconds between two passes (once a day, as the cron it replaces). *[switches.md : experimental]* |
| `taches.state` | experimental | `--taches.state` | `PLI_TASKS_STATE` or `PLI_TACHES_STATE` | texte | vide | `pli/tasks/__init__.py:245` |  |  | Folder of the background tasks' progress (one JSON file per task), read back at start. *[switches.md : experimental]* |

### reseau

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reseau.allowed_hosts` | recommande | `--reseau.allowed_hosts` | `PLI_ALLOWED_HOSTS` or `PLI_RESEAU_ALLOWED_HOSTS` (`PLI_HOTES_ADMIS`) | texte | vide | `pli/server/host_guard.py:75` |  |  | Extra host names admitted by the Host/Origin guard (for example the machine's LAN name). *[switches.md : machine]* |
| `reseau.api_key` | recommande | `--api-key`, `--reseau.api_key` | `PLI_RESEAU_API_KEY` | liste | `[]` | `pli/cli_args.py:41` |  |  | require this API key; repeat for more keys |
| `reseau.api_key_file` | recommande | `--api-key-file`, `--reseau.api_key_file` | `PLI_RESEAU_API_KEY_FILE` | texte |  | `pli/cli_args.py:42` |  |  | restricted key file, one key or label: key per line; # comments |
| `reseau.bench_api_key` | experimental | `--reseau.bench_api_key` | `PLI_API_KEY` or `PLI_RESEAU_BENCH_API_KEY` (`TENSORFOLD_API_KEY`) | texte | vide | `pli/bench/pli_bench.py:336`<br>`pli/server/authentication.py:148` |  |  |  |
| `reseau.host` | recommande | `--host`, `--reseau.host` | `PLI_RESEAU_HOST` | texte | `127.0.0.1` | `pli/cli_args.py:37` |  |  | address to listen on (0.0.0.0: every interface) |
| `reseau.host_guard` | recommande | `--reseau.host_guard` | `PLI_HOST_GUARD` or `PLI_RESEAU_HOST_GUARD` (`PLI_GARDE_HOTE`) | choix : 1/on/oui/yes | vide | `pli/server/host_guard.py:73` |  | m1-ultra/common `1` | Host/Origin guard: 403 for a non-local Host on a local address, or an Origin whose host is not admitted. *[switches.md : stable]* |
| `reseau.keep_reply` | recommande | `--reseau.keep_reply` | `PLI_KEEP_REPLY` or `PLI_RESEAU_KEEP_REPLY` (`PLI_GARDE_REPONSE`) | drapeau | `0` | `pli/engine/family_prefill.py:11`<br>`pli/engine/lane_engine.py:12` (+2) |  | m1-ultra/common `1` | Keeps the reply for the next agent turn: the chat template keeps the empty think block, the engine keeps the decoded state (text can change). *[switches.md : stable]* |
| `reseau.metrics_open` | recommande | `--metrics-open`, `--reseau.metrics_open` | `PLI_RESEAU_METRICS_OPEN` | drapeau | `False` | `pli/cli_args.py:43` |  |  | allow metrics without an API key |
| `reseau.no_update_check` | obsolete | `--no-update-check`, `--reseau.no_update_check` | `PLI_RESEAU_NO_UPDATE_CHECK` | drapeau | `False` | `pli/cli_args.py:119` |  |  | kept for compatibility: PLI never asks GitHub for a newer release (also PLI_NO_UPDATE_CHECK=1) |
| `reseau.no_update_check_env` | obsolete | `--reseau.no_update_check_env` | `PLI_NO_UPDATE_CHECK` or `PLI_RESEAU_NO_UPDATE_CHECK_ENV` (`TENSORFOLD_NO_UPDATE_CHECK`) | texte | vide | `pli/update.py:79` |  |  |  |
| `reseau.port` | recommande | `--port`, `--reseau.port` | `PLI_RESEAU_PORT` | entier | `8080` | `pli/cli_args.py:38` |  |  |  |
| `reseau.test_launchd` | experimental | `--reseau.test_launchd` | `PLI_TEST_LAUNCHD` or `PLI_RESEAU_TEST_LAUNCHD` (`TENSORFOLD_TEST_LAUNCHD`) | texte |  | `control/test_macos_smoke.py:19` |  |  |  **To check : lu seulement hors de src/ : control/test_macos_smoke.py:19.** |

### journal

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `journal.anchor_log` | experimental | `--journal.anchor_log` | `PLI_ANCHOR_LOG` or `PLI_JOURNAL_ANCHOR_LOG` (`PLI_ANCRE_JOURNAL`) | drapeau |  | `pli/kernels/qwen/dense/v1/exact_attention.py:137` |  |  | Logs the reasons of the anchored reduced attention. *[switches.md : experimental]* |
| `journal.ane_stats` | experimental | `--journal.ane_stats` | `PLI_ANE_STATS` or `PLI_JOURNAL_ANE_STATS` (`PLI_ANE_RELEVE`) | texte | vide | `pli/additions/ane/ffn.py:76` |  |  | Writes cumulative Neural Engine time and CPU requantization time to a file every 0.5 s (load survey). *[switches.md : experimental]* |
| `journal.arbiter_log` | experimental | `--journal.arbiter_log` | `PLI_ARBITER_LOG` or `PLI_JOURNAL_ARBITER_LOG` (`PLI_ARBITRE_JOURNAL`) | texte | vide | `pli/additions/arbiter.py:30` |  |  | Log file of the arbiter. *[switches.md : experimental]* |
| `journal.checkpoint_log` | recommande (?) | `--journal.checkpoint_log` | `PLI_CHECKPOINT_LOG` or `PLI_JOURNAL_CHECKPOINT_LOG` (`PLI_POINTS_JOURNAL`) | drapeau | `0` | `pli/server/checkpoints.py:15` |  |  | Writes checkpoint_kept, checkpoint_evicted and checkpoint_refused events to the PLI_EVENTS log. *[switches.md : experimental]* |
| `journal.copy_log` | recommande (?) | `--journal.copy_log` | `PLI_COPY_LOG` or `PLI_JOURNAL_COPY_LOG` (`PLI_COPIE_JOURNAL`) | texte | vide | `pli/engine/lane_family.py:28` |  |  | jsonl file: each round writes its kind and the tokens it gave; a copy round also writes the copy, the head's queued chain and the match length (to choose offline when the copy should go first). The choice itself does not change. *[switches.md : experimental]* |
| `journal.draft_capture` | experimental | `--journal.draft_capture` | `PLI_DRAFT_CAPTURE` or `PLI_JOURNAL_DRAFT_CAPTURE` (`TF_DRAFT_CAPTURE`) | texte | vide | `pli/drafters/dflash_proposer.py:245` |  |  |  |
| `journal.draft_log` | experimental | `--journal.draft_log` | `PLI_DRAFT_LOG` or `PLI_JOURNAL_DRAFT_LOG` (`TF_DRAFT_LOG`) | texte | vide | `pli/families/qwen3_5/dflash_head.py:127` |  |  |  |
| `journal.draft_profile` | experimental | `--journal.draft_profile` | `PLI_DRAFT_PROFILE` or `PLI_JOURNAL_DRAFT_PROFILE` (`TF_DRAFT_PROFILE`) | drapeau | vide | `pli/drafters/dflash_proposer.py:18` |  |  |  |
| `journal.events` | recommande (?) | `--journal.events` | `PLI_EVENTS` or `PLI_JOURNAL_EVENTS` (`PLI_EVENEMENTS`) | texte | vide | `pli/additions/events.py:30` |  |  | Path of the event journal: one JSON line per scheduler action, with the conversation, no message content. *[switches.md : machine]* |
| `journal.events_max_mb` | experimental | `--journal.events_max_mb` | `PLI_EVENTS_MAX_MB` or `PLI_JOURNAL_EVENTS_MAX_MB` (`PLI_EVENEMENTS_MAX_MO`) | entier | `50` | `pli/additions/events.py:32` |  |  | The event journal rotates to <file>.1 beyond this size in MB. *[switches.md : reglage-fin]* |
| `journal.family_profile` | experimental | `--journal.family_profile` | `PLI_FAMILY_PROFILE` or `PLI_JOURNAL_FAMILY_PROFILE` (`TF_FAMILY_PROFILE`) | drapeau | vide | `pli/engine/family_common.py:9` |  |  |  |
| `journal.family_round_log` | experimental | `--journal.family_round_log` | `PLI_FAMILY_ROUND_LOG` or `PLI_JOURNAL_FAMILY_ROUND_LOG` (`TF_FAMILY_ROUND_LOG`) | texte | vide | `pli/engine/family_common.py:11` |  |  |  |
| `journal.hybrid_log` | experimental | `--journal.hybrid_log` | `PLI_HYBRID_LOG` or `PLI_JOURNAL_HYBRID_LOG` (`PLI_HYBRIDE_JOURNAL`) | texte | vide | `pli/families/qwen3_5_moe/hybrid.py:73` |  |  | File logging each switch of the hybrid draft. *[switches.md : experimental]* |
| `journal.lattice_log` | experimental | `--journal.lattice_log` | `PLI_LATTICE_LOG` or `PLI_JOURNAL_LATTICE_LOG` (`PLI_TREILLIS_JOURNAL`) | texte | vide | `pli/additions/lattice_log.py:14` |  |  | Folder of the log of the DFlash2 lattices served by the head, to replay offline the chain, the trees and the candidates' ceiling on the same text. Nothing changes in the computation or in the text. *[switches.md : experimental]* |
| `journal.log_times` | recommande (?) | `--journal.log_times` | `PLI_LOG_TIMES` or `PLI_JOURNAL_LOG_TIMES` (`PLI_HEURES`) | drapeau | `0` | `pli/server/app.py:22`<br>`pli/server/scheduler.py:564` (+1) |  | m1-ultra/common `1` | Arrival and end time added at the end of each request's done log line. *[switches.md : stable]* **To check : different defaults depending on the location: '', '0'.** |
| `journal.no_live` | experimental | `--journal.no_live` | `PLI_NO_LIVE` or `PLI_JOURNAL_NO_LIVE` (`TENSORFOLD_NO_LIVE`) | drapeau |  | `pli/server/live.py:204` |  |  |  |
| `journal.prefill_cpu_trace` | experimental | `--journal.prefill_cpu_trace` | `PLI_PREFILL_CPU_TRACE` or `PLI_JOURNAL_PREFILL_CPU_TRACE` (`PLI_LECTURE_CPU_TRACE`) | drapeau | `0` | `pli/families/qwen3_5/prefill_cpu.py:49` |  |  | One log line per chunk with the CPU share's times. *[switches.md : experimental]* |
| `journal.request_log` | experimental | `--journal.request_log` | `PLI_REQUEST_LOG` or `PLI_JOURNAL_REQUEST_LOG` (`TENSORFOLD_REQUEST_LOG`) | texte | vide | `pli/server/http.py:29` |  |  |  |
| `journal.round_profile` | experimental | `--journal.round_profile` | `PLI_ROUND_PROFILE` or `PLI_JOURNAL_ROUND_PROFILE` (`PLI_PROFIL_TOUR`) | drapeau | `0` | `pli/additions/round_profile.py:14` |  |  | Where the time of a decode round goes: one [PLI] round profile line per kind of round with the mean time of each step (verification, head, draw, next draft...). Each step is evaluated at its end, so rounds are a little longer; same operations, same order, the text does not change. *[switches.md : experimental]* |
| `journal.round_profile_bucket` | experimental | `--journal.round_profile_bucket` | `PLI_ROUND_PROFILE_BUCKET` or `PLI_JOURNAL_ROUND_PROFILE_BUCKET` (`PLI_PROFIL_TOUR_TRANCHE`) | entier | `0` | `pli/additions/round_profile.py:16` |  |  | PLI_ROUND_PROFILE: kinds of round also split by context bucket of N tokens (0 = not split). *[switches.md : experimental]* |
| `journal.round_profile_n` | experimental | `--journal.round_profile_n` | `PLI_ROUND_PROFILE_N` or `PLI_JOURNAL_ROUND_PROFILE_N` (`PLI_PROFIL_TOUR_N`) | entier | `200` | `pli/additions/round_profile.py:15` |  |  | PLI_ROUND_PROFILE: rounds between two lines, per kind. *[switches.md : experimental]* |
| `journal.trace` | experimental | `--journal.trace` | `PLI_TRACE` or `PLI_JOURNAL_TRACE` | texte | vide | `pli/additions/trace.py:36` |  |  | Step trace file, on from the start (benches, tests). *[switches.md : experimental]* |
| `journal.trace_dir` | recommande (?) | `--journal.trace_dir` | `PLI_TRACE_DIR` or `PLI_JOURNAL_TRACE_DIR` (`PLI_TRACE_DOSSIER`) | texte | vide | `pli/additions/trace.py:37` |  |  | Directory of the step trace, off until switched on with python -m pli.additions.trace on [--duration 24h] (no restart). *[switches.md : machine]* |
| `journal.trace_layers` | experimental | `--journal.trace_layers` | `PLI_TRACE_LAYERS` or `PLI_JOURNAL_TRACE_LAYERS` (`PLI_TRACE_COUCHES`) | drapeau | `0` | `pli/additions/trace_layers.py:25` |  |  | With the trace on, one line per chunk with the attention, DeltaNet and FFN times (slows reading). *[switches.md : experimental]* |
| `journal.trace_max_mb` | experimental | `--journal.trace_max_mb` | `PLI_TRACE_MAX_MB` or `PLI_JOURNAL_TRACE_MAX_MB` (`PLI_TRACE_MAX_MO`) | entier | `500` | `pli/additions/trace.py:39` |  |  | The step trace rotates to <file>.1 beyond this size in MB. *[switches.md : reglage-fin]* |
| `journal.tree_trace` | experimental | `--journal.tree_trace` | `PLI_TREE_TRACE` or `PLI_JOURNAL_TREE_TRACE` (`TF_TREE_TRACE`) | texte | vide | `pli/drafters/dflash_proposer.py:243` |  |  |  |

### expert

| Key | Status | Option | Variable (old names) | Type | Default | Read at | Production | m1-ultra profile | Aide |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `expert.f16_clamp` | experimental | `--expert.f16_clamp` | `PLI_F16_CLAMP` or `PLI_EXPERT_F16_CLAMP` (`PLI_F16_ECRETE`) | drapeau | `0` | `pli/additions/attention_attm.py:28`<br>`pli/kernels/qwen/dense/v1/prompt_attention_f16.py:31` |  |  | Clamps Q, K and V to +-65280 before the f16 conversion. *[switches.md : experimental]* |
| `expert.flash_dense` | experimental | `--expert.flash_dense` | `PLI_FLASH_DENSE` or `PLI_EXPERT_FLASH_DENSE` (`TF_FLASH_DENSE`) | texte |  | `pli/families/qwen4_exp/decode.py:84`<br>`pli/families/qwen4_exp/decode.py:165` |  |  |  |
| `expert.flash_fused` | experimental | `--expert.flash_fused` | `PLI_FLASH_FUSED` or `PLI_EXPERT_FLASH_FUSED` (`TF_FLASH_FUSED`) | drapeau | `1` | `pli/families/qwen4_exp/model.py:440` |  |  |  |
| `expert.lane_kernels` | experimental | `--lane-kernels`, `--expert.lane_kernels` | `PLI_EXPERT_LANE_KERNELS` | choix : auto/on/off | `auto` | `pli/cli_args.py:83` |  |  | lane kernels for Qwen3.8 dense (auto: on GPUs with tensor units) |
| `expert.moe_rows` | experimental | `--expert.moe_rows` | `PLI_MOE_ROWS` or `PLI_EXPERT_MOE_ROWS` (`TF_MOE_ROWS`) | texte | `batched` | `pli/kernels/qwen/dense/v1/row_forward.py:288` |  |  |  |
| `expert.relayout` | recommande | `--expert.relayout` | `PLI_RELAYOUT` or `PLI_EXPERT_RELAYOUT` (`PLI_RANGE`) | drapeau | `0` | `pli/additions/relayout.py:39` |  | m1-ultra/qwen3_5 `1` | 27B weights and scales re-laid out for the decode kernel; each matrix is checked at load (same bits required, else it keeps its layout). *[switches.md : stable]* |
| `expert.relayout_draft` | recommande | `--expert.relayout_draft` | `PLI_RELAYOUT_DRAFT` or `PLI_EXPERT_RELAYOUT_DRAFT` (`PLI_RANGE_BROUILLON`) | drapeau | `0` | `pli/additions/relayout.py:219` |  | m1-ultra/qwen3_5 `1` | Also relays out the draft's projections that read only the 8 rows of the block (q_proj, o_proj, gate, up, down, kernel_projection); each matrix checked at the same bits. Same drafts, same text. *[switches.md : stable]* |
| `expert.startup_cache` | experimental | `--expert.startup_cache` | `PLI_STARTUP_CACHE` or `PLI_EXPERT_STARTUP_CACHE` | texte | `0` | `pli/server/startup_cache.py:33` |  |  | PLI 5.6.3: startup measurements kept on disk, like pli serve --startup-cache on (1 or on; remeasure). Prefer the option: a PLI_* variable enters the settings fingerprint, so saved conversations would not be resumed. PLI_STARTUP_CACHE_DIR (default ~/.cache/pli/startup) holds one JSON file per key. *[switches.md : reglage-fin]* |
| `expert.startup_cache_dir` | experimental | `--expert.startup_cache_dir` | `PLI_STARTUP_CACHE_DIR` or `PLI_EXPERT_STARTUP_CACHE_DIR` | chemin | `~/.cache/pli/startup` | `pli/server/startup_cache.py:36` |  |  |  |
| `expert.startup_remeasure` | experimental | `--expert.startup_remeasure` | `PLI_STARTUP_REMEASURE` or `PLI_EXPERT_STARTUP_REMEASURE` | drapeau | `0` | `pli/server/startup_cache.py:35` |  |  |  |
| `expert.tiles_per_group` | recommande | `--expert.tiles_per_group` | `PLI_TILES_PER_GROUP` or `PLI_EXPERT_TILES_PER_GROUP` (`PLI_PAVES_PAR_GROUPE`) | entier | `2` | `pli/kernels/qwen/dense/v1/simd_qmm.py:14` |  | m1-ultra/qwen3_5 `1` | 8-row tiles per threadgroup in the exact row kernel (1 instead of TensorFold's 2); same bits. *[switches.md : stable]* |
| `expert.v8` | recommande | `--expert.v8` | `PLI_V8` or `PLI_EXPERT_V8` | drapeau | `0` | `pli/kernels/qwen/dense/v1/qmm8.py:20` |  | m1-ultra/qwen3_5 `1` | Kapitsa's 8-row verify GEMM for 4-bit, group-64 weights, for every call of at most 128 rows; row-exact (text can change against simd_qmm). *[switches.md : stable]* |
| `expert.v8_prep2` | recommande | `--expert.v8_prep2` | `PLI_V8_PREP2` or `PLI_EXPERT_V8_PREP2` | drapeau | `0` | `pli/kernels/qwen/dense/v1/qmm8.py:167` |  | m1-ultra/qwen3_5 `1` | The 8-row tables of PLI_V8 prepared over ceil(G / 8) threadgroups per row instead of one; same computation: same bits. *[switches.md : stable]* |
| `expert.xf_bf16` | recommande | `--expert.xf_bf16` | `PLI_XF_BF16` or `PLI_EXPERT_XF_BF16` | drapeau | `0` | `pli/kernels/qwen/dense/v1/simd_qmm.py:21` |  | m1-ultra/qwen3_5 `1` | Input fragments of the 2 to 8-row kernel stored as bf16 pairs instead of fp32 pairs; same bits, half the bytes read. *[switches.md : stable]* |

### JSON keys for generation (gen, gen.voix, gen.image, gen.video, gen.lora)

File passed via `PLI_GEN_CONFIG` (pli-gen.json) and files it names (voix, catalogues). `<nom>`: free name (family, model, voix, LoRA).

| Family | File | Key | Type | Read at | Example from repo | Production |
| --- | --- | --- | --- | --- | --- | --- |
| gen | pli-gen.json | `familles` | objet | `pli/gen/config.py:6`<br>`pli/gen/config.py:46` | oui |  |
| gen | pli-gen.json | `familles.<nom>.argv` | liste | `pli/gen/config.py:6`<br>`pli/gen/config.py:9` | oui |  |
| gen | pli-gen.json | `familles.<nom>.delai_decharge_s` | entier | `pli/gen/config.py:7`<br>`pli/gen/config.py:10` | oui |  |
| gen | pli-gen.json | `familles.<nom>.memoire_go` | entier | `pli/gen/config.py:7`<br>`pli/gen/config.py:10` | oui |  |
| gen | pli-gen.json | `familles.<nom>.port` | entier | `pli/gen/config.py:7`<br>`pli/gen/config.py:53` | oui |  |
| gen | pli-gen.json | `familles.<nom>.relais_delai_s` | entier | `pli/gen/config.py:57` | oui |  |
| gen | pli-gen.json | `familles.<nom>.sante_delai_s` | entier | `pli/gen/config.py:7`<br>`pli/gen/config.py:57` | oui |  |
| gen | pli-gen.json | `reserve_go` | entier | `pli/gen/config.py:5`<br>`pli/gen/config.py:60` | oui |  |
| gen.voix | voix.json | `defaut` | objet | `pli/gen/fils_audio.py:61`<br>`pli/gen/fils_audio.py:571` | oui |  |
| gen.voix | voix.json | `defaut.*` | texte | `pli/gen/fils_audio.py:61`<br>`pli/gen/fils_audio.py:63` | oui |  |
| gen.voix | voix.json | `defaut.drama3` | texte | `pli/gen/fils_audio.py:8`<br>`pli/gen/fils_audio.py:55` | oui |  |
| gen.voix | voix.json | `voix` | objet | `pli/gen/fils_audio.py:571`<br>`pli/gen/fils_audio.py:579` | oui |  |
| gen.voix | voix.json | `voix.<nom>.alias` | liste | `pli/gen/fils_audio.py:62`<br>`pli/gen/fils_audio.py:593` | oui |  |
| gen.voix | voix.json | `voix.<nom>.lang_code` | texte | `pli/gen/fils_audio.py:10`<br>`pli/gen/fils_audio.py:11` | oui |  |
| gen.voix | voix.json | `voix.<nom>.memoire_go` | entier | `pli/gen/config.py:7`<br>`pli/gen/config.py:10` | oui |  |
| gen.voix | voix.json | `voix.<nom>.modele` | texte | `pli/gen/catalogue_prive.py:94`<br>`pli/gen/fils_audio.py:8` | oui |  |
| gen.voix | voix.json | `voix.<nom>.references` | liste | `pli/gen/fils_audio.py:9`<br>`pli/gen/fils_audio.py:11` | oui |  |
| gen.voix | voix.json | `voix.<nom>.voice` | texte | `pli/gen/fils_audio.py:10`<br>`pli/gen/fils_audio.py:11` | oui |  |
| gen.image | catalogue-image.json | `defaut` | texte | `pli/gen/fils_audio.py:61`<br>`pli/gen/fils_audio.py:571` | oui |  |
| gen.image | catalogue-image.json | `modeles` | objet | `pli/gen/fils_image.py:20`<br>`pli/gen/fils_image.py:110` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.cache_limit_go` | entier | `pli/gen/fils_image.py:22`<br>`pli/gen/fils_image.py:492` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.chemin` | texte | `pli/gen/catalogue_prive.py:90`<br>`pli/gen/fils_image.py:20` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.cibles` | liste | `pli/gen/catalogue_prive.py:41`<br>`pli/gen/catalogue_prive.py:92` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.cibles.[].modele` | texte | `pli/gen/catalogue_prive.py:94`<br>`pli/gen/fils_audio.py:8` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.cibles.[].source` | texte | `pli/gen/catalogue_prive.py:94`<br>`pli/gen/fils_image.py:23` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.classe` | texte | `pli/gen/fils_image.py:20`<br>`pli/gen/fils_image.py:116` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.etapes` | entier | `pli/gen/fils_image.py:21`<br>`pli/gen/fils_image.py:29` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.guidance` | reel | `pli/gen/catalogue_prive.py:44`<br>`pli/gen/catalogue_prive.py:117` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.horaire` | texte | `pli/gen/fils_image.py:20`<br>`pli/gen/fils_image.py:29` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.liberer_encodeur` | drapeau | `pli/gen/fils_image.py:22`<br>`pli/gen/fils_image.py:494` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.low_ram` | drapeau | `pli/gen/fils_image.py:22`<br>`pli/gen/fils_image.py:493` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.memoire_go` | entier | `pli/gen/config.py:7`<br>`pli/gen/config.py:10` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.quantize` | entier | `pli/gen/fils_image.py:21`<br>`pli/gen/fils_image.py:127` | oui |  |
| gen.image | catalogue-image.json | `modeles.<nom>.taille` | texte | `pli/gen/catalogue_prive.py:44`<br>`pli/gen/catalogue_prive.py:113` | oui |  |
| gen.video | catalogue-video.json | `cache_fusions` | texte | `pli/gen/fils_video.py:50`<br>`pli/gen/fils_video.py:52` | oui |  |
| gen.video | catalogue-video.json | `cache_go` | entier | `pli/gen/fils_video.py:52`<br>`pli/gen/fils_video.py:69` | oui |  |
| gen.video | catalogue-video.json | `defaut` | texte | `pli/gen/fils_audio.py:61`<br>`pli/gen/fils_audio.py:571` | oui |  |
| gen.video | catalogue-video.json | `dossier_fusions` | texte | `pli/gen/fils_video.py:50`<br>`pli/gen/fils_video.py:51` | oui |  |
| gen.video | catalogue-video.json | `fusion` | objet | `pli/gen/fils_video.py:26`<br>`pli/gen/fils_video.py:62` | oui |  |
| gen.video | catalogue-video.json | `fusion.marge_go` | entier | `pli/gen/fils_video.py:70`<br>`pli/gen/fils_video.py:284` | oui |  |
| gen.video | catalogue-video.json | `fusion.python` | texte | `pli/gen/fils_video.py:61`<br>`pli/gen/fils_video.py:70` | oui |  |
| gen.video | catalogue-video.json | `fusion.script` | texte | `pli/gen/fils_video.py:61`<br>`pli/gen/fils_video.py:70` | oui |  |
| gen.video | catalogue-video.json | `fusion.taille_go` | entier | `pli/gen/catalogue_prive.py:42`<br>`pli/gen/catalogue_prive.py:101` | oui |  |
| gen.video | catalogue-video.json | `fusions` | liste | `pli/gen/fils_video.py:50`<br>`pli/gen/fils_video.py:69` | oui |  |
| gen.video | catalogue-video.json | `h3` | texte | `pli/gen/catalogue_prive.py:19`<br>`pli/gen/catalogue_prive.py:86` | oui |  |
| gen.video | catalogue-video.json | `lora_natif` | objet | `pli/gen/fils_video.py:61`<br>`pli/gen/fils_video.py:271` | oui |  |
| gen.video | catalogue-video.json | `lora_natif.dossier` | texte | `pli/gen/fils_video.py:61`<br>`pli/gen/fils_video.py:286` | oui |  |
| gen.video | catalogue-video.json | `lora_natif.python` | texte | `pli/gen/fils_video.py:61`<br>`pli/gen/fils_video.py:70` | oui |  |
| gen.video | catalogue-video.json | `lora_natif.script` | texte | `pli/gen/fils_video.py:61`<br>`pli/gen/fils_video.py:70` | oui |  |
| gen.video | catalogue-video.json | `modeles` | objet | `pli/gen/fils_image.py:20`<br>`pli/gen/fils_image.py:110` | oui |  |
| gen.video | catalogue-video.json | `modeles.<nom>.chemin` | texte | `pli/gen/catalogue_prive.py:90`<br>`pli/gen/fils_image.py:20` | oui |  |
| gen.video | catalogue-video.json | `modeles.<nom>.memoire_go` | entier | `pli/gen/config.py:7`<br>`pli/gen/config.py:10` | oui |  |
| gen.video | catalogue-video.json | `modeles.<nom>.memoire_go_resident` | entier | `pli/gen/fils_video.py:30`<br>`pli/gen/fils_video.py:66` | oui |  |
| gen.video | catalogue-video.json | `ssd_streaming` | drapeau | `pli/gen/fils_video.py:5`<br>`pli/gen/fils_video.py:65` | oui |  |
| gen.lora | catalogue-image.json | `lora` | objet | `pli/gen/catalogue_prive.py:160`<br>`pli/gen/catalogue_prive.py:167` | oui |  |
| gen.lora | catalogue-image.json | `lora.<nom>.base` | liste | `pli/gen/catalogue_prive.py:41`<br>`pli/gen/catalogue_prive.py:42` | oui |  |
| gen.lora | catalogue-image.json | `lora.<nom>.chemin` | texte | `pli/gen/catalogue_prive.py:90`<br>`pli/gen/fils_image.py:20` | oui |  |
| gen.lora | catalogue-image.json | `lora.<nom>.cibles` | liste | `pli/gen/catalogue_prive.py:41`<br>`pli/gen/catalogue_prive.py:92` | oui |  |
| gen.lora | catalogue-image.json | `lora.<nom>.cibles.[].modele` | texte | `pli/gen/catalogue_prive.py:94`<br>`pli/gen/fils_audio.py:8` | oui |  |
| gen.lora | catalogue-image.json | `lora.<nom>.cibles.[].source` | texte | `pli/gen/catalogue_prive.py:94`<br>`pli/gen/fils_image.py:23` | oui |  |
| gen.lora | catalogue-image.json | `lora.<nom>.etapes` | entier | `pli/gen/fils_image.py:21`<br>`pli/gen/fils_image.py:29` | oui |  |
| gen.lora | catalogue-image.json | `lora.<nom>.force` | reel | `pli/gen/catalogue_prive.py:41`<br>`pli/gen/catalogue_prive.py:42` | oui |  |
| gen.lora | catalogue-image.json | `lora.<nom>.guidance` | reel | `pli/gen/catalogue_prive.py:44`<br>`pli/gen/catalogue_prive.py:117` | oui |  |
| gen.lora | catalogue-image.json | `lora.<nom>.horaire` | texte | `pli/gen/fils_image.py:20`<br>`pli/gen/fils_image.py:29` | oui |  |
| gen.lora | catalogue-video.json | `lora` | objet | `pli/gen/catalogue_prive.py:160`<br>`pli/gen/catalogue_prive.py:167` | oui |  |
| gen.lora | catalogue-video.json | `lora.<nom>.base` | liste | `pli/gen/catalogue_prive.py:41`<br>`pli/gen/catalogue_prive.py:42` | oui |  |
| gen.lora | catalogue-video.json | `lora.<nom>.chemin` | texte | `pli/gen/catalogue_prive.py:90`<br>`pli/gen/fils_image.py:20` | oui |  |
| gen.lora | catalogue-video.json | `lora.<nom>.force` | reel | `pli/gen/catalogue_prive.py:41`<br>`pli/gen/catalogue_prive.py:42` | oui |  |
| gen.lora | catalogue-video.json | `lora.<nom>.mode` | texte | `pli/gen/catalogue_prive.py:42`<br>`pli/gen/catalogue_prive.py:98` | oui |  |
| gen.lora | catalogue-video.json | `lora.<nom>.steps` | entier | `pli/gen/fils_image.py:243`<br>`pli/gen/fils_image.py:244` | oui |  |

### Statuts

| Status | Settings | Of which deployed in reference production |
| --- | --: | --: |
| recommande | 123 | 93 |
| experimental | 176 | 0 |
| obsolete | 3 | 0 |

Doubtful cases (marked « (?) » in the tables): proposed status, to be confirmed.

- `modele.tool_grammar` (`PLI_TOOL_GRAMMAR`): proposed **recommande**; in production, but docs/switches.md classes it « reglage-fin ».
- `brouillons.falcon_tokens` (`PLI_FALCON_TOKENS`): proposed **recommande**; in production, but docs/switches.md classes it « experimental ».
- `brouillons.falcon_tokens_days` (`PLI_FALCON_TOKENS_DAYS`): proposed **recommande**; in production, but docs/switches.md classes it « experimental ».
- `cache.disk_reply_end` (`PLI_DISK_REPLY_END`): proposed **recommande**; in production, but docs/switches.md classes it « reglage-fin ».
- `cache.spill_background` (`PLI_SPILL_BACKGROUND`): proposed **recommande**; in production, but docs/switches.md classes it « reglage-fin ».
- `parallele.admit_by_work` (`PLI_ADMIT_BY_WORK`): proposed **recommande**; in production, but docs/switches.md classes it « experimental ».
- `attention.attec` (`PLI_ATTENTION_ATTEC`): proposed **recommande**; in production, but only as a fallback (another setting is used first).
- `attention.s3b_rows` (`PLI_S3B_ROWS`): proposed **recommande**; in production, but docs/switches.md classes it « reglage-fin ».
- `gen.garde_journal` (`PLI_GEN_GARDE_JOURNAL`): proposed **recommande**; in production, but only used for logging or capturing (no effect on computation).
- `journal.checkpoint_log` (`PLI_CHECKPOINT_LOG`): proposed **recommande**; in production, but docs/switches.md classes it « experimental ».
- `journal.copy_log` (`PLI_COPY_LOG`): proposed **recommande**; in production, but docs/switches.md classes it « experimental ».
- `journal.events` (`PLI_EVENTS`): proposed **recommande**; in production, but only used for logging or capturing (no effect on computation).
- `journal.log_times` (`PLI_LOG_TIMES`): proposed **recommande**; in production, but only used for logging or capturing (no effect on computation).
- `journal.trace_dir` (`PLI_TRACE_DIR`): proposed **recommande**; in production, but only used for logging or capturing (no effect on computation).

### Dead, duplicate or inconsistent settings (automatic survey)

- `memoire.mlx_max_mb_per_buffer` (`MLX_MAX_MB_PER_BUFFER`) : never read directly in src/ (read by MLX).
- `memoire.mlx_max_ops_per_buffer` (`MLX_MAX_OPS_PER_BUFFER`) : never read directly in src/ (read by MLX).
- `taches.prune_cap_gb` (`PLI_TASKS_PRUNE_CAP_GB`) : different defaults depending on the location: '', '60.0'.
- `reseau.test_launchd` (`PLI_TEST_LAUNCHD`) : lu seulement hors de src/ : control/test_macos_smoke.py:19.
- `journal.log_times` (`PLI_LOG_TIMES`) : different defaults depending on the location: '', '0'.

<!-- fin inventaire -->
