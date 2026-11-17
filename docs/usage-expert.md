# PLI : usage expert

Everything that launch profiles configure, and what they do not. Short version:
[utilisation-rapide.md](utilisation-rapide.md). Each `PLI_*` engine setting is described in
[switches.md](switches.md); here, what matters for launching PLI 5 and what each choice costs in memory and disk.

> **Licenses, read before use.** Qwen-Image 2.1 and the Lightning v2 LoRA: non-commercial use only
> (Qwen Research License). Lightning v1: no license declared on its page. MiniMax-H3 and the Turbo LoRA:
> MiniMax community license, which excludes the European Union, the United Kingdom, South Korea, and the United States
> ("Excluded Territories"): it does not authorize use from these countries. Details and links:
> [§8](#8-model-sources-and-licenses).

"Measured" means measured on the validated machine (Mac Studio M1 Ultra, 64 GB) or in a cited recipe;
"estimated" or "not measured" is written when that is not the case. The `file:line` references are those of the
commit that updates this page.

## 1. Deux couches de profils

| Layer | Where | What it carries | How it is chosen |
| --- | --- | --- | --- |
| Launch profile | `templates/launch-profiles/NOM.json` | model, `pli serve` options, generation families, requested memory and disk | `tools/pli-profil NOM` |
| Machine profile | `src/pli/additions/profiles/NOM/*.env` | `PLI_*` settings measured for a chip (currently `m1-ultra`) | `pli serve --profile NOM` |

A launch profile designates its machine profile via `"machine_profile"`: `"auto"` (default in examples: `m1-ultra`
for an Ultra chip with 64 GB or more, the rule in `pli setup`, `src/pli/install/machine.py:83`; otherwise
none), `"none"`, or a folder name from `src/pli/additions/profiles/`. An unknown name is refused. The two layers
remain separate: a machine profile does not know about use cases, a launch profile does not copy the chip's settings.


## 2. The launcher `tools/pli-profil`

| Commande | Effet |
| --- | --- |
| `pli-profil` | list of launch profiles, memory and disk of each |
| `pli-profil NOM --dry-run` | displays the command, variables, files to write, checks; does not launch or write anything |
| `pli-profil NOM --plan-json` | the same plan in JSON (scripts, tests) |
| `pli-profil NOM` | verifies, writes generation configurations, then replaces its process with `pli serve` |
| `pli-profil NOM -- OPTIONS` | adds options to `pli serve` last: for argparse, the last value wins |
| `--dossier DOSSIER` | your own launch profiles (also `PLI_PROFILS_LANCEMENT`); default: `templates/launch-profiles` |
| `--machine FICHIER` | variables file (default: `machine.env` from the profiles folder, otherwise `~/.config/pli/machine.env`) |
| `--forcer` | launches despite a refusal (memory, disk, missing file) |
| `--simuler-ram-go`, `--simuler-disque-go`, `--simuler-puce` | tests without Mac or model |

Order of variables: launcher defaults (`DEFAUTS` at the top of the script), then the machine file, then
the environment, which always takes precedence. Commented list: `templates/launch-profiles/machine.exemple.env`.

Checks, in order:

1. Memory: `sysctl -n hw.memsize` against `memoire.ram_min_go` of the profile (total memory, not free memory).
2. Disk: free space (`df` of the `PLI_DONNEES` folder) against the sum of missing files and the expected
   growth (snapshots, outputs, merges). A file already present is not counted; a Hugging Face repository is looked up
   in `HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`. Models stored on another disk are
   not checked separately.
3. Required files (image and video models, `h3`, LoRA, merge tool), Python of each family, `ffmpeg` for
   video. Exit code 2 and message in case of refusal.

At launch, the launcher writes `PLI_DONNEES/profils/NOM/pli-gen.json` and the catalogs, paths replaced, then
sets `PLI_GEN=1` and `PLI_GEN_CONFIG`. Tests: `tests/test_pli_profil.py` (19 tests, no model; one launches a fake
`pli`).

### Format of a launch profile

| Key | Content |
| --- | --- |
| `modele`, `arguments`, `environnement` | model, `pli serve` options, variables; `${VAR}` replaced |
| `machine_profile` | `auto`, `none` or a machine profile name (see 1) |
| `conditions` | blocks added only if everything fits: `si_profil_machine` (the selected machine profile), `si_ram_min_go`, `si_fichier`, `si_module` (importable by `pli`'s Python); `effet` and `sinon` are displayed |
| `generation` | `config` (content of `PLI_GEN_CONFIG`) and `catalogues` (models from `catalogues/` files) |
| `memoire` | `ram_min_go`, `statut` (measured or estimated), `detail` (the calculation), `si_trop_petit` (refusal message) |
| `disque` | lines `quoi`, `go`, `statut`, `chemin` (missing: always counted) |
| `fichiers_requis`, `dossiers` | paths to find before launch, subfolders of `PLI_DONNEES` to create |

Add a profile: copy a `.json` file into your folder (`--dossier`), change the name and values, then
`pli-profil NOUVEAU --dossier DOSSIER --dry-run`.

## 3. `pli serve` options used by profiles

All options: `pli serve --help` (`src/pli/cli_args.py:32-123`).

| Option | Default | Profiles | Memory, disk |
| --- | --- | --- | --- |
| `--profile` | none | set by `machine_profile` | settings measured on M1 Ultra (`src/pli/additions/profiles/m1-ultra/`) |
| `--context N` | depending on model and memory | 229,376 with `m1-ultra`; 32,768 on small machine | KV cache grows with the window; a window that does not fit is refused at startup (RUNBOOK) |
| `--parallel N` | auto (up to 8, depending on memory) | 4 with `m1-ultra`; 1 on small machines | each ongoing conversation has its own cache |
| `--drafter` | auto | auto; none on small machine under 32 GB | DFlash2: 3.6 GB on disk; with it Bonsai requires about 20 GiB of budget (recipe ternary-bonsai-2.md:40) |
| `--no-thinking` | thinking enabled | 27B profiles | a request can re-enable it (`chat_template_kwargs`) |
| `--decode-share 1` | 0.25 | 27B profiles (production value) | no known memory cost |
| `--snapshot-dir`, `--spill-gib` | `~/.cache/tensorfold/prefix-snapshots`, 0 | `PLI_DONNEES/instantanes/...`, 40 (10 on small machines) | disk: at most `--spill-gib` Gio |
| `--startup-cache` | `off` | `on` | start in about 9 s instead of about 70 s with the same weights, code and settings; `remeasure` once on a quiet machine |
| `--prompt-cache-gib`, `--checkpoint-slots`, `--mlx-cache-gib` | see `--help` | unchanged | memory for prefixes and the MLX cache |
| `--vision`, `--api-key`, `--host` | off, no key, 127.0.0.1 | not set | `--vision` requires `pip install 'pli[vision]'`; on the local network, set `--api-key` |

Process memory budget: 70% of memory by default, including 3 Gio outside the allocator; `PLI_MEMORY_LIMIT_GB`
changes this (RUNBOOK.md:82-83).

## 4. Important variables

Set by the 27B profiles with the machine profile `m1-ultra` (production of the validated machine):
`MLX_MAX_MB_PER_BUFFER=500` and `MLX_MAX_OPS_PER_BUFFER=100` (read by MLX at startup, so in the environment, not
in a machine profile), `PLI_ADMIT_BY_WORK=1`, `PLI_ADMIT_SMALL=0`, `PLI_SPILL_BACKGROUND=1`, `PLI_DISK_REPLY_END=1`.
Depending on files: `PLI_HYBRID=1` and `PLI_MTP_HEAD` if the MTP head exists; `PLI_A1_LSA_MAP`,
`PLI_S3B_THRESHOLD=16384`, `PLI_S3B_ROWS=4096` if the A1 card exists (with `m1-ultra` only); `PLI_TOOL_GRAMMAR=1`
if `xgrammar` is installed (`pip install 'pli[grammar]'`, which brings torch). Always: `PLI_EVENTS` (log without
message content) and `PLI_NO_UPDATE_CHECK=1`.

PLI 5, generation (read by the code in `src/pli/gen/`):

| Variable | Default | Role |
| --- | --- | --- |
| `PLI_GEN`, `PLI_GEN_CONFIG` | off | `1` and the JSON file of families: audio, image, video routes (`server/http.py:95`) |
| `PLI_GEN_RESERVE_GO` | 6 | memory kept free in addition to a family's need (`gen/superviseur.py:27`, 184) |
| `PLI_GEN_MARGE_GO`, `PLI_GEN_CACHE_GO` | 2, 2 | margin of the 27B when it returns memory, MLX cache it keeps during a child process (`gen/memoire_general.py:172-173`) |
| `PLI_GEN_VIDEO_AVEC_GENERAL` | off | `1`: video is never warned that the 27B is loaded (`gen/routes.py:57`) |
| `PLI_GEN_GARDE_NOMS`, `PLI_GEN_GARDE_JOURNAL` | none | list of real people's names refused, JSONL log of refusals (`gen/garde.py:13`) |
| `PLI_GEN_GARDE_LLM`, `PLI_GEN_GARDE_LLM_DELAI_S` | off, 5 s | second opinion of the 27B on an image or video request (`gen/garde.py:50`) |
| `PLI_GEN_CATALOGUE_PRIVE` | none | private LoRA TOML files, separated by `:`, read by image and video child processes (section 5, "Private LoRA Catalog"); preferably given via the `catalogue_prive` key of `PLI_GEN_CONFIG`, which does not change the checkpoint key |
| `PLI_GEN_SORTIES`, `PLI_GEN_PIDS` | none | images folder if `--sorties` is missing; folder of child process files |
| `H3_FFMPEG` | none | ffmpeg path for h3 (the launcher sets it if ffmpeg is not in PATH) |

## 5. Familles PLI 5

The server launches each family on the first request, in a separate process ("child"), and stops it after
`delai_decharge_s` without request (audio 600 s, image 300 s, video 0: as soon as the queue is empty).

| Family | Child and Python | Routes | Profile models |
| --- | --- | --- | --- |
| audio | `gen/fils_audio.py`, mlx-audio environment (0.6.5 tested) | `/v1/audio/speech`, `/v1/audio/transcriptions` | Kokoro (voice `kokoro-siwis`, French), Qwen3-ASR 0.6B; Drama 3 and Qwen3-TTS possible (voice catalog) |

Voices: the code only knows Kokoro (`kokoro-siwis`). A voice catalog (`--voix FICHIER`) adds its own and
chooses defaults by its `defaut` key: `{"drama3": NOM, "kokoro": NOM, "*": NOM}`, where `*` is used when the request does not
named, nor a known synthesis voice or model. A voice can keep its old names: `"alias": ["OLD"]`. Without a default
Drama 3, a request `"model": "drama3"` without `voice` is refused (400). The Drama 3 default is optional: a catalog that does
it does not provide serves Kokoro for voiceless requests, and a renamed voice remains served under its old name by `alias`. `GET /sante` of the child process gives the voices and the
defaults. Example: `docs/pli-gen-voix.exemple.json`.

| Family | Child process and Python | Routes | Profile models |
| --- | --- | --- | --- |
| image | `gen/fils_image.py`, mflux environment (0.24.0 tested) | `/v1/images/generations` (`/v1/images/edits`: 501) | Qwen-Image 2.1; qwen2512, flux3klein, krea2 possible |
| video | `gen/fils_video.py`, server Python, `h3` program | `/v1/videos`, `GET /v1/videos/{id}`, `/content`, `DELETE` | MiniMax-H3, read from SSD (`ssd_streaming`) |

Keys of the `PLI_GEN_CONFIG` file (`gen/config.py`): `reserve_go`, `seuil_echange_mo` (default 64 MB, production
4,096), `echange_dispo_min_go` (production 4), the `video_*` keys for the voice's place during a video (absent
from profiles: previous behavior), and per family `argv`, `port`, `memoire_go`, `delai_decharge_s`, `sante_delai_s`,
`relais_delai_s`, `reserve_go`, `surveiller_echange`. A family's `memoire_go` is the need of its smallest useful model;
the child process then controls the need of each model in its catalog. Commented examples: `docs/gen/` and
`docs/pli-gen*.exemple.json` (paths `/chemin/pli` to replace).

### Memory before a launch

The supervisor requests `memoire_go` of the family plus the reserve (`gen/superviseur.py:418`). If there is not enough
space, it stops inactive families, then asks the 27B to free memory (MLX cache emptied, checkpoints
already written to disk evicted, `gen/memoire_general.py:55`), then refuses with 503 `memoire`. The 27B reclaims its
memory when the last child process stops (`memoire_general.py:75`).

### Preemption

The 27B always goes first. While it is working, a flag file exists: the image waits between two
denoising steps, the voice between two pieces, the video is paused (SIGSTOP of the h3 group, SIGCONT
then). Only one video at a time; it also waits for the end of an image or a voice. Cancellation:
`DELETE /v1/videos/{id}` for a video; an image or a voice whose client closes the connection is
cancelled (499 response from the child process), a video continues.

### Garde

Image and video requests pass a local filter before the child process: minors (words and ages under 18, French
and English), names of real people (`PLI_GEN_GARDE_NOMS`, heuristic "photo of Firstname Lastname"), optional second opinion
from the 27B. Refusal: 400 of type `garde`, without the reason; the log keeps only the reason and a fingerprint.
Audio is not filtered.

### LoRA

Image: one or more LoRA per request (`"lora":[{"nom":"lightning","force":1.0}]`), applied at loading;
changing LoRA reloads the model. Measured cost: 0.2 GB more (krea2: 28.2 GB alone, 28.4 with a LoRA). mflux
0.24 reads the `lora_A` / `lora_B` keys; a LoRA published as `lora.down` / `lora.up` with `alpha`, or as PEFT with
alpha in its metadata (Lightning case), is first converted with `tools/pli-lora-image`:

```bash
tools/pli-lora-image INPUT.safetensors OUTPUT.safetensors --dry-run   # keys and scales, nothing written
tools/pli-lora-image INPUT.safetensors OUTPUT.safetensors           # writes OUTPUT (refused if it exists)
```

The alpha / rank scale is folded into `lora_B` (the output has no alpha); F32, F16 and BF16 are kept; any
unknown key, rslora or `alpha_pattern` is refused without writing anything (code 2). On Lightning v1, the output is
byte-for-byte identical to the converted file served in production. Tests: `tests/test_pli_lora_image.py`.

Video: three modes per LoRA (key `mode` of the video catalog, or of the private catalog).

| Mode | How | Where it goes | When it disappears |
| --- | --- | --- | --- |
| `en_memoire` (5.6.5) | h3 adds the LoRA to each block's weights as it loads them; up to 4 LoRA at once | `lora_natif.dossier`: one small file per LoRA (73 MB to 1.9 GB) | never (to be deleted manually) |
| `sauvegardee` | fused copy of the transformer, one LoRA | `dossier_fusions` | never (to be deleted manually) |
| `a_la_demande` | fused copy of the transformer, one LoRA | `cache_fusions`, capped at `cache_go` (default 150, profiles 70) | least recently used goes first |

In memory (`en_memoire`):

- **h3 with the patch**: h3.c needs `tools/h3c-lora.patch` (option `--lora FILE@FORCE`, repeatable). On h3.c 8974cc0:
  `git apply /path/to/pli/tools/h3c-lora.patch && make`. Point the catalog's `h3` key to that binary.
- **Conversion**: `tools/pli-lora-natif MODELE LORA OUT.safetensors` writes the LoRA in h3's native layout (same key
  reading as `pli-lora-fusion`: native, Kohya, Comfy, PEFT diffusers). The video child process runs it itself at the
  first request (catalog key `lora_natif`: `dossier`, and `python`, `script` if they are not the fusion's), in a few
  seconds. A LoRA that touches anything other than the blocks' four BF16 matrices (for instance `audio_patch_proj`) is
  refused: keep it in a fusion mode.
- **Same video**: on a Turbo video (512², 22 frames, 8 steps, same seed), the MP4 rendered with the LoRA in memory is
  byte-for-byte identical to the one rendered from the fused copy. The weights get the same single rounding to BF16.
- **Cost**: no measurable time cost (148 s in memory, 195 s from the fusion, with other work running); memory: the LoRA
  files and a 0.6 GB buffer, counted by the child process before it starts a job. No copy of 33 to 62 GB per LoRA.
- **Several at once**: `"lora": [{"nom": "turbo"}, {"nom": "style", "force": 0.7}]`. All must be `en_memoire`; a
  fused LoRA stays alone. The defaults (`steps`, `size`, `seconds`) come from the first LoRA of the list that has them.

Fused copies: an already done fusion (same model, same file, same strength) is reused without recalculation: it
carries a `fusion_lora_h3.json`. Disk space checked before fusion (507 `disque`): `taille_go` of the LoRA or of
`fusion`, plus `marge_go`. Measured: 33 to 62 GB per fused LoRA depending on the LoRA. Fusion memory: not measured.
A profile cannot impose a default LoRA: it is the request that chooses it.

### Private LoRA catalog

LoRAs that should not live in the repository or in JSON catalogs (private use) are described in one or
more TOML files outside the repository, named by the `"catalogue_prive"` key in the `PLI_GEN_CONFIG` file (path or list
of paths). Absent: nothing changes. The key is passed to child processes only (variable `PLI_GEN_CATALOGUE_PRIVE` in their
environment). Do not put this variable in the server environment (service `.env` file): like all
`PLI_*`, it would enter the fingerprint of the checkpoint key settings, and saved conversations would
no longer be resumed. Both at the same time are refused. Code: `src/pli/gen/catalogue_prive.py`; tests: `tests/test_gen_catalogue_prive.py`.

```toml
[image.mon-lora]
fichier = "/chemin/mon-lora.safetensors"     # obligatoire
base = "krea2"                               # model(s) from the image catalog (text or list), required
force = 1.0
cibles = [{source = "img_mod.1", modele = "img_mod_linear"}]   # if needed (qwen21, qwen2512)
defauts = {passes = 8, taille = "1024x1024", guidance = 1.0}

[video.mon-lora-video]
fichier = "/chemin/lora-video.safetensors"
base = "h3"                                  # default: h3
mode = "en_memoire"                          # or sauvegardee, or a_la_demande (default)
taille_go = 62                               # disk space for a new fusion (fusion modes)
defauts = {passes = 8, taille = "512x512", secondes = 2}
```

- Entries join their family's catalog when the child process starts; the server also checks them at
  startup and refuses to start if a name already exists in the public catalog or in two private files (never
  silent override), if a key is unknown or a value is out of bounds. The server log only shows the
  number of private LoRAs per family, not their names.
- `defauts`: values used when a request names this LoRA without providing them (`steps`, `size`, `guidance` for
  images; `steps`, `size`, `seconds` for video). The request always takes precedence. A public JSON catalog can
  carry the same `defauts` key on its LoRAs. The schedule shift is not adjustable in PLI today.
- The guard (refusal of minors and real people) applies to requests as before.
- Python 3.11 or newer (`tomllib`) in the PLI venv and in the mflux one; otherwise the `tomli` package.

`pli-lora-fusion` manually:

```bash
tools/pli-lora-fusion MODELE SORTIE --lora FICHIER@1.0 --essai     # checks keys and shapes, writes nothing
tools/pli-lora-fusion MODELE SORTIE --lora FICHIER@1.0             # writes SORTIE (missing or empty directory)
```

It reads native LoRAs, Comfy (`diffusion_model.*`, `lora_unet_*`, qkv reordered by head) and PEFT diffusers
(`transformer_blocks.*`, alpha in metadata), and LoKr; any other name is refused without writing anything (code 2).
Only the touched transformer files are rewritten, the rest are links to MODELE. Tests:
`tests/test_pli_lora_fusion.py` (tiny fake model).

## 6. Memory and disk per model

| Model | Memory | Disk (GB) | Status |
| --- | ---: | ---: | --- |
| Qwen3.8-27B 4-bit + DFlash2 | approx 36 GB wired in service; peak 43.7 GiB at 140,288 tokens | 15 + 3.6 | measured (README; recipe qwen3.8-27b.md, M5 Pro 64 GB, TensorFold 0.3.5.1) |
| Qwen3.8-27B without drafter | peak 39.9 GiB at 152,576 tokens | 15 | measured (same recipe) |
| MTP head of the 27B | not measured separately | 0.85 (bf16) + 0.23 (4-bit) | disk measured |
| Ternary Bonsai 2 27B 2 bits | codes 7.2 Gio, 13.9 Gio expanded on M1 to M4; approximately 20 Gio budget with drafter | 8.6 | TensorFold measurements, not measured under PLI |
| Kokoro 82M | 3 GB | 0.34 | measured (`MEMOIRE_GO_DEFAUT` in `gen/fils_audio.py`) |
| Drama 3 4-bit | 24 GB (22.4 after loading, peak 25.8 at 27.5 during a voice) | 22 | measured (commit b4ee4a8); requires a reference voice (WAV and text) |
| Qwen3-TTS 1.7B | 7 GB | 4.2 | measured |
| Qwen3-ASR 0.6B 8-bit | 1 to 1.6 GB (measured addition next to Drama 3) | 0.96 | measured (commit b4ee4a8) |
| Qwen-Image 2.1 mflux 8-bit | 26 GB (peak) | 17 | measured (`gen/fils_image.py:69`) |
| Qwen-Image 2512 q6 | 34 GB | 29 | measured |
| FLUX 3 klein | 16 GB | 19 | measured |
| Krea 2 Turbo (quantize 8) | 29 GB (28.2 measured) | 33 | measured |
| MiniMax-H3, SSD streaming | 12 GB (10.5 to 10.9 measured at 512x512) | 134 | measured (`gen/fils_video.py:117`) |
| MiniMax-H3, all in memory | 43 GB (39 to 43 measured) | 134 | measured; refused next to the 27B |
| Video LoRA fusion | not measured | 33 to 62 | measured on disk |
| Python environments: PLI, mlx-audio, mflux | | 1.6; 0.7; 1.4 | measured |

Disk: `du` on the validated machine. Memory: peak process footprint. A model file is not
all of its memory: the conversation cache is added to it (RUNBOOK).

## 7. Memory and disk per profile

Calculation: 27B in service (36 GB) + the largest family loaded alongside (its need + its reserve). The supervisor
stops families that are not in use; when the calculation exceeds memory, the 27B releases memory before
launching the child process (section 5).

| Profile | Mac Memory | Calculation | Disk | Status |
| --- | ---: | --- | ---: | --- |
| `code` | 48 GB | 27B alone | 63 GB | validated on 64 GB; 48 GB is the threshold of `pli setup` (`install/machine.py:78`), not measured |
| `assistant-vocal` | 48 GB | 36 + Kokoro 3 + transcription 1.6 + reserve 6 = 46.6 GB | 65 GB | estimated; validated on 64 GB |
| `creation` | 64 GB | 36 + image 26 + reserve 4 = 66 GB (the 27B returns the difference); video 36 + 12 + 6 = 54 GB | 222 GB | measured in production |
| `creation-lora` | 64 GB | like `creation`, + 0.2 GB per image LoRA; video fusion not measured | 290 GB | measured, except fusion; 62 GB per fused video LoRA included |
| `tout-inclus` | 64 GB | largest family: image, 66 GB; with Drama 3 added to the voice catalog: 36 + 24 + 1.6 + 6 = 67.6 GB | 224 GB | measured in production |
| `petite-machine` | 24 GB | Bonsai 2 bits, budget 16.8 GiB (70% of 24) | 24 GB | estimated |

`petite-machine` targets Macs with 24 to 32 GB. 24 GB Macs are M2, M3, M4 and M4 Pro: no M1 has 24 GB
(M1: 8 or 16; M1 Pro: 16 or 32; M1 Max: 32 or 64; M1 Ultra: 64 or 128). From 32 GB, the profile adds the
DFlash2 drafter. Below 24 GB, no PLI model fits: the smallest family is a 27B.

Disk counts the whole profile, including the snapshot cap (40 GB, 10 on petite-machine), and `pli-profil` only
requests what is missing.

## 8. Model sources and licenses

PLI provides no weights. Each model keeps its license: read it before use, especially for commercial use.
"Conditional access": Hugging Face requires accepting the terms before downloading.

| Model | Original page | License (repository card) |
| --- | --- | --- |
| Qwen3.8-27B 4 bits | [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | Apache-2.0 |
| DFlash2 | [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) | Apache-2.0 |
| MTP Head | [EigenLabs/Qwen3.8-27B-MTP-bf16](https://huggingface.co/EigenLabs/Qwen3.8-27B-MTP-bf16) | Apache-2.0 |
| Ternary Bonsai 2 27B | [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) | Apache-2.0 |
| Kokoro 82M | [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16) | Apache-2.0 |
| Qwen3-ASR 0.6B | [mlx-community/Qwen3-ASR-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit) | Apache-2.0 |
| Drama 3 4 bits | [mlx-community/fish-audio-drama-3-4bit](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit) | Fish Audio Community License |
| Qwen3-TTS 1.7B | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Apache-2.0 |
| Qwen-Image 2.1, mflux 8 bits | [OsaurusAI/Qwen-Image-2.1-mflux-8bit](https://huggingface.co/OsaurusAI/Qwen-Image-2.1-mflux-8bit), quantized from [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | Qwen Research License: non-commercial use only (§2 a and b), commercial license to request from Qwen |
| Qwen-Image 2512, mflux q6 | [mflux-community/qwen-image-2512-mflux-q6](https://huggingface.co/mflux-community/qwen-image-2512-mflux-q6), depuis [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | Apache-2.0 |
| FLUX 3 Image Klein | [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) | FLUX Kommunity License, conditional access |
| Krea 2 Turbo | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) | Krea 2 Community License, conditional access |
| MiniMax-H3 | [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (498 GB total; h3 reads the `FL2VA` folder, 134 GB with root files) | MiniMax-H3 Community License Agreement: limited to the "Applicable Territory", which excludes the European Union, the United Kingdom, South Korea and the United States (definitions 3 and 5); use outside this territory not authorized (IV.4); beyond $20M annual revenue, written authorization (IV.1) |
| h3 (programme) | [antirez/h3.c](https://github.com/antirez/h3.c) | MIT |
| LoRA Lightning (image, 4 steps) | [lightx2v/Qwen-Image-2.1-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning) (v1); [lightx2v/Qwen-Image-2.1-Lightning-v2-RL](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning-v2-RL), file `qwen_image_21_lightning_v2_rl_4steps_lora_rank64.safetensors` (v2, profile `creation-lora`) | v1: no license declared on the card; v2: Qwen Research License (link to Qwen-Image 2.1's) |
| LoRA Turbo (video, 8 steps) | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo), file `minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors` | MiniMax-H3 Community License Agreement (same excluded territories) |

For this table, only §2 of the Qwen Research License and the cited sections of the MiniMax-H3 license were
read; the other licenses are named after the repository sheet, without a summary of their terms.

## 9. Rollback

- Switch profile: stop PLI (a single Ctrl-C or a single SIGTERM: a second signal cuts off snapshot writing),
  then start the other profile.
- Stop generation: start the `code` profile, or remove `PLI_GEN`; without `PLI_GEN=1`, no generation route
  exists and PLI behaves as before PLI 5 (`server/http.py:95`).
- Revert to an option: `pli-profil NOM -- --option value` (the last value wins), or an environment variable,
  which overrides the profile and `machine.env`.
- Free up disk: delete `PLI_DONNEES/video/fusions` (saved fusions) and `PLI_DONNEES/instantanes`.

## 10. Known pitfalls

| Pitfall | Effect | Fixed by |
| --- | --- | --- |
| h3 shaders read via a relative path (`h3_shaders.metal`) | "cannot compile h3_shaders.metal" | the child process launches h3 from its folder (`gen/fils_video.py:179`): keep `h3` in its compilation folder |
| ffmpeg outside PATH (launchd service, ssh) | « cannot start FFmpeg » | `H3_FFMPEG` set by the child process (`/opt/homebrew/bin`, `fils_video.py:183`) and by the launcher (also `/usr/local/bin`) |
| Swap guard: on first loading of Drama 3 after 27B, macOS sends 3 to 4.8 GB of cold pages to swap | voice child process stopped by mistake (502) | `seuil_echange_mo` 4,096 and `echange_dispo_min_go` 4: available memory must also be low |
| Two stop signals | snapshots cut off mid-write | a single Ctrl-C or SIGTERM, then wait |
| `MLX_MAX_*` in a machine profile | no effect: MLX reads them before | in the environment (the launcher sets them) |
| Video without SSD streaming next to 27B | 503 `general_charge` | `ssd_streaming: true` in the catalog (default of profiles) |
| Image LoRA in `lora.down` / `lora.up` or PEFT format | not read by mflux 0.24 | `tools/pli-lora-image` (section 5) |

## 11. Settings by family (`pli.toml`, `pli config`)

Design, complete inventory (268 settings: 39 options, 228 variables, 1 derived setting, and the JSON keys
of generation) and
dead or duplicate settings: [reglages.md](reglages.md). Code: `src/pli/config/`, table
`src/pli/config/table.json` produced by `tools/pli-reglages-inventaire`.

### Noms

| Forme | Exemple | Remarque |
| --- | --- | --- |
| Key | `arbres.deltanet` | stable; 17 families: modele, brouillons, cache, memoire, parallele, lecture, attention, arbres, ane, gen, gen.voix, gen.image, gen.video, gen.lora, reseau, journal, expert |
| `pli.toml` line | `[arbres]` then `deltanet = 3` | one section per family |
| Option | `--arbres.deltanet 3` or `--arbres.deltanet=3` | for any setting; a flag alone means true (`--ane.ffn`); existing options (`--context`) remain |
| Code variable | `PLI_TREE_DELTANET` | the name read by the code, unchanged |
| Variable derived from the key | `PLI_ARBRES_DELTANET` | accepted in addition; for an option: `PLI_CACHE_CONTEXT` gives the default of `--context` |
| Old name | `PLI_ARBRE_DN` | still accepted; `pli serve` lists old names set in one line at startup |

Naming exceptions: `PLI_MTP_HEAD`, `PLI_MTP_DRAFTS`, `PLI_MTP_WINDOW` (MTP head of the 27B) become
`brouillons.mtp_head`, `brouillons.mtp_head_drafts`, `brouillons.mtp_head_window`, because `--mtp-drafts`
(`brouillons.mtp_drafts`) is another setting. `PLI_PROFILE` and `PLI_NO_UPDATE_CHECK`, duplicates of `--profile` and
`--no-update-check`, are `modele.profile_env` and `reseau.no_update_check_env`. `PLI_API_KEY` (bench client) is
`reseau.bench_api_key`.

### Priority

option > variable > `pli.toml` > machine profile (`--profile`) > code default.

`pli.toml` is only read if named (`--config FILE` or `PLI_CONFIG=FILE`). It is applied in `cli.main`,
before the profile and before any import of MLX or a kernel: it acts like a variable set in the shell, including
for `MLX_MAX_MB_PER_BUFFER` and `MLX_MAX_OPS_PER_BUFFER` (`[memoire]`). `PLI_CONFIG` and variables derived from keys
are removed from the environment once read: the checkpoint key does not see them, it only sees the
usual variables. The code in `src/pli/config/` is not included in the `|code=` hash.

### Commandes

| Command | Role | Options |
| --- | --- | --- |
| `pli config:list [family]` | key, status, value, source | `--poses` (only what is set), `--statut recommande` (or `experimental`, `obsolete`), `--profile m1-ultra --famille-modele qwen3_5` (count the profile) |
| `pli config:get cle` | the value only | accepts a variable or an old name |
| `pli config:set cle valeur` | writes to `pli.toml` after verification; keeps comments | `--retirer`; warns if a set variable will win |
| `pli config:check` | types and values of the file and environment; unknown names (error), old names (warning) | `--sans-env`; return code 1 if there is an error |
| `pli config:explain cle` | help, type, default, `file:line` where the code reads it, accepted names, value and source | |
| `pli config:import-env A.env [B.env]` | translates `.env` files (read as `set -a; . A; . B`) to `pli.toml` | `--options "--context 229376 ..."`, `-o FILE` |

File targeted by commands: `-f FILE`, otherwise `PLI_CONFIG`, otherwise `./pli.toml`.

### Migrating a machine from `.env` to `pli.toml`

```bash
pli config:import-env ~/pli/pli.env ~/pli/pli-27b.env \
  --options "--context 229376 --parallel 4 --no-thinking --vision --decode-share 1 --spill-gib 120" -o pli-27b.toml
PLI_CONFIG=pli-27b.toml pli config:check --sans-env
pli serve MODELE --config pli-27b.toml --host 127.0.0.1 --port 9080
```

The test `tests/test_pli_reglages.py` proves, for production files, that the two launches give the same
`PLI_*` environment variable by variable, the same settings fingerprint and the same arguments
(`REGLAGES_PROD=DIR` reruns it on the real files of a machine).

### Statuts

`recommande`: safe, documented, measured or in production; `experimental`: research, effect not guaranteed; `obsolete`:
kept for compatibility. `config:check` warns when an experimental or obsolete setting is set. Rules and cases
in doubt: [reglages.md](reglages.md) (2.10 and inventory).

`attention.per_row` (`PLI_ATTENTION_PER_ROW`) is a threshold: one attention call per line from N keys.
`attention.per_row_always` (`PLI_ATTENTION_PER_ROW_ALWAYS=1`) says "from the first key"; it sets
`PLI_ATTENTION_PER_ROW=1` and is rejected with a different threshold.

### Types

`flag` (`true`/`false`, i.e. `1`/`0` for the code), `integer`, `float`, `text`, `choice`, `list` (`alias`,
`api_key`). A type is inferred from what the code does with the value; `config:check` reports as an error a badly
written number and as a warning a value that the code never compares (for example `true` for a setting read `== "1"`,
which then remains off). The JSON generation files do not change: `pli.toml` names them (`[gen] config`).

### Rollback

Without `--config`, without `PLI_CONFIG`, without `--famille.reglage` and without variable derived from a key, nothing changes. After a
switch to `pli.toml`, removing `--config` (or `PLI_CONFIG`) and restoring the `.env` is sufficient.
