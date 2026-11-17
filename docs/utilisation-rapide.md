# PLI : utilisation rapide

A page to launch PLI without knowing its settings. Details are in [usage-expert.md](usage-expert.md).

> **Licenses, read before use.** Qwen-Image 2.1 and the Lightning v2 LoRA: non-commercial use only
> (Qwen Research License). Lightning v1: no license declared on its page. MiniMax-H3 and the Turbo LoRA:
> MiniMax community license, which excludes the European Union, the United Kingdom, South Korea, and the United States
> ("Excluded Territories"): it does not authorize use from these countries. Details and links:
> [usage-expert.md §8](usage-expert.md#8-sources-et-licences-des-modèles).

## 1. Installer

```bash
git clone https://github.com/nicopingu/pli.git ~/pli-source
cd ~/pli-source
./install.sh install          # PLI in ~/pli, the 27B, its drafters; answer "y" to the download
```

For voice, images, or video, you additionally need (one time only):

| For | To do |
| --- | --- |
| Voix | `python3 -m venv ~/pli/venvs/mlx-audio && ~/pli/venvs/mlx-audio/bin/pip install mlx-audio` |
| Images | `python3 -m venv ~/pli/venvs/mflux && ~/pli/venvs/mflux/bin/pip install mflux`, then [OsaurusAI/Qwen-Image-2.1-mflux-8bit](https://huggingface.co/OsaurusAI/Qwen-Image-2.1-mflux-8bit) in `~/pli/modeles-image/Qwen-Image-2.1-mflux-8bit` (non-commercial license) |
| Video | `brew install ffmpeg`, the `h3` program compiled from [h3.c](https://github.com/antirez/h3.c) in `~/pli/h3.c/`, the `FL2VA` folder, and the files from the root of [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) in `~/pli/modeles-video/MiniMax-H3` |

The licenses for each model are in [usage-expert.md](usage-expert.md#8-sources-et-licences-des-modèles).
Vos chemins sont ailleurs ? Copiez `templates/launch-profiles/machine.exemple.env` en `~/.config/pli/machine.env`
and change the relevant lines.

## 2. Choose a profile

| Profile | For | Mac Memory | Disk |
| --- | --- | ---: | ---: |
| `code` | coding, chatting | 48 GB (validated on 64) | 63 GB |
| `assistant-vocal` | speaking, listening, responding | 48 GB (estimated, validated on 64) | 65 GB |
| `creation` | images and videos | 64 GB | 222 GB |
| `creation-lora` | images and videos with LoRA | 64 GB | 290 GB |
| `tout-inclus` | everything: code, voice, images, videos | 64 GB | 224 GB |
| `petite-machine` | Mac with 24 to 32 GB (24 GB: M2, M3, M4, M4 Pro) | 24 GB (estimated) | 24 GB |

Memory: measured on an M1 Ultra 64 GB Mac Studio, unless noted "estimated". Disk: the entire profile, including
conversation snapshots (up to 40 GB). Below 24 GB, PLI has no model that fits; no M1 has 24 GB.
Each profile automatically selects the chip settings (`"machine_profile": "auto"`).

## 3. Lancer

```bash
cd ~/pli-source
tools/pli-profil                        # the list of profiles
tools/pli-profil code --dry-run         # what will be launched, and the memory and disk verification
tools/pli-profil code                   # launches PLI (Ctrl-C once to stop it)
```

If memory or disk is insufficient, the launcher refuses and says what to do. The first launch downloads the
model if it is missing.

## 4. Appeler PLI

The address is `http://127.0.0.1:8080/v1` (OpenAI clients); the model is called `qwen3.8-27b` (`petit` for
`petite-machine`).

```bash
# Discuter
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Bonjour"}]}'

# Faire parler (WAV)
curl -s http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","voice":"kokoro-siwis","input":"Bonjour, je suis PLI."}' -o bonjour.wav

# Transcribe an audio file
curl -s http://127.0.0.1:8080/v1/audio/transcriptions -F file=@bonjour.wav -F language=fr

# An image (response: PNG in base64 in data[0].b64_json)
curl -s http://127.0.0.1:8080/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt":"a lighthouse at sunrise","size":"1024x1024"}'

# A video: create the job, track its state, retrieve the MP4
curl -s http://127.0.0.1:8080/v1/videos -H 'Content-Type: application/json' \
  -d '{"prompt":"a fox runs in the snow","seconds":2,"size":"512x512"}'
curl -s http://127.0.0.1:8080/v1/videos/ID
curl -s http://127.0.0.1:8080/v1/videos/ID/content -o video.mp4
```

Before `creation-lora`, convert the image LoRA for mflux (once):
`tools/pli-lora-image qwen_image_21_lightning_v2_rl_4steps_lora_rank64.safetensors ~/pli/lora/lightning-v2rl-r64.safetensors`
(fichier de [lightx2v/Qwen-Image-2.1-Lightning-v2-RL](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning-v2-RL)).
With `creation-lora`, add `"lora":[{"nom":"lightning"}]` to the image (4 steps instead of 40) or
`"lora":[{"nom":"turbo"}]` to the video. The first video with a LoRA first fuses it into the model
(`tools/pli-lora-fusion`, 62 GB written to disk, only once).

A 503 response of type `memoire` means: not enough free memory at that time. Retry when the demand in


## 5. Settings

Each setting has a name `family.setting` (`cache.context`, `ane.ffn`, `brouillons.hybrid`...). The five useful commands
(none loads a model):

```bash
pli config:list cache                       # settings of a family, their value and where it comes from
pli config:explain cache.checkpoint_bits    # help, default, accepted names, current value and its source
pli config:set cache.context 229376         # writes to ./pli.toml (or -f FILE)
pli config:check                            # checks pli.toml and the environment (unknown names, old names)
pli serve MODEL --config pli.toml          # launches with this file; --family.setting VALUE for a test
```

The option and the variable win over `pli.toml`, which wins over the machine profile. Details:
[usage-expert.md](usage-expert.md#11-settings-by-families-plitoml-pli-config) and [reglages.md](reglages.md).
