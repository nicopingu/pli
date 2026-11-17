# Profils de lancement (exemples)

Ready-to-run PLI launches by use case, read by `tools/pli-profil`:

| Profile | For | Mac Memory | Disk |
| --- | --- | ---: | ---: |
| `code` | coder, discuter | 48 Go | 63 Go |
| `assistant-vocal` | voice: transcription, response, synthesis | 48 GB (estimated) | 65 GB |
| `creation` | images and videos | 64 GB | 222 GB |
| `creation-lora` | images and videos with a chosen LoRA | 64 GB | 290 GB |
| `all-inclusive` | code, voice, images, videos | 64 GB | 224 GB |
| `petite-machine` | Mac with 24 to 32 GB | 24 GB (estimated) | 24 GB |

```bash
tools/pli-profil                    # the list
tools/pli-profil code --dry-run     # what will be launched, memory and disk checked
tools/pli-profil code               # lance PLI
```

- These files are examples: to change them, copy them to your folder and pass `--dossier DOSSIER`.
- Machine paths are `${...}` variables: see `machine.exemple.env`.
- `catalogues/`: models from the image, video, and voice catalogs that the launcher writes with your paths.
- A launch profile is not a machine profile (`src/pli/additions/profiles/`, option `--profile`): it
  designates it via `"machine_profile"` (`auto`: chosen based on the chip).

Setting details, memory and disk per model, licenses: [docs/usage-expert.md](../../docs/usage-expert.md).
