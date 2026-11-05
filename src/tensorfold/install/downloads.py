"""What ``pli setup`` downloads, shown in full before anything is downloaded, and the consent to download it.

For each item: the exact source (Hugging Face repo, revision, page), its license as its model card gives it, its size
as the Hub lists it (no weights are fetched to know it), where it goes on disk and what it brings. A file already in
the Hugging Face cache is marked "already downloaded" and not counted again.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable

from tensorfold.install import lsa_map
from tensorfold.install.machine import SetupError, say

MODEL = "mlx-community/Qwen3.8-27B-4bit"
DRAFTER = "z-lab/Qwen3.8-27B-DFlash2"
MTP_SOURCE = "EigenLabs/Qwen3.8-27B-MTP-bf16"
MTP_FILE = "mtp-head-4bit.safetensors"          # under PREFIX/models
REVISION = "main"                               # not pinned: what the repo's main branch holds when it is downloaded
# what the Hub listed on 3 November 2026, shown as "about" when the Hub cannot be asked
APPROXIMATE = {MODEL: 16_081_490_933, DRAFTER: 3_849_113_002, MTP_SOURCE: 849_407_885}
MTP_4BIT_BYTES = 228 * 2**20                    # the head in 4 bits (810 MiB in bf16)
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
LEAVE_OUT = ("Leave items out: --no-hybrid (no MTP head), --no-drafter (no draft at all: slower, the same text), "
             "--no-a1 (no A1 map: long prompts are read without it), --model DIR (a model already on this Mac), "
             "--no-models (nothing downloaded).")


@dataclass
class Item:
    role: str                      # model, drafter, mtp (the bf16 head), mtp4 (the head in 4 bits, written here),
    #                                lsa (A1's map, downloaded from a release and converted here)
    title: str
    purpose: str
    repo: str = ""                 # Hugging Face repo id; "" for a file or directory on this Mac
    url: str = ""                  # a file downloaded from elsewhere than the Hub (checked by its SHA-256)
    path: Path | None = None       # local item: what the user gave, or the file written here
    commit: str = ""
    license: str = ""
    size: int | None = None        # bytes of the whole repo (local: of the written file)
    missing: int | None = None     # bytes still to download; 0: already downloaded
    approximate: bool = False

    @property
    def downloads(self) -> bool:
        return bool(self.repo or self.url) and self.missing != 0


def is_repo_id(name: str) -> bool:
    return bool(_REPO_ID.match(name)) and not Path(name).expanduser().exists()


def hub_cache() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def repo_dir(repo: str) -> Path:
    return hub_cache() / f"models--{repo.replace('/', '--')}"


def _license(card: Any) -> str:
    if card is None:
        return ""
    get = card.get if hasattr(card, "get") else (lambda key, default=None: getattr(card, key, default))
    name = get("license", None)
    if isinstance(name, (list, tuple)):
        name = ", ".join(map(str, name))
    if name == "other" and get("license_name", None):
        name = f"other ({get('license_name', None)})"
    return str(name or "")


def hub_info(repo: str) -> dict[str, Any]:
    """The repo's files (name, bytes, blob id), commit and license, from the Hub's API: no weights downloaded."""

    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, revision=REVISION, files_metadata=True)
    files = []
    for sibling in info.siblings or ():
        lfs = sibling.lfs
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        files.append((sibling.rfilename, sibling.size, sha or sibling.blob_id))
    return {"commit": info.sha or "", "license": _license(getattr(info, "card_data", None)), "files": files}


def cached_complete(repo: str) -> bool:
    """The repo's weights are complete in the Hugging Face cache (what ``pli serve`` asks before serving)."""

    from tensorfold import hub

    found = hub.cached(repo)
    if found is None:
        return False
    return hub._cached_weights_complete(found)


def plan(model: str, drafter: str | None, mtp_head: str | None, hybrid: bool, prefix: Path,
         a1: bool = False) -> list[Item]:
    """The items of an install: the model, the DFlash2 draft (None: no draft), the MTP head for the hybrid draft,
    A1's map (``a1``)."""

    items = [_item("model", "Model (required)", "the model that writes every reply", model)]
    if drafter is not None:
        items.append(_item("drafter", "DFlash2 draft", "a draft model: replies come faster, the text is the same",
                           drafter))
    if hybrid and mtp_head:
        path = Path(mtp_head).expanduser().absolute()
        if not path.is_file():
            raise SetupError(f"--mtp-head {mtp_head}: no such file")
        items.append(Item("mtp4", "MTP head", "the hybrid draft, with DFlash2: faster, the same text", path=path,
                          size=path.stat().st_size))
    elif hybrid:
        items.append(Item("mtp", "MTP head of Qwen3.8-27B (bf16)",
                          "the hybrid draft, with DFlash2: replies come faster, the text is the same",
                          repo=MTP_SOURCE))
        items.append(Item("mtp4", "MTP head in 4 bits",
                          "written on this Mac from the MTP head above; nothing more to download",
                          path=prefix / "models" / MTP_FILE, size=MTP_4BIT_BYTES, approximate=True))
    if a1:
        path = prefix / "models" / lsa_map.NPZ
        items.append(Item("lsa", "A1 map (Latent Sparse Attention)",
                          "long prompts are read faster (chunks past 16 384 positions read 10% of the old context, "
                          "chosen by this map); the text can change", url=lsa_map.URL, path=path,
                          license=lsa_map.LICENSE, size=lsa_map.BYTES, missing=0 if path.is_file() else lsa_map.BYTES))
    return items


def _item(role: str, title: str, purpose: str, name: str) -> Item:
    path = Path(name).expanduser()
    if path.is_dir():
        return Item(role, title, purpose, path=path.absolute())
    if not is_repo_id(name):
        raise SetupError(f"{name} is neither a directory nor a Hugging Face repo id (owner/name)")
    return Item(role, title, purpose, repo=name)


def measure(items: list[Item], info: Callable[[str], dict[str, Any]] | None = None) -> None:
    """Fill in each repo's size, license and commit from the Hub, and what is already in the cache."""

    info = info or hub_info
    for item in items:
        if not item.repo:
            continue
        try:
            found = info(item.repo)
        except Exception as exc:  # noqa: BLE001 - offline, or the Hub refuses: sizes become approximate
            print(f"[pli] the Hub did not answer for {item.repo} ({type(exc).__name__}): sizes are approximate",
                  file=sys.stderr, flush=True)
            item.size = APPROXIMATE.get(item.repo)
            item.approximate = True
            item.missing = 0 if cached_complete(item.repo) else item.size
            continue
        item.commit, item.license = found["commit"], found["license"]
        blobs = repo_dir(item.repo) / "blobs"
        item.size = sum(size or 0 for _, size, _ in found["files"])
        item.missing = sum(size or 0 for _, size, blob in found["files"]
                           if not (blob and (blobs / blob).is_file()))


def human(n: int | None, approximate: bool = False) -> str:
    if n is None:
        return "size unknown"
    text = f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"
    return f"about {text}" if approximate else text


def existing(path: Path) -> Path:
    """``path``, or its nearest folder that exists: where it will be written."""

    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def free_bytes(path: Path) -> int:
    """Free disk where ``path`` is, or will be."""

    return shutil.disk_usage(existing(path)).free


def screen(items: list[Item]) -> str:
    """The list shown before anything is downloaded."""

    lines = ["Before anything is downloaded, here is what pli setup takes from Hugging Face and writes on this Mac:",
             ""]
    for n, item in enumerate(items, 1):
        if item.repo:
            lines.append(f"  {n}. {item.title}: {item.repo} - {item.purpose}")
            commit = f", commit {item.commit[:7]}" if item.commit else ""
            lines.append(f"     https://huggingface.co/{item.repo}, revision {REVISION} (not pinned{commit}), "
                         f"license: {item.license or 'see model card'}")
            where = repo_dir(item.repo)
            if item.missing == 0:
                lines.append(f"     already downloaded ({human(item.size, item.approximate)} in {where})")
            elif item.missing is not None and item.size and item.missing < item.size:
                lines.append(f"     {human(item.missing, item.approximate)} to download (of "
                             f"{human(item.size, item.approximate)}; the rest is already downloaded), into {where}")
            else:
                lines.append(f"     {human(item.missing, item.approximate)} to download, into {where}")
        elif item.url:
            lines.append(f"  {n}. {item.title}: {item.url} - {item.purpose}")
            lines.append(f"     release {lsa_map.RELEASE}, license: {item.license}, SHA-256 checked "
                         f"({lsa_map.SHA256[:12]}...)")
            if item.missing == 0:
                lines.append(f"     already converted ({item.path})")
            else:
                lines.append(f"     {human(item.missing)} to download, converted on this Mac into {item.path}")
        elif item.role == "mtp4" and any(i.role == "mtp" for i in items):
            lines.append(f"  {n}. {item.title} - {item.purpose}")
            lines.append(f"     {human(item.size, True)}, into {item.path}")
        else:
            lines.append(f"  {n}. {item.title}: {item.path} - {item.purpose}")
            lines.append("     on this Mac already, not downloaded")
    need = total(items)
    lines.append("")
    approx = any(i.approximate for i in items if i.downloads)
    lines.append(f"To download: {human(need, approx)}. Free disk: {human(free_bytes(hub_cache()))} "
                 f"({hub_cache()}).")
    head = next((i.path for i in items if i.role == "mtp4" and any(j.role == "mtp" for j in items)), None)
    if head is not None and os.stat(existing(head)).st_dev != os.stat(existing(hub_cache())).st_dev:
        lines.append(f"Free disk for the MTP head: {human(free_bytes(head))} ({head.parent}).")
    lines.append(LEAVE_OUT)
    return "\n".join(lines)


def total(items: list[Item]) -> int:
    return sum(i.missing or 0 for i in items if i.downloads)


def consent(items: list[Item], yes: bool, ask: Callable[[str], str] | None = None,
            terminal: Callable[[], bool] | None = None) -> None:
    """Go on only with the user's yes (or --yes); stop when there is no terminal to ask and no --yes."""

    if not any(i.downloads for i in items):
        say("Nothing to download: every file is already on this Mac.")
        return
    need, free = total(items), free_bytes(hub_cache())
    if need > free:
        raise SetupError(f"not enough free disk: {human(need)} to download, {human(free)} free in {hub_cache()}")
    if yes:
        say("Download accepted (--yes).")
        return
    terminal = terminal or (lambda: sys.stdin.isatty())
    if not terminal():
        raise SetupError("pli setup asks before it downloads, and there is no terminal to ask: add --yes to accept "
                         "the download above, or --no-models to download nothing")
    try:
        answer = (ask or input)("Download them? [y/N] ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        raise SetupError("Nothing downloaded. Leave items out with the options above, or run pli setup --no-models.",
                         code=0)
