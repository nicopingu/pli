"""English names of PLI's environment variables, and the old names that keep working.

PLI (27/10/2026): for the public release, the French setting names of PLI (``PLI_ARBRE_*``, ``PLI_LECTURE_CPU*``...) and
TensorFold's ``TF_*`` and ``TENSORFOLD_*`` names became English ``PLI_*`` names. :func:`apply` runs first thing in
``tensorfold/__init__.py``, before any module reads its settings: each old name set in the environment whose new name is
not set gives its value to the new name. The code reads only the new names; a new name already set wins over its old one.
"""

from __future__ import annotations

import os
from typing import MutableMapping

# (old name, new name)
RENAMED: tuple[tuple[str, str], ...] = (
    ("PLI_ADMISSION_PETITES", "PLI_ADMIT_SMALL"),
    ("PLI_ADMISSION_TRAVAIL", "PLI_ADMIT_BY_WORK"),
    ("PLI_ANCRE_JOURNAL", "PLI_ANCHOR_LOG"),
    ("PLI_ANE_BLOCS", "PLI_ANE_BLOCKS"),
    ("PLI_ANE_BLOCS_IDS", "PLI_ANE_BLOCK_IDS"),
    ("PLI_ANE_FFN_LIGNES", "PLI_ANE_FFN_MIN_ROWS"),
    ("PLI_ANE_FFN_UNITES", "PLI_ANE_FFN_UNITS"),
    ("PLI_ANE_GARDE", "PLI_ANE_KEEPAWAKE"),
    ("PLI_ANE_GARDE_MS", "PLI_ANE_KEEPAWAKE_MS"),
    ("PLI_ANE_PALIERS", "PLI_ANE_STEPS"),
    ("PLI_ANE_RELEVE", "PLI_ANE_STATS"),
    ("PLI_ANE_REQUANT_FILS", "PLI_ANE_REQUANT_THREADS"),
    ("PLI_ANE_RQ_APRES", "PLI_ANE_REQUANT_AFTER"),
    ("PLI_ARBITRE", "PLI_ARBITER"),
    ("PLI_ARBITRE_JOURNAL", "PLI_ARBITER_LOG"),
    ("PLI_ARBITRE_PALIER", "PLI_ARBITER_MIN_ROWS"),
    ("PLI_ARBITRE_TABLE", "PLI_ARBITER_TABLE"),
    ("PLI_ARBRE_ARETE", "PLI_TREE_EDGE"),
    ("PLI_ARBRE_BRUIT", "PLI_TREE_NOISE"),
    ("PLI_ARBRE_CORR", "PLI_TREE_CORRECT"),
    ("PLI_ARBRE_CTX_MAX", "PLI_TREE_CTX_MAX"),
    ("PLI_ARBRE_DN", "PLI_TREE_DELTANET"),
    ("PLI_ARBRE_ENFANTS", "PLI_TREE_CHILDREN"),
    ("PLI_ARBRE_EXACT", "PLI_TREE_EXACT"),
    ("PLI_ARBRE_NOEUDS", "PLI_TREE_NODES"),
    ("PLI_ARBRE_SEUL", "PLI_TREE_ALONE"),
    ("PLI_ARBRE_TAU", "PLI_TREE_TAU"),
    ("PLI_ARBRES_ATTM", "PLI_TREES_ATTM"),
    ("PLI_ATTENTION_F16_HC", "PLI_ATTENTION_F16_HEADS"),
    ("PLI_ATTENTION_GEMM_PVLONG", "PLI_ATTENTION_GEMM_PV_LONG"),
    ("PLI_ATTENTION_GEMM_PVSEUIL", "PLI_ATTENTION_GEMM_PV_THRESHOLD"),
    ("PLI_ATTENTION_GROUPEE", "PLI_ATTENTION_GROUPED"),
    ("PLI_ATTENTION_LIGNE", "PLI_ATTENTION_PER_ROW"),
    ("PLI_ATTENTION_REDUITE", "PLI_ATTENTION_REDUCED"),
    ("PLI_ATTENTION_REDUITE_ATTM", "PLI_ATTENTION_REDUCED_ATTM"),
    ("PLI_ATTENTION_REDUITE_MIN", "PLI_ATTENTION_REDUCED_MIN_ROWS"),
    ("PLI_ATTENTION_REDUITE_PLEINES", "PLI_ATTENTION_REDUCED_FULL_LAYERS"),
    ("PLI_ATTENTION_REDUITE_SEUIL", "PLI_ATTENTION_REDUCED_THRESHOLD"),
    ("PLI_ATTENTION_REGLAGE", "PLI_ATTENTION_TUNING_FILE"),
    ("PLI_ATTM_COURT", "PLI_ATTM_SHORT"),
    ("PLI_ATTM_PETITES", "PLI_ATTM_SMALL_KEYS"),
    ("PLI_ATTN_MORCEAUX", "PLI_ATTN_SLICES"),
    ("PLI_BATTEMENT_ANE", "PLI_HEARTBEAT_ANE"),
    ("PLI_BATTEMENT_FENETRE_S", "PLI_HEARTBEAT_WINDOW_S"),
    ("PLI_BATTEMENT_MS", "PLI_HEARTBEAT_MS"),
    ("PLI_CROISE_DENSE", "PLI_INTERLEAVE_DENSE"),
    ("PLI_EVENEMENTS", "PLI_EVENTS"),
    ("PLI_EVENEMENTS_MAX_MO", "PLI_EVENTS_MAX_MB"),
    ("PLI_F16_ECRETE", "PLI_F16_CLAMP"),
    ("PLI_FALCON_CIBLES_UN_SUR", "PLI_FALCON_TARGETS_EVERY"),
    ("PLI_FALCON_CONTEXTE8", "PLI_FALCON_CONTEXT8"),
    ("PLI_FALCON_FLOT", "PLI_FALCON_STREAM"),
    ("PLI_FALCON_LOT", "PLI_FALCON_BATCH"),
    ("PLI_FALCON_MAX_GO", "PLI_FALCON_MAX_GB"),
    ("PLI_FALCON_TETE", "PLI_FALCON_HEAD"),
    ("PLI_FONDRE_FIN", "PLI_MERGE_TAIL"),
    ("PLI_GARDE_HOTE", "PLI_HOST_GUARD"),
    ("PLI_GARDE_REPONSE", "PLI_KEEP_REPLY"),
    ("PLI_HEURES", "PLI_LOG_TIMES"),
    ("PLI_HOTES_ADMIS", "PLI_ALLOWED_HOSTS"),
    ("PLI_HYBRIDE", "PLI_HYBRID"),
    ("PLI_HYBRIDE_FORCE", "PLI_HYBRID_FORCE"),
    ("PLI_HYBRIDE_GLOBAL_TOURS", "PLI_HYBRID_GLOBAL_ROUNDS"),
    ("PLI_HYBRIDE_JOURNAL", "PLI_HYBRID_LOG"),
    ("PLI_HYBRIDE_PROFONDEUR", "PLI_HYBRID_DEPTH"),
    ("PLI_INSTANTANE_REGLAGES", "PLI_SNAPSHOT_SETTINGS"),
    ("PLI_LECTURE_CPU", "PLI_PREFILL_CPU"),
    ("PLI_LECTURE_CPU_MIN", "PLI_PREFILL_CPU_MIN"),
    ("PLI_LECTURE_CPU_MODE", "PLI_PREFILL_CPU_MODE"),
    ("PLI_LECTURE_CPU_PRIORITE", "PLI_PREFILL_CPU_PRIORITY"),
    ("PLI_LECTURE_CPU_TRACE", "PLI_PREFILL_CPU_TRACE"),
    ("PLI_LECTURE_DEPLIE", "PLI_PREFILL_UNPACK"),
    ("PLI_LECTURE_DEPLIE_CONTROLE", "PLI_PREFILL_UNPACK_CHECK"),
    ("PLI_LECTURE_DEPLIE_LIGNES", "PLI_PREFILL_UNPACK_ROWS"),
    ("PLI_LECTURE_DEPLIE_MLP", "PLI_PREFILL_UNPACK_MLP"),
    ("PLI_LECTURE_DEPLIE_NN", "PLI_PREFILL_UNPACK_NN"),
    ("PLI_LECTURE_DEPLIE_NOYAU", "PLI_PREFILL_UNPACK_KERNEL"),
    ("PLI_LECTURE_DEPLIE_PILE", "PLI_PREFILL_UNPACK_STACK"),
    ("PLI_MLP_ENT", "PLI_MLP_INT"),
    ("PLI_MORCEAU_COUCHES", "PLI_CHUNK_LAYERS"),
    ("PLI_MORCEAU_JETONS", "PLI_CHUNK_TOKENS"),
    ("PLI_MTP27", "PLI_MTP_HEAD"),
    ("PLI_MTP27_BROUILLONS", "PLI_MTP_DRAFTS"),
    ("PLI_MTP27_FENETRE", "PLI_MTP_WINDOW"),
    ("PLI_PAVES_PAR_GROUPE", "PLI_TILES_PER_GROUP"),
    ("PLI_PROFIL", "PLI_PROFILE"),
    ("PLI_RANGE", "PLI_RELAYOUT"),
    ("PLI_RAPPEL", "PLI_RECALL"),
    ("PLI_RAPPEL_JETONS", "PLI_RECALL_TOKENS"),
    ("PLI_REDUITE_ANCRE", "PLI_REDUCED_ANCHOR"),
    ("PLI_REDUITE_TOURS", "PLI_REDUCED_ROUNDS"),
    ("PLI_RQ_NC_MAX", "PLI_REQUANT_NC_MAX"),
    ("PLI_TOURS_ENTRE_LECTURES", "PLI_ROUNDS_BETWEEN_PREFILLS"),
    ("PLI_TRACE_COUCHES", "PLI_TRACE_LAYERS"),
    ("PLI_TRACE_DOSSIER", "PLI_TRACE_DIR"),
    ("PLI_TRACE_MAX_MO", "PLI_TRACE_MAX_MB"),
    ("TENSORFOLD_API_KEY", "PLI_API_KEY"),
    ("TENSORFOLD_MEMORY_LIMIT_GB", "PLI_MEMORY_LIMIT_GB"),
    ("TENSORFOLD_NO_LIVE", "PLI_NO_LIVE"),
    ("TENSORFOLD_NO_UPDATE_CHECK", "PLI_NO_UPDATE_CHECK"),
    ("TENSORFOLD_REQUEST_LOG", "PLI_REQUEST_LOG"),
    ("TENSORFOLD_SEED_SALT", "PLI_SEED_SALT"),
    ("TENSORFOLD_TEST_LAUNCHD", "PLI_TEST_LAUNCHD"),
    ("TENSORFOLD_VIDEO_TOKENS", "PLI_VIDEO_TOKENS"),
    ("TF_ATTN_DIRECT_P", "PLI_ATTN_DIRECT_P"),
    ("TF_ATTN_TILES", "PLI_ATTN_TILES"),
    ("TF_COPY_MATCH", "PLI_COPY_MATCH"),
    ("TF_COPY_ROWS", "PLI_COPY_ROWS"),
    ("TF_DRAFT_CAPTURE", "PLI_DRAFT_CAPTURE"),
    ("TF_DRAFT_LEAN", "PLI_DRAFT_LEAN"),
    ("TF_DRAFT_LOG", "PLI_DRAFT_LOG"),
    ("TF_DRAFT_PROFILE", "PLI_DRAFT_PROFILE"),
    ("TF_DRAFT_VOCAB", "PLI_DRAFT_VOCAB"),
    ("TF_FAMILY_PROFILE", "PLI_FAMILY_PROFILE"),
    ("TF_FAMILY_ROUND_LOG", "PLI_FAMILY_ROUND_LOG"),
    ("TF_FLASH_DENSE", "PLI_FLASH_DENSE"),
    ("TF_FLASH_DRAFT_VOCAB", "PLI_FLASH_DRAFT_VOCAB"),
    ("TF_FLASH_FUSED", "PLI_FLASH_FUSED"),
    ("TF_FLASH_MTP", "PLI_FLASH_MTP"),
    ("TF_FLASH_PREFILL", "PLI_FLASH_PREFILL"),
    ("TF_FLASH_QUEUED", "PLI_FLASH_QUEUED"),
    ("TF_GLM_CLEAR_THINKING", "PLI_GLM_CLEAR_THINKING"),
    ("TF_GLM_DENSE", "PLI_GLM_DENSE"),
    ("TF_LANE_TILE", "PLI_LANE_TILE"),
    ("TF_MOE_ROWS", "PLI_MOE_ROWS"),
    ("TF_MTP_ROUNDS", "PLI_MTP_ROUNDS"),
    ("TF_NEMOTRON_MTP", "PLI_NEMOTRON_MTP"),
    ("TF_NGRAM_HOST", "PLI_NGRAM_HOST"),
    ("TF_NGRAM_WEIGHT", "PLI_NGRAM_WEIGHT"),
    ("TF_QWEN36_DRAFT_VOCAB", "PLI_QWEN36_DRAFT_VOCAB"),
    ("TF_QWEN36_MTP", "PLI_QWEN36_MTP"),
    ("TF_REQUIRE_KERNELS", "PLI_REQUIRE_KERNELS"),
    ("TF_ROW_ATTENTION", "PLI_ROW_ATTENTION"),
    ("TF_TREE_TRACE", "PLI_TREE_TRACE"),
    ("PLI_ARBRE_EXACT2", "PLI_TREE_EXACT2"),
    ("PLI_CHAINE_GPU", "PLI_GPU_CHAIN"),
    ("PLI_CONTROLE_TEXTE", "PLI_CONTROL_AS_TEXT"),
    ("PLI_HYBRIDE_ARBRES", "PLI_HYBRID_TREES"),
    ("PLI_HYBRIDE_IM_START", "PLI_HYBRID_IM_START"),
    ("PLI_HYBRIDE_MESSAGES", "PLI_HYBRID_MESSAGES"),
    ("PLI_KV_PARTAGE", "PLI_KV_SHARING"),
    ("PLI_POINTS_JOURNAL", "PLI_CHECKPOINT_LOG"),
    # PLI 4.0.0 (production of 02/11/2026)
    ("PLI_S3B_LIGNES", "PLI_S3B_ROWS"),
    ("PLI_S3B_SEUIL", "PLI_S3B_THRESHOLD"),
    ("PLI_BROUILLON_ATTN", "PLI_DRAFT_ATTN"),
    ("PLI_BROUILLON_PILE", "PLI_DRAFT_STACK"),
    ("PLI_BROUILLON_QKR", "PLI_DRAFT_QKR"),
    ("PLI_BROUILLON_TOPK", "PLI_DRAFT_TOPK"),
    ("PLI_ATTM_CHARGE", "PLI_ATTM_LOAD"),
    ("PLI_RANGE_BROUILLON", "PLI_RELAYOUT_DRAFT"),
    ("PLI_RELIRE_DISQUE", "PLI_REREAD_DISK"),
    ("PLI_DISQUE_FIN_REPONSE", "PLI_DISK_REPLY_END"),
    ("PLI_ARBRES_SEUIL_JETONS", "PLI_TREES_TOKEN_THRESHOLD"),
    ("PLI_ARBRES_COURTES", "PLI_TREES_SHORT_ONLY"),
    ("PLI_PROFIL_TOUR", "PLI_ROUND_PROFILE"),
    ("PLI_PROFIL_TOUR_TRANCHE", "PLI_ROUND_PROFILE_BUCKET"),
    ("PLI_PROFIL_TOUR_N", "PLI_ROUND_PROFILE_N"),
    ("PLI_ARBRE_GAIN_MIN", "PLI_TREE_GAIN_MIN"),
    ("PLI_TETE_AVANT_COPIE", "PLI_HEAD_BEFORE_COPY"),
    ("PLI_TETE_COPIE_SEUIL", "PLI_HEAD_COPY_THRESHOLD"),
    ("PLI_TETE_COPIE_POIDS", "PLI_HEAD_COPY_WEIGHT"),
    ("PLI_COPIE_JOURNAL", "PLI_COPY_LOG"),
    ("PLI_TREILLIS_JOURNAL", "PLI_LATTICE_LOG"),
    ("PLI_LISTE_PLUS", "PLI_EXTENDED_DRAFT_LIST"),
    ("PLI_DEVERSE_INVITE", "PLI_SPILL_PROMPT"),
    ("PLI_FOURCHE_MIN", "PLI_FORK_MIN"),
    # PLI 4.0.1 (production of 03/11/2026)
    ("PLI_DEBUT_DERNIER", "PLI_LAST_MESSAGE_CHECKPOINT"),
    # PLI 4.1.0 (production of 04 and 05/11/2026)
    ("PLI_POINTS_BITS", "PLI_CHECKPOINT_BITS"),
    ("PLI_POINTS_GROUPE", "PLI_CHECKPOINT_GROUP"),
    ("PLI_ANCRE1_LSA", "PLI_A1_LSA_MAP"),
    ("PLI_ANCRE1_LECTURE_SEULE", "PLI_A1_PREFILL_ONLY"),
    ("PLI_A1_TRANCHE", "PLI_A1_SLICE"),
    ("PLI_A1_PETITE", "PLI_A1_SMALL_SLICE"),
    ("PLI_A1_GARDES_MAX", "PLI_A1_MAX_TILES"),
)

NEW_NAME = dict(RENAMED)
OLD_NAME = {new: old for old, new in RENAMED}

# Old values still accepted, read as the new ones.
OLD_VALUES = {"PLI_PREFILL_CPU_MODE": {"decale": "shifted", "couche": "layer"}}


def apply(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Set each new name from its old name when only the old one is set; return the new names set."""

    environ = os.environ if environ is None else environ
    done = []
    for old, new in RENAMED:
        if old in environ and new not in environ:
            environ[new] = environ[old]
            done.append(new)
    return done


def value(name: str, text: str) -> str:
    """``text`` read for the variable ``name``, an old value replaced by its new one."""

    return OLD_VALUES.get(name, {}).get(text, text)


def pop(name: str, default: str = "", environ: MutableMapping[str, str] | None = None) -> str:
    """Remove ``name`` and its old name from the environment; return the value, the new name's first."""

    environ = os.environ if environ is None else environ
    old = environ.pop(OLD_NAME[name], None) if name in OLD_NAME else None
    new = environ.pop(name, None)
    return new if new is not None else old if old is not None else default


def pli_settings(environ: MutableMapping[str, str] | None = None, exclude: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """The ``PLI_*`` settings of ``environ`` as sorted (new name, value) pairs, each setting once.

    A setting set under its old name counts once, under its new name (the new name's value wins when both are set).
    Settings renamed from TensorFold's ``TF_*`` and ``TENSORFOLD_*`` stay out, as they did before the renaming.
    """

    environ = os.environ if environ is None else environ
    items = {}
    for key, text in environ.items():
        new = NEW_NAME.get(key, key)
        if not new.startswith("PLI_") or not OLD_NAME.get(new, new).startswith("PLI_") or new in exclude:
            continue
        if key == new or new not in environ:
            items[new] = text
    return sorted(items.items())
