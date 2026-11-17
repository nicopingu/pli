"""Generation child process supervisor: on-demand launch, HTTP relay, unloading, memory guard, preemption.

Room for the voice assistant during a video (14/11/2026; PLI_GEN_CONFIG video_* keys, all off by
default): parallele_video (the general model allows only N generations at a time while a video job runs, recall
parallele given by memoire_general.brancher), reserve_voix (a video job starts only if memory also covers the
voice; the background returns what is missing to the general model), pause_voix (h3 paused during a voice or transcription).

PLI 5.6.4 (17/11/2026), when a child lacks memory, in this order (the 27B stays loaded as long as memory holds):
1. inactive families are stopped, then the general model frees its cache (as before);
2. room asked of the bench queue (PLI_GEN_CONFIG keys place_fichier and place_attente_s): the supervisor writes
   ``place_fichier`` (JSON: family, missing bytes, since), the queue's probe on the same Mac reads it and the queue
   pauses its shared passages that can resume (--pause reprise); the supervisor waits up to place_attente_s for the
   memory to come back. The file stays while a child launched this way is alive, touched every turn: the queue
   relaunches its passages a few minutes after it is gone;
3. the general model is set aside, if a ``mettre_de_cote`` callback is given (not connected yet in 5.6.5: without it
   the supervisor goes straight to step 4);
4. otherwise 503 « memoire », as before.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

GO = 1024**3
MO = 1024**2
RESERVE_PAR_DEFAUT_GO = 6.0
FAMILLES_ECHANGE = ("audio", "image", "video")   # all families: an increase in swap stops the child process
# voice need without « memoire_go » in its catalog (copy of fils_audio.MEMOIRE_GO_DEFAUT: the child process is autonomous)
BESOIN_VOIX_GO = {"drama3": 24.0, "kokoro": 3.0, "qwen3tts": 7.0}


@dataclass
class Famille:
    """A generation child process: how to launch it, where it listens, its memory cost, when to unload it."""

    nom: str
    argv: list[str]                         # child process command (argv list)
    port: int                               # port local (127.0.0.1)
    memoire_octets: int = 0                 # declared need, checked before launch
    delai_decharge_s: float = 300.0         # inactivity after which the child process is stopped
    sante_delai_s: float = 60.0             # maximum wait for GET /sante at startup
    relais_delai_s: float = 900.0           # maximum wait for the child process response
    sante: Callable[[Famille], bool] | None = None   # replaces the default GET /sante
    occupe: Callable[[Famille], bool] | None = None  # replaces reading « occupe » in the /sante response
    surveiller_echange: bool | None = None  # None: true for all families (FAMILLES_ECHANGE)
    reserve_octets: int | None = None       # family-specific reserve; None: the supervisor's


class Fils:
    """A launched child process, with the reason for its stop if the supervisor stopped it."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self.raison: str | None = None
        self.type_erreur = "fils_arrete"


@dataclass
class Etat:
    verrou: threading.RLock = field(default_factory=threading.RLock)
    fils: Fils | None = None
    actifs: int = 0
    dernier_usage: float = 0.0
    echange_base: int | None = None
    demarrage: bool = False                 # read without lock by /health: _lancer holds the lock up to 60 s
    dernier_usage_epoch: float = 0.0
    modele: str | None = None               # PLI 5.6.4: model named by the last request relayed (for /health)


Reponse = tuple[int, dict[str, str], bytes]


def analyser_vm_stat(texte: str) -> int:
    """Recoverable memory in bytes from vm_stat output: free + inactive + speculative + purgeable pages.

    Format seen on an M1 Ultra 64 GB: first line « Mach Virtual Memory Statistics: (page size of 16384 bytes) », then lines
    « Pages free:      4494. » (number followed by a dot).
    """

    taille = re.search(r"page size of (\d+) bytes", texte)
    if not taille:
        raise ValueError("taille de page absente de vm_stat")
    pages = 0
    for ligne in texte.splitlines():
        m = re.match(r"\s*Pages (free|inactive|speculative|purgeable):\s+(\d+)\.?\s*$", ligne)
        if m:
            pages += int(m.group(2))
    return pages * int(taille.group(1))


def analyser_swapusage(texte: str) -> int:
    """Swap used in bytes from « vm.swapusage: total = 2048.00M  used = 522.94M  free = ... ».

    Depending on the system language, a comma replaces the dot (« 522,94M »): both are read.
    """

    m = re.search(r"used\s*=\s*([\d.,]+)\s*([KMGT]?)", texte)
    if not m:
        raise ValueError("« used » absent de vm.swapusage")
    valeur = float(m.group(1).replace(",", "."))
    unite = {"": 1, "K": 1024, "M": MO, "G": GO, "T": 1024 * GO}[m.group(2)]
    return int(valeur * unite)


def _commande(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=10, check=True,
                          env={**os.environ, "LC_ALL": "C"}).stdout


def memoire_disponible_systeme() -> int:
    """Recoverable memory now: vm_stat on macOS, MemAvailable from /proc/meminfo elsewhere."""

    if shutil.which("vm_stat"):
        return analyser_vm_stat(_commande(["vm_stat"]))
    with open("/proc/meminfo") as f:
        for ligne in f:
            if ligne.startswith("MemAvailable:"):
                return int(ligne.split()[1]) * 1024
    raise OSError("available memory unreadable")


def echange_octets_systeme() -> int:
    """Swap used in bytes (sysctl vm.swapusage on macOS, SwapTotal - SwapFree elsewhere)."""

    if shutil.which("sysctl") and os.uname().sysname == "Darwin":
        return analyser_swapusage(_commande(["sysctl", "vm.swapusage"]))
    valeurs: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for ligne in f:
            cle, _, reste = ligne.partition(":")
            if cle in ("SwapTotal", "SwapFree"):
                valeurs[cle] = int(reste.split()[0]) * 1024
    return valeurs["SwapTotal"] - valeurs["SwapFree"]


def _type_erreur(reponse: "Reponse") -> str | None:
    try:
        d = json.loads(reponse[2] or b"{}")
        return d.get("error", {}).get("type") if isinstance(d, dict) and isinstance(d.get("error"), dict) else None
    except (ValueError, UnicodeDecodeError):
        return None


def erreur(code: int, type_: str, message: str, **plus: Any) -> Reponse:
    corps = json.dumps({"error": {"type": type_, "message": message, **plus}}, ensure_ascii=False).encode()
    return code, {"Content-Type": "application/json"}, corps


class Superviseur:
    def __init__(self, familles: dict[str, Famille], *,
                 memoire_disponible: Callable[[], int] | None = None,
                 echange_octets: Callable[[], int] | None = None,
                 reserve_octets: int | None = None,
                 drapeau: str | None = None,
                 intervalle_s: float = 1.0,
                 seuil_echange_octets: int = 64 * MO,
                 dispo_min_echange_octets: int | None = None,
                 delai_kill_s: float = 10.0,
                 env: dict[str, str] | None = None,
                 dossier_pid: str | None = None,
                 liberer: Callable[[int], int] | None = None,
                 rendre: Callable[[], None] | None = None,
                 plafonner: Callable[[], bool] | None = None,
                 delai_remesure_s: float = 2.0,
                 parallele: Callable[[int | None], None] | None = None,
                 parallele_video: int | None = None,
                 reserve_voix: bool = False,
                 reserve_voix_octets: int | None = None,
                 pause_voix: bool = False,
                 attente_memoire_s: float = 1800.0,
                 delai_besoin_s: float = 10.0,
                 place_fichier: str | None = None,
                 place_attente_s: float = 120.0,
                 mettre_de_cote: Callable[[int], int] | None = None) -> None:
        self.familles = dict(familles)
        # room for the voice assistant during a video (see the header); parallele: callback of the general model (memoire_general.brancher)
        self.parallele = parallele
        self.parallele_video = None if parallele_video is None else max(1, int(parallele_video))
        self.reserve_voix = bool(reserve_voix)
        self.reserve_voix_octets = reserve_voix_octets
        self.pause_voix = bool(pause_voix)
        self.attente_memoire_s = float(attente_memoire_s)
        self.delai_besoin_s = float(delai_besoin_s)
        self._parallele_reduit = False
        self._derniere_demande = 0.0
        self._liberation_video: threading.Thread | None = None
        self.liberations_video = 0
        self._catalogues: dict[str, tuple[float, Any]] = {}     # path -> (date, content) of child process catalogs
        # general model requirements (provided by gen/memoire_general.py): the supervisor does not import the engine
        self.liberer = liberer
        self.rendre = rendre
        # PLI 5.6.3 (11/17/2026): caps the general model at what it holds while a child runs, even when nothing had to
        # be freed (night of 11/17: the 27B regrew from 31 to 42 GB during an image, the image child starved, 504)
        self.plafonner = plafonner
        # PLI 5.6.4: room asked of the bench queue (file read by its probe), then the general model set aside
        self.place_fichier = os.path.expanduser(place_fichier) if place_fichier else None
        self.place_attente_s = float(place_attente_s)
        self._place: set[str] = set()             # families launched after a room request: the file stays while they live
        self._place_id: str | None = None
        self.places_demandees = 0
        self.mettre_de_cote = mettre_de_cote
        self.mises_de_cote = 0
        if self.place_fichier:
            self._effacer_place()                 # a file left by a server that stopped is not a request
        self.delai_remesure_s = delai_remesure_s
        self._liberees: set[str] = set()          # families launched after a general model release
        self._en_lancement: set[str] = set()
        self._verrou_liberation = threading.Lock()
        self.memoire_disponible = memoire_disponible or memoire_disponible_systeme
        self.echange_octets = echange_octets or echange_octets_systeme
        if reserve_octets is None:
            reserve_octets = int(float(os.environ.get("PLI_GEN_RESERVE_GO", RESERVE_PAR_DEFAUT_GO)) * GO)
        self.reserve_octets = reserve_octets
        self.intervalle_s = intervalle_s
        self.seuil_echange_octets = seuil_echange_octets
        # None: an increase in swap is enough to stop the child process. Otherwise, available memory must also be
        # below this threshold: on the first voice loading, macOS swaps out cold pages while there is still
        # memory available (deployments of 11/13/2026: +3 and +4.8 GB in 11 s, child processes stopped incorrectly)
        self.dispo_min_echange_octets = dispo_min_echange_octets
        self.delai_kill_s = delai_kill_s
        self.env = env
        self._dossier: str | None = None
        if drapeau is None:
            self._dossier = tempfile.mkdtemp(prefix="pli-gen-")
            drapeau = os.path.join(self._dossier, "general.drapeau")
        self.drapeau = drapeau
        # set as long as an image or voice request is in progress (including launch): video does not start during this time
        self.drapeau_autres = drapeau + ".autres"
        self._autres = 0
        self._verrou_autres = threading.Lock()
        self.drapeau_voix = drapeau + ".voix"        # a voice or transcription in progress (pause_voix)
        self.drapeau_video = drapeau + ".video"      # set by the video child while a job runs (parallele_video)
        self.drapeau_audio = drapeau + ".audio"      # the voice is loaded (reserve_voix)
        self.fichier_besoin = drapeau + ".besoin"    # memory missing for the pending video job (reserve_voix)
        self._voix = 0
        self._etats = {nom: Etat() for nom in self.familles}
        if dossier_pid is None:
            dossier_pid = os.environ.get("PLI_GEN_PIDS") or os.path.join(
                tempfile.gettempdir(), f"pli-gen-pids-{os.getuid()}")
        self.dossier_pid = dossier_pid
        os.makedirs(self.dossier_pid, exist_ok=True)
        self.orphelins_tues = self.nettoyer_orphelins()
        self._general = 0
        self._verrou_general = threading.Lock()
        self._stop = threading.Event()
        self._fond: threading.Thread | None = None
        self._verrou_fond = threading.Lock()

    # preemption: the general model is working, image and video child processes wait between steps

    def general_debut(self) -> None:
        with self._verrou_general:
            self._general += 1
            if self._general == 1:
                open(self.drapeau, "w").close()

    def general_fin(self) -> None:
        with self._verrou_general:
            self._general = max(0, self._general - 1)
            if self._general == 0:
                try:
                    os.unlink(self.drapeau)
                except FileNotFoundError:
                    pass

    def _compter_autres(self, pas: int) -> None:
        with self._verrou_autres:
            self._autres = max(0, self._autres + pas)
            try:
                if self._autres > 0:
                    open(self.drapeau_autres, "w").close()
                else:
                    os.unlink(self.drapeau_autres)
            except FileNotFoundError:
                pass

    def _compter_voix(self, pas: int) -> None:
        with self._verrou_autres:
            self._voix = max(0, self._voix + pas)
            try:
                if self._voix > 0:
                    open(self.drapeau_voix, "w").close()
                else:
                    os.unlink(self.drapeau_voix)
            except FileNotFoundError:
                pass

    # need of the requested model (11/15/2026: Krea 2, 29 GB, launched on the family's need, 26 GB, image killed by the
    # swap guard): read from the child process's catalog (option --catalogue or --voix in its argv)

    def _catalogue(self, fam: Famille) -> Any:
        chemin = None
        for option in ("--catalogue", "--voix"):
            if option in fam.argv[:-1]:
                chemin = fam.argv[fam.argv.index(option) + 1]
        if not chemin:
            return None
        try:
            date = os.stat(chemin).st_mtime
            vu = self._catalogues.get(chemin)
            if vu is None or vu[0] != date:
                with open(chemin, encoding="utf-8") as f:
                    vu = (date, json.load(f))
                self._catalogues[chemin] = vu
            return vu[1]
        except (OSError, ValueError):
            return None

    def besoin_demande(self, fam: Famille, methode: str, chemin: str, corps: bytes) -> int:
        """Bytes requested by the target model (memoire_go from its catalog); 0 if unknown (keep the family's one)."""

        if methode != "POST" or not corps:
            return 0
        cat = self._catalogue(fam)
        if not isinstance(cat, dict):
            return 0
        try:
            d = json.loads(corps)
        except (ValueError, UnicodeDecodeError):
            return 0                                   # multipart (transcription): family requirement
        if not isinstance(d, dict):
            return 0
        go = None
        if fam.nom == "audio":
            if not chemin.split("?", 1)[0].rstrip("/").endswith("/speech"):
                return 0
            voix = cat.get("voix") if isinstance(cat.get("voix"), dict) else cat
            defaut = cat.get("defaut") if isinstance(cat.get("defaut"), dict) else {}
            nom = d.get("voice") or defaut.get(str(d.get("model") or "")) or defaut.get("*")
            entree = voix.get(nom) if isinstance(nom, str) else None
            if entree is None and isinstance(nom, str):
                for v in voix.values():
                    alias = v.get("alias") if isinstance(v, dict) else None
                    if nom == alias or (isinstance(alias, list) and nom in alias):
                        entree = v
            if isinstance(entree, dict):
                go = entree.get("memoire_go", BESOIN_VOIX_GO.get(str(entree.get("modele"))))
        else:
            modeles = cat.get("modeles") if isinstance(cat.get("modeles"), dict) else {}
            m = modeles.get(d.get("model") or cat.get("defaut"))
            if isinstance(m, dict):
                go = m.get("memoire_go")
                if fam.nom == "video" and cat.get("ssd_streaming") is False:
                    go = m.get("memoire_go_resident", go)
        try:
            return int(float(go) * GO) if go else 0
        except (TypeError, ValueError):
            return 0

    def _gardees(self) -> tuple[str, ...]:
        """Families not to stop to make room: audio as long as the video-voice reserve serves a video."""

        if self.reserve_voix and "audio" in self.familles and "video" in self._etats and self._vivant("video"):
            if os.path.exists(self.drapeau_video) or self._occupe(self.familles["video"]):
                return ("audio",)
        return ()

    def _faire_place(self, fam: Famille, requis: int) -> tuple[int, bool]:
        """Stops other inactive families (excluding _gardees) then waits for memory to return (up to 10 s).
        Returns (available memory, at least one family stopped)."""

        try:
            libre = int(self.memoire_disponible())
        except Exception:  # noqa: BLE001
            return 0, False
        if libre >= requis:
            return libre, False
        arretees = self._arreter_inactives(fam, garder=self._gardees())
        if arretees:
            debut = time.monotonic()
            while True:
                try:
                    libre = int(self.memoire_disponible())
                except Exception:  # noqa: BLE001 - the next measurement will decide
                    break
                if libre >= requis or time.monotonic() - debut >= max(self.delai_remesure_s, 10.0):
                    break
                time.sleep(0.1)
        return libre, bool(arretees)

    # relais

    def relayer(self, famille: str, methode: str, chemin: str, en_tetes: dict[str, str], corps: bytes,
                client_parti: Callable[[], bool] | None = None) -> Reponse:
        """Relaie la demande au fils (voir _relayer). Une demande image ou voix pose drapeau_autres le temps de son
        relay: the video child process does not start a job during this time (PLI_GEN_DRAPEAU_AUTRES)."""

        autre = famille != "video" and famille in self.familles
        voix = self.pause_voix and famille == "audio" and famille in self.familles   # h3 paused during voice
        if famille in self._etats and methode == "POST" and corps[:1] == b"{":
            try:                                   # PLI 5.6.4: the model the child serves, shown by /health
                nom = json.loads(corps).get("model")
                if isinstance(nom, str) and nom:
                    self._etats[famille].modele = nom[:80]
            except (ValueError, UnicodeDecodeError, AttributeError):
                pass
        if autre:
            self._compter_autres(+1)
        if voix:
            self._compter_voix(+1)
        try:
            return self._relayer(famille, methode, chemin, en_tetes, corps, client_parti)
        finally:
            if voix:
                self._compter_voix(-1)
            if autre:
                self._compter_autres(-1)

    def _relayer(self, famille: str, methode: str, chemin: str, en_tetes: dict[str, str], corps: bytes,
                 client_parti: Callable[[], bool] | None = None) -> Reponse:
        """Relays the request to the child. If the relay gives up (timeout, client gone according to client_parti()), the child
        receives POST /annuler to avoid finishing a job that no one is waiting for. Except for video: its creation
        responds 202 immediately and the job continues without the client; /annuler would target the current job, which may
        be another one's (cancelling a video job goes through DELETE /v1/videos/{id})."""

        fam = self.familles.get(famille)
        if fam is None:
            return erreur(404, "famille", f"unknown generation family: {famille}")
        etat = self._etats[famille]
        self._demarrer_fond()
        besoin = self.besoin_demande(fam, methode, chemin, corps)
        with etat.verrou:
            if etat.fils is not None and etat.fils.proc.poll() is not None:
                code = etat.fils.proc.returncode
                raison = etat.fils.raison or "child process stopped"
                type_ = etat.fils.type_erreur
                etat.fils = None                      # the next call restarts it
                return erreur(502, type_, raison, code_sortie=code)
            if etat.fils is None:
                refus = self._lancer(fam, etat, besoin)
                if refus is not None:
                    return refus
            fils = etat.fils
            if etat.actifs == 0:
                etat.echange_base = self._lire_echange()
            etat.actifs += 1
            etat.dernier_usage = time.monotonic()
        arret_veille = threading.Event()
        if client_parti is not None and methode == "POST" and famille != "video":
            threading.Thread(target=self._veiller_client, args=(fam, client_parti, arret_veille), daemon=True).start()
        try:
            reponse = self._http(fam, methode, chemin, en_tetes, corps)
            if reponse[0] == 503 and methode == "POST" and famille in ("audio", "image") and _type_erreur(reponse) == "memoire":
                # the child process refuses to load the requested model: stop other inactive families and retry once
                reserve = self.reserve_octets if fam.reserve_octets is None else fam.reserve_octets
                _, arretees = self._faire_place(fam, max(fam.memoire_octets, besoin) + reserve)
                if arretees:
                    reponse = self._http(fam, methode, chemin, en_tetes, corps)
            return reponse
        except socket.timeout:
            if famille != "video":
                self._annuler_fils(fam)
            return erreur(504, "delai", f"child {famille} did not respond in time")
        except (OSError, http.client.HTTPException):
            return self._apres_echec(etat, fils)
        finally:
            arret_veille.set()
            with etat.verrou:
                etat.actifs -= 1
                etat.dernier_usage = time.monotonic()
                etat.dernier_usage_epoch = time.time()

    def _annuler_fils(self, fam: Famille) -> None:
        """POST /annuler on the child (best effort: a failure or a child without this route is ignored)."""

        conn = http.client.HTTPConnection("127.0.0.1", fam.port, timeout=2.0)
        try:
            conn.request("POST", "/annuler")
            conn.getresponse().read()
        except (OSError, http.client.HTTPException):
            pass
        finally:
            conn.close()

    def _veiller_client(self, fam: Famille, client_parti: Callable[[], bool], arret: threading.Event,
                        pas_s: float = 0.5) -> None:
        while not arret.wait(pas_s):
            try:
                parti = bool(client_parti())
            except Exception:  # noqa: BLE001 - unreadable probe: we cut nothing
                return
            if parti:
                self._annuler_fils(fam)
                return

    def _apres_echec(self, etat: Etat, fils: Fils | None) -> Reponse:
        """Connection to the child failed: it is dead (or stopped by us); the next request restarts it."""

        if fils is not None:
            try:
                fils.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        with etat.verrou:
            if fils is not None and etat.fils is fils and fils.proc.poll() is not None:
                etat.fils = None
        if fils is None:
            return erreur(502, "fils_arrete", "child stopped")
        code = fils.proc.poll()
        return erreur(502, fils.type_erreur, fils.raison or "child stopped", code_sortie=code)

    def _http(self, fam: Famille, methode: str, chemin: str, en_tetes: dict[str, str], corps: bytes) -> Reponse:
        # x-pli-general: set by routes.py when the general model is loaded (the video child refuses if it runs without streaming)
        passe = {k: v for k, v in en_tetes.items() if k.lower() in ("content-type", "accept", "x-pli-general")}
        conn = http.client.HTTPConnection("127.0.0.1", fam.port, timeout=fam.relais_delai_s)
        try:
            conn.request(methode, chemin, body=corps if corps else None, headers=passe)
            rep = conn.getresponse()
            donnees = rep.read()
            type_ = rep.getheader("Content-Type") or "application/octet-stream"
            return rep.status, {"Content-Type": type_}, donnees
        finally:
            conn.close()

    # cycle de vie

    def _memoire_basse(self) -> bool:
        """True if no memory threshold is set, if the measurement fails, or if available memory is below the threshold."""

        if self.dispo_min_echange_octets is None:
            return True
        try:
            return int(self.memoire_disponible()) < self.dispo_min_echange_octets
        except Exception:  # noqa: BLE001 - measurement impossible: we keep the guard
            return True

    def _lire_echange(self) -> int | None:
        try:
            return int(self.echange_octets())
        except Exception:  # noqa: BLE001 - measurement unavailable: no guard, no stop
            return None

    def _lancer(self, fam: Famille, etat: Etat, besoin: int = 0) -> Reponse | None:
        """Checks memory, starts the child, waits for its health. None if the child is running, otherwise the error response.
        besoin: bytes of the requested model (besoin_demande); the largest of it and the family's counts."""

        with self._verrou_liberation:
            self._en_lancement.add(fam.nom)           # the background does not return the general model during this launch
        try:
            return self._lancer_controle(fam, etat, besoin)
        finally:
            with self._verrou_liberation:
                self._en_lancement.discard(fam.nom)

    def _lancer_controle(self, fam: Famille, etat: Etat, besoin: int = 0) -> Reponse | None:
        try:
            libre = int(self.memoire_disponible())
        except Exception as exc:  # noqa: BLE001
            return erreur(503, "memoire", f"available memory unreadable: {exc}")
        reserve = self.reserve_octets if fam.reserve_octets is None else fam.reserve_octets
        if fam.nom == "video" and self.reserve_voix and "audio" in self.familles:
            # the video job waits for video + voice + reserve ("memoire_voix") itself: the child launch does not refuse
            # for a reserve it will obtain while waiting
            reserve = 0
        requis = max(fam.memoire_octets, int(besoin)) + reserve
        if libre < requis:
            # first other loaded but inactive families (deployment of 11/13/2026: voice stayed loaded 10 min and
            # image was refused, 15 GB available for 30 required), before taking the general model's memory; audio
            # stays if the video-voice reserve serves a video (_gardees)
            libre, _ = self._faire_place(fam, requis)
        liberes = 0
        if libre < requis and self.liberer is not None:
            # the general model first frees memory (buffer cache, checkpoints already on disk), then we remeasure
            try:
                liberes = int(self.liberer(requis - libre))
            except Exception as exc:  # noqa: BLE001 - a failed release does not break the request: we measure
                print(f"[gen] general model release failed: {type(exc).__name__}: {exc}", flush=True)
            if liberes > 0:
                with self._verrou_liberation:
                    self._liberees.add(fam.nom)
                debut = time.monotonic()
                while True:
                    try:
                        libre = int(self.memoire_disponible())
                    except Exception as exc:  # noqa: BLE001
                        self._rendre_apres_refus(fam)
                        return erreur(503, "memoire", f"available memory unreadable: {exc}")
                    if libre >= requis or time.monotonic() - debut >= self.delai_remesure_s:
                        break
                    time.sleep(0.1)
        if libre < requis and self.place_fichier and self.place_attente_s > 0:
            libre = self._demander_place(fam, requis, libre)
        if libre < requis and self.mettre_de_cote is not None:
            libre = self._mettre_de_cote(fam, requis, libre)
            if libre >= requis:
                liberes = max(liberes, 1)
        if libre < requis:
            if liberes > 0:
                self._rendre_apres_refus(fam)
            self._place_sans(fam)
            return erreur(503, "memoire",
                          f"insufficient memory to start {fam.nom}: {libre // MO} MB available, "
                          f"{requis // MO} MB required (including {reserve // MO} MB reserve)",
                          disponible_octets=libre, requis_octets=requis, liberes_octets=liberes)
        if self.place_fichier and fam.nom in self._place:
            try:
                with self._verrou_liberation:
                    self._ecrire_place(fam, requis, libre, attente=False)
            except OSError:
                pass
        if liberes == 0 and self.plafonner is not None:
            try:
                if self.plafonner():
                    with self._verrou_liberation:
                        self._liberees.add(fam.nom)      # limits given back by rendre once no such child is alive
            except Exception as exc:  # noqa: BLE001 - a failed cap does not block the launch
                print(f"[gen] general model cap failed: {type(exc).__name__}: {exc}", flush=True)
        argv = self._argv(fam)
        env = {**os.environ, **(self.env or {}), "PLI_GEN_DRAPEAU": self.drapeau,
               "PLI_GEN_DRAPEAU_AUTRES": self.drapeau_autres}
        if fam.nom == "video":
            env.update(self._env_video())
        # its own process group: the stop also reaches grandchildren (h3 launched by the video child)
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
        fils = Fils(proc)
        self._ecrire_pid(fam, proc.pid, argv)
        etat.demarrage = True
        try:
            return self._attendre_sante(fam, etat, fils)
        finally:
            etat.demarrage = False

    # PLI 5.6.4: room asked of the bench queue, then the general model set aside

    def _ecrire_place(self, fam: Famille, requis: int, libre: int, attente: bool = True) -> None:
        if self._place_id is None:
            self._place_id = uuid.uuid4().hex[:12]
        donnees = {"id": self._place_id, "famille": fam.nom, "requis_octets": int(requis), "disponible_octets": int(libre),
                   "manque_octets": max(0, int(requis) - int(libre)), "depuis": time.time(), "pid": os.getpid(),
                   "attente": bool(attente)}      # false: the child has its memory, the request stays for the relaunch only
        temp = self.place_fichier + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(donnees, f)
        os.replace(temp, self.place_fichier)

    def _effacer_place(self) -> None:
        self._place_id = None
        for chemin in (self.place_fichier, self.place_fichier + ".tmp"):
            try:
                os.unlink(chemin)
            except FileNotFoundError:
                pass

    def _demander_place(self, fam: Famille, requis: int, libre: int) -> int:
        """Writes the room request, waits up to place_attente_s for the memory; the file stays if the child launches."""

        try:
            with self._verrou_liberation:
                self._place.add(fam.nom)
                self._ecrire_place(fam, requis, libre)
        except OSError as exc:
            print(f"[gen] room request not written ({self.place_fichier}): {exc}", flush=True)
            return libre
        self.places_demandees += 1
        print(f"[gen] {fam.nom}: {libre // MO} MB available for {requis // MO} MB: room asked of the bench queue, "
              f"up to {self.place_attente_s:.0f} s", flush=True)
        debut = time.monotonic()
        while time.monotonic() - debut < self.place_attente_s and not self._stop.is_set():
            time.sleep(1.0)
            try:
                libre = int(self.memoire_disponible())
            except Exception:  # noqa: BLE001 - measured again next second
                continue
            if libre >= requis:
                print(f"[gen] {fam.nom}: room made in {time.monotonic() - debut:.0f} s ({libre // MO} MB available)",
                      flush=True)
                return libre
        print(f"[gen] {fam.nom}: still {libre // MO} MB for {requis // MO} MB after {time.monotonic() - debut:.0f} s",
              flush=True)
        return libre

    def _place_sans(self, fam: Famille) -> None:
        """A refused launch no longer needs the room it asked for (the file goes once no family holds it)."""

        if not self.place_fichier:
            return
        with self._verrou_liberation:
            self._place.discard(fam.nom)
            if not self._place:
                self._effacer_place()

    def _suivre_place(self) -> None:
        """Each turn: the request stays (touched) while a child launched with it lives; gone once none does."""

        if not self.place_fichier:
            return
        with self._verrou_liberation:
            for nom in list(self._place):
                etat = self._etats.get(nom)
                fils = None if etat is None else etat.fils
                if not (etat is not None and (etat.demarrage or nom in self._en_lancement
                                              or (fils is not None and fils.proc.poll() is None))):
                    self._place.discard(nom)
            if not self._place:
                if self._place_id is not None:
                    self._effacer_place()
                return
        try:
            os.utime(self.place_fichier)
        except OSError:
            pass

    def _mettre_de_cote(self, fam: Famille, requis: int, libre: int) -> int:
        try:
            liberes = int(self.mettre_de_cote(requis - libre))
        except Exception as exc:  # noqa: BLE001 - a failed set-aside is a refusal, as before
            print(f"[gen] general model not set aside: {type(exc).__name__}: {exc}", flush=True)
            return libre
        if liberes <= 0:
            return libre
        self.mises_de_cote += 1
        with self._verrou_liberation:
            self._liberees.add(fam.nom)             # the general model comes back through rendre once no such child lives
        debut = time.monotonic()
        while time.monotonic() - debut < max(self.delai_remesure_s, 5.0):
            try:
                libre = int(self.memoire_disponible())
            except Exception:  # noqa: BLE001
                break
            if libre >= requis:
                break
            time.sleep(0.2)
        print(f"[gen] {fam.nom}: general model set aside ({liberes // MO} MB), {libre // MO} MB available", flush=True)
        return libre

    def _reserve_de_voix(self) -> int:
        """Reserve kept next to the voice: video_reserve_voix_go, otherwise the one requested by the audio launch."""

        if self.reserve_voix_octets is not None:
            return int(self.reserve_voix_octets)
        audio = self.familles.get("audio")
        if audio is not None and audio.reserve_octets is not None:
            return int(audio.reserve_octets)
        return int(self.reserve_octets)

    def _env_video(self) -> dict[str, str]:
        """Video child variables for the voice assistant's place (empty if keys are off: previous behavior)."""

        plus: dict[str, str] = {}
        if self.parallele_video is not None:
            plus["PLI_GEN_DRAPEAU_VIDEO"] = self.drapeau_video
        if self.pause_voix:
            plus["PLI_GEN_DRAPEAU_VOIX"] = self.drapeau_voix
        audio = self.familles.get("audio")
        if self.reserve_voix and audio is not None:
            reserve = self._reserve_de_voix()
            plus.update({"PLI_GEN_VOIX_OCTETS": str(int(audio.memoire_octets) + reserve),
                         "PLI_GEN_VOIX_RESERVE_OCTETS": str(reserve), "PLI_GEN_DRAPEAU_AUDIO": self.drapeau_audio,
                         "PLI_GEN_BESOIN": self.fichier_besoin, "PLI_GEN_ATTENTE_MEMOIRE_S": f"{self.attente_memoire_s:g}"})
        return plus

    def _attendre_sante(self, fam: Famille, etat: Etat, fils: Fils) -> Reponse | None:
        proc = fils.proc
        debut = time.monotonic()
        while True:
            if proc.poll() is not None:
                self._effacer_pid(fam)
                return erreur(502, "fils_arrete", "child stopped at startup", code_sortie=proc.returncode)
            try:
                if self._sante(fam):
                    break
            except Exception:  # noqa: BLE001 - not ready yet
                pass
            if time.monotonic() - debut > fam.sante_delai_s:
                self._arreter_proc(proc)
                self._effacer_pid(fam)
                return erreur(504, "delai", f"child {fam.nom} is not ready after {fam.sante_delai_s:g} s")
            time.sleep(0.02)
        etat.fils = fils
        etat.dernier_usage = time.monotonic()
        etat.dernier_usage_epoch = time.time()
        return None

    def _sante(self, fam: Famille) -> bool:
        if fam.sante is not None:
            return bool(fam.sante(fam))
        conn = http.client.HTTPConnection("127.0.0.1", fam.port, timeout=2.0)
        try:
            conn.request("GET", "/sante")
            rep = conn.getresponse()
            rep.read()
            return rep.status == 200
        except OSError:
            return False
        finally:
            conn.close()

    def _occupe(self, fam: Famille) -> bool:
        """The child works outside any open request (video job in queue): "busy" in its /sante response."""

        if fam.occupe is not None:
            try:
                return bool(fam.occupe(fam))
            except Exception:  # noqa: BLE001
                return False
        conn = http.client.HTTPConnection("127.0.0.1", fam.port, timeout=2.0)
        try:
            conn.request("GET", "/sante")
            rep = conn.getresponse()
            donnees = rep.read()
            return rep.status == 200 and bool(json.loads(donnees or b"{}").get("occupe"))
        except (OSError, ValueError, http.client.HTTPException, AttributeError):
            return False
        finally:
            conn.close()

    def _arreter_proc(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        self._signaler(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=self.delai_kill_s)
        except subprocess.TimeoutExpired:
            self._signaler(proc, signal.SIGKILL)
            proc.wait()
        self._signaler_groupe(proc.pid, signal.SIGKILL)      # a grandchild left alone in the group

    @staticmethod
    def _signaler(proc: subprocess.Popen, sig: int) -> None:
        """Signals the child's group (it is the leader); otherwise, the child alone."""

        try:
            os.killpg(proc.pid, sig)
        except OSError:
            try:
                proc.send_signal(sig)
            except OSError:
                pass

    @staticmethod
    def _signaler_groupe(pgid: int, sig: int) -> bool:
        try:
            os.killpg(pgid, sig)
            return True
        except OSError:
            return False

    # PID files: one per family and port, to find children from a previous launch that died without stopping them

    @staticmethod
    def _argv(fam: Famille) -> list[str]:
        return [a.replace("{port}", str(fam.port)) for a in fam.argv]

    def _chemin_pid(self, fam: Famille) -> str:
        return os.path.join(self.dossier_pid, f"{fam.nom}-{fam.port}.pid")

    def _ecrire_pid(self, fam: Famille, pid: int, argv: list[str]) -> None:
        with open(self._chemin_pid(fam), "w", encoding="utf-8") as f:
            json.dump({"pid": pid, "argv": argv}, f)

    def _effacer_pid(self, fam: Famille) -> None:
        try:
            os.unlink(self._chemin_pid(fam))
        except FileNotFoundError:
            pass

    @staticmethod
    def _ligne_de_commande(pid: int) -> str | None:
        """Process command line (ps), None if it no longer exists."""

        try:
            sortie = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                                    timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
        return sortie or None

    def nettoyer_orphelins(self) -> list[str]:
        """Stops children left by a previous server launch (died without a clean stop).

        Each PID file gives the group leader and its command. If the leader is alive: we only signal if its command
        line matches the recorded one (a PID reused by another program is left alone). If the leader is dead: its
        PID cannot be reassigned while a group member is alive, so the group (h3 left alone) is stopped.
        """

        tues: list[str] = []
        for fam in self.familles.values():
            chemin = self._chemin_pid(fam)
            try:
                with open(chemin, encoding="utf-8") as f:
                    note = json.load(f)
                pid, argv = int(note["pid"]), [str(a) for a in note["argv"]]
            except (OSError, ValueError, KeyError, TypeError):
                self._effacer_pid(fam)
                continue
            ligne = self._ligne_de_commande(pid)
            # The first word is not compared: on macOS, a venv Python restarts via the system Python
            # (Homebrew), and ps shows that path instead of the recorded argv one.
            if ligne is None or ligne.split()[1:] == " ".join(argv).split()[1:]:
                if self._signaler_groupe(pid, signal.SIGTERM):
                    fin = time.monotonic() + 3.0
                    while time.monotonic() < fin and self._signaler_groupe(pid, 0):
                        time.sleep(0.05)
                    self._signaler_groupe(pid, signal.SIGKILL)
                    tues.append(fam.nom)
            self._effacer_pid(fam)
        return tues

    def _arreter_inactives(self, fam: Famille, garder: tuple[str, ...] = ()) -> list[str]:
        """Stops children of other families (except guard) running without an ongoing request; returns their names.

        Family lock taken without waiting: a busy family (or one holding the lock) is not touched, nor a
        child working outside any open request (video in queue or in progress: "busy" in /sante)."""

        arretees: list[str] = []
        for nom, etat in self._etats.items():
            if nom == fam.nom or nom in garder or not etat.verrou.acquire(blocking=False):
                continue
            try:
                fils = etat.fils
                if (fils is not None and fils.proc.poll() is None and etat.actifs == 0 and not etat.demarrage
                        and not self._occupe(self.familles[nom])):
                    self._arreter_fils(etat, f"stopped to make room for {fam.nom}", "place")
                    arretees.append(nom)
            finally:
                etat.verrou.release()
        if arretees:
            print(f"[gen] {', '.join(arretees)} stopped to make room for {fam.nom}", flush=True)
        return arretees

    def _arreter_fils(self, etat: Etat, raison: str | None = None, type_erreur: str = "fils_arrete") -> None:
        fils = etat.fils
        if fils is None:
            return
        fils.raison = raison
        fils.type_erreur = type_erreur
        etat.fils = None
        self._arreter_proc(fils.proc)
        for fam in self.familles.values():
            if self._etats[fam.nom] is etat:
                self._effacer_pid(fam)

    def _rendre_apres_refus(self, fam: Famille) -> None:
        with self._verrou_liberation:
            self._en_lancement.discard(fam.nom)
        self._rendre_si_inutile(True)

    def _rendre_si_inutile(self, refus: bool = False) -> None:
        """Returns the general model when no child launched after a release is alive or starting (except family lock)."""

        if self.rendre is None:
            return
        with self._verrou_liberation:
            if not self._liberees and not refus:
                return
            for nom in self._liberees:
                etat = self._etats[nom]
                fils = etat.fils
                if nom in self._en_lancement or etat.demarrage or (fils is not None and fils.proc.poll() is None):
                    return
            self._liberees = set()
        try:
            self.rendre()
        except Exception as exc:  # noqa: BLE001
            print(f"[gen] general model return failed: {type(exc).__name__}: {exc}", flush=True)

    # background: unloading inactive models and monitoring swap

    def _demarrer_fond(self) -> None:
        with self._verrou_fond:
            if self._fond is None and not self._stop.is_set():
                self._fond = threading.Thread(target=self._boucle, name="pli-gen-superviseur", daemon=True)
                self._fond.start()

    def _boucle(self) -> None:
        while not self._stop.wait(self.intervalle_s):
            self.balayer()

    # the voice assistant's place during a video (except family lock: _lancer holds it until 60 s)

    def _vivant(self, nom: str) -> bool:
        etat = self._etats.get(nom)
        fils = None if etat is None else etat.fils
        return fils is not None and fils.proc.poll() is None

    def _suivre_video(self) -> None:
        if "video" not in self._etats:
            return
        video = self._vivant("video")
        if self.parallele_video is not None and self.parallele is not None:
            tourne = video and os.path.exists(self.drapeau_video)
            if tourne != self._parallele_reduit:
                try:
                    self.parallele(self.parallele_video if tourne else None)
                    self._parallele_reduit = tourne
                    print(f"[gen] video {'in progress' if tourne else 'finished'}: the general model admits "
                          f"{self.parallele_video if tourne else 'its normal number of'} generation(s) at a time",
                          flush=True)
                except Exception as exc:  # noqa: BLE001 - the next loop retries
                    print(f"[gen] general model parallelism not changed: {type(exc).__name__}: {exc}", flush=True)
        if self.reserve_voix and "audio" in self._etats:
            self._marquer_audio(video)
            if video:
                self._servir_besoin()

    def _voix_chargee(self) -> bool:
        """The audio child is alive and (if it says so in /sante, "tts_charge") keeps a synthesis model loaded."""

        etat = self._etats["audio"]
        if not self._vivant("audio") or etat.demarrage:
            return False
        fam = self.familles["audio"]
        if fam.sante is not None:
            return True
        conn = http.client.HTTPConnection("127.0.0.1", fam.port, timeout=2.0)
        try:
            conn.request("GET", "/sante")
            rep = conn.getresponse()
            donnees = json.loads(rep.read() or b"{}")
            return rep.status == 200 and bool(donnees.get("tts_charge", True))
        except (OSError, ValueError, http.client.HTTPException, AttributeError):
            return False
        finally:
            conn.close()

    def _marquer_audio(self, video: bool) -> None:
        charge = video and self._voix_chargee()
        try:
            if charge:
                open(self.drapeau_audio, "w").close()
            else:
                os.unlink(self.drapeau_audio)
        except FileNotFoundError:
            pass

    def _servir_besoin(self) -> None:
        """The video job waits for the voice's memory: the general model returns what is missing (separate child, once per
        delai_besoin_s), then other inactive families except audio if that is not enough."""

        try:
            if time.time() - os.stat(self.fichier_besoin).st_mtime > 5.0:     # the child rewrites it every 0.5 s
                return
            with open(self.fichier_besoin, encoding="utf-8") as f:
                manque = int(json.load(f)["octets"])
        except (OSError, ValueError, KeyError, TypeError):
            return
        if manque <= 0 or time.monotonic() - self._derniere_demande < self.delai_besoin_s:
            return
        if self._liberation_video is not None and self._liberation_video.is_alive():
            return
        self._derniere_demande = time.monotonic()

        def liberer() -> None:
            rendu = 0
            if self.liberer is not None:
                try:
                    rendu = int(self.liberer(manque))
                except Exception as exc:  # noqa: BLE001
                    print(f"[gen] general model release failed: {type(exc).__name__}: {exc}", flush=True)
                if rendu > 0:
                    with self._verrou_liberation:
                        self._liberees.add("video")
            self.liberations_video += 1
            print(f"[gen] video waiting for voice memory: {manque // MO} MB missing, the general model returned "
                  f"{rendu // MO} Mo", flush=True)
            if rendu < manque:
                self._arreter_inactives(self.familles["video"], garder=("audio",))

        self._liberation_video = threading.Thread(target=liberer, name="pli-gen-memoire-voix", daemon=True)
        self._liberation_video.start()

    def balayer(self) -> None:
        """A turn: stop child processes inactive for too long, and image or video ones if swap has increased."""

        try:
            self._suivre_video()
        except Exception as exc:  # noqa: BLE001 - the background does not die
            print(f"[gen] video tracking failed: {type(exc).__name__}: {exc}", flush=True)
        for nom, fam in self.familles.items():
            etat = self._etats[nom]
            with etat.verrou:
                fils = etat.fils
                if fils is None:
                    continue
                if fils.proc.poll() is not None:
                    continue                                   # dead: the next call reports it
                if etat.actifs == 0:
                    if time.monotonic() - etat.dernier_usage > fam.delai_decharge_s:
                        if self._occupe(fam):                  # a job is running in the child process (video): it stays
                            etat.dernier_usage = time.monotonic()
                            etat.dernier_usage_epoch = time.time()
                        else:
                            self._arreter_fils(etat)
                    continue
                surveille = fam.surveiller_echange
                if surveille is None:
                    surveille = nom in FAMILLES_ECHANGE
                if not surveille:
                    continue
                courant = self._lire_echange()
                if courant is None:
                    continue
                if etat.echange_base is None:
                    etat.echange_base = courant
                elif courant - etat.echange_base > self.seuil_echange_octets and self._memoire_basse():
                    self._arreter_fils(
                        etat, f"child process stopped: swap grew by {(courant - etat.echange_base) // MO} MB", "echange")
        self._rendre_si_inutile()
        try:
            self._suivre_place()
        except Exception as exc:  # noqa: BLE001 - the background does not die
            print(f"[gen] room request tracking failed: {type(exc).__name__}: {exc}", flush=True)

    def etat_fils(self, famille: str) -> dict[str, Any]:
        etat = self._etats[famille]
        with etat.verrou:
            vivant = etat.fils is not None and etat.fils.proc.poll() is None
            return {"vivant": vivant, "pid": etat.fils.proc.pid if etat.fils else None, "actifs": etat.actifs}

    def etat_general(self) -> dict[str, Any]:
        """For /health: per family the state (stopped, starting, ready), the pid, the last usage and the declared memory.

        No family lock: _lancer holds it during the entire child process startup.
        """

        sortie: dict[str, Any] = {}
        for nom, fam in self.familles.items():
            etat = self._etats[nom]
            fils = etat.fils
            vivant = fils is not None and fils.proc.poll() is None
            if etat.demarrage:
                nom_etat = "demarrage"
            else:
                nom_etat = "pret" if vivant else "arrete"
            sortie[nom] = {"etat": nom_etat, "pid": fils.proc.pid if vivant else None,
                           "dernier_usage": etat.dernier_usage_epoch or None, "memoire_octets": fam.memoire_octets,
                           # PLI 5.6.4: the model of the last request (while the child lives) and the requests in progress
                           "modele": etat.modele if vivant else None, "actifs": etat.actifs}
        return sortie

    def arreter(self) -> None:
        """Stops the background and all child processes, removes the flag."""

        self._stop.set()
        if self._fond is not None:
            self._fond.join(5)
        for etat in self._etats.values():
            with etat.verrou:
                self._arreter_fils(etat)
        for fam in self.familles.values():
            self._effacer_pid(fam)
        self._rendre_si_inutile()
        if self._parallele_reduit and self.parallele is not None:
            try:
                self.parallele(None)
            except Exception:  # noqa: BLE001 - server shutdown
                pass
            self._parallele_reduit = False
        if self.place_fichier:
            self._effacer_place()
        for chemin in (self.drapeau, self.drapeau_autres, self.drapeau_voix, self.drapeau_video, self.drapeau_audio,
                       self.fichier_besoin):
            try:
                os.unlink(chemin)
            except FileNotFoundError:
                pass
        if self._dossier:
            shutil.rmtree(self._dossier, ignore_errors=True)
