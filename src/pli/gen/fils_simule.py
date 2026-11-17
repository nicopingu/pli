"""Simulated generation child process, for supervisor tests: no model, only fake responses.

    python -m pli.gen.fils_simule --port N [--etapes K] [--duree-etape S] [--planter-apres N] [--memoire-mo M]

Routes : GET /sante ; POST /v1/audio/speech, /v1/audio/transcriptions, /v1/images/generations, /v1/images/edits.
For images, K steps; before each step the child waits as long as the PLI_GEN_DRAPEAU file exists
(the general model is working: preemption between two steps).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ATTENTE_MAX_S = 600.0     # a flag never removed does not block a child forever
PAS_ATTENTE_S = 0.02


def faire_serveur(port: int, *, etapes: int = 1, duree_etape: float = 0.0, planter_apres: int = 0,
                  memoire_mo: int = 0) -> ThreadingHTTPServer:
    drapeau = os.environ.get("PLI_GEN_DRAPEAU", "")
    verrou = threading.Lock()
    compteur = [0]
    # bytes actually touched, so that the simulated memory is resident
    tampon = bytearray(memoire_mo * 1024 * 1024)
    for i in range(0, len(tampon), 4096):
        tampon[i] = 1

    def attendre_drapeau() -> int:
        """Number of waits spent waiting behind the flag (0 if the path is free)."""

        if not drapeau:
            return 0
        debut = time.monotonic()
        attentes = 0
        while os.path.exists(drapeau) and time.monotonic() - debut < ATTENTE_MAX_S:
            attentes += 1
            time.sleep(PAS_ATTENTE_S)
        return attentes

    class Gestionnaire(BaseHTTPRequestHandler):
        def log_message(self, format, *args):      # silencieux
            pass

        def _envoyer(self, code: int, corps: bytes, type_: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", type_)
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)
            self.wfile.flush()

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] == "/sante":
                self._envoyer(200, b'{"ok": true}', "application/json")
            else:
                self._envoyer(404, b'{"error": {"message": "inconnu"}}', "application/json")

        def do_POST(self) -> None:
            longueur = int(self.headers.get("Content-Length") or 0)
            if longueur:
                self.rfile.read(longueur)
            chemin = self.path.split("?", 1)[0].rstrip("/")
            if chemin == "/v1/audio/speech":
                self._envoyer(200, b"RIFF" + bytes(64), "audio/wav")
            elif chemin == "/v1/audio/transcriptions":
                self._envoyer(200, json.dumps({"text": "simulated text"}).encode(), "application/json")
            elif chemin in ("/v1/images/generations", "/v1/images/edits"):
                attentes = 0
                for _ in range(max(1, etapes)):
                    attentes += attendre_drapeau()
                    time.sleep(duree_etape)
                charge = {"created": int(time.time()),
                          "data": [{"b64_json": base64.b64encode(b"image-simulee").decode()}],
                          "simule": {"etapes": max(1, etapes), "attentes": attentes}}
                self._envoyer(200, json.dumps(charge).encode(), "application/json")
            else:
                self._envoyer(404, b'{"error": {"message": "inconnu"}}', "application/json")
                return
            with verrou:
                compteur[0] += 1
                fini = planter_apres > 0 and compteur[0] >= planter_apres
            if fini:
                os._exit(3)       # brutal crash after the N-th response

    serveur = ThreadingHTTPServer(("127.0.0.1", port), Gestionnaire)
    serveur.tampon = tampon       # type: ignore[attr-defined]
    return serveur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--etapes", type=int, default=1)
    p.add_argument("--duree-etape", type=float, default=0.0)
    p.add_argument("--planter-apres", type=int, default=0)
    p.add_argument("--memoire-mo", type=int, default=0)
    a, _ = p.parse_known_args()          # --catalogue, --voix: accepted and ignored (supervisor tests)
    faire_serveur(a.port, etapes=a.etapes, duree_etape=a.duree_etape, planter_apres=a.planter_apres,
                  memoire_mo=a.memoire_mo).serve_forever()


if __name__ == "__main__":
    main()
