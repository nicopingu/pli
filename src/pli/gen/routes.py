"""OpenAI generation routes for the HTTP server: routing to the supervisor (behind PLI_GEN=1)."""

from __future__ import annotations

import os
import re
import select
import socket
from typing import Any

from pli.gen import garde
from pli.gen.superviseur import Superviseur, erreur
from pli.server.errors import RequestError
from pli.server.request_body import read_body

LIMITE = 128 * 1024**2        # an audio file to transcribe exceeds the limit of text routes

# only with the /v1 prefix: auth_http.gated() keeps only these paths
ROUTES = {
    "/v1/audio/speech": "audio",
    "/v1/audio/transcriptions": "audio",
    "/v1/images/generations": "image",
    "/v1/images/edits": "image",
    "/v1/videos": "video",
}
# reading a video job: state, then file (GET only)
ROUTES_GET = re.compile(r"^/v1/videos/[^/]+(/content)?$")


def famille_de(route: str) -> str | None:
    return ROUTES.get(route)


def famille_de_lecture(route: str) -> str | None:
    return "video" if ROUTES_GET.match(route) else None


def general_charge(app: Any) -> bool:
    """Is the general model loaded in this process?

    The server loads it before listening (cli.py, load then wire_resident) and never unloads it: the only release,
    residency.unwire, happens at shutdown. App.scheduler (app.py:187) exists only when this model is served; an app
    without it (tests, server without model) counts as unloaded. "general_charge" on the app (boolean or
    function) takes precedence, for a future unloading. No "reduced" state exists today: it is not detected.
    """

    surcharge = getattr(app, "general_charge", None)
    if surcharge is not None:
        return bool(surcharge() if callable(surcharge) else surcharge)
    return getattr(app, "scheduler", None) is not None


def refus_general(app: Any) -> bool:
    """True if the video child process must be warned that the general is loaded (header X-PLI-General: charge): it refuses
    then with 503 "general_charge" if it runs without --ssd-streaming. PLI_GEN_VIDEO_AVEC_GENERAL=1: never warned."""

    return general_charge(app) and os.environ.get("PLI_GEN_VIDEO_AVEC_GENERAL") != "1"


def servir(handler: Any, superviseur: Superviseur, famille: str, app: Any = None, methode: str = "POST") -> None:
    """Relays the request to the family child process and returns its response as is (or the supervisor's error).

    Video: with the general model loaded, creation carries the header X-PLI-General: charge; the child decides (accepts in
    --ssd-streaming mode, memory deciding as usual; refuses with 503 "general_charge" without streaming).
    GET and DELETE /v1/videos/{id} relay the reading and cancellation of a job. Image and video (creation) pass
    through the guard (garde.py) before the relay: 400 {"error":{"type":"garde"}}; audio is not filtered.
    """

    corps = b""
    en_tetes = dict(handler.headers.items())
    if methode != "POST" and (handler.headers.get("Content-Length") or handler.headers.get("Transfer-Encoding")):
        try:
            read_body(handler, limit=LIMITE)              # cleared: it must not reach the next request
        except RequestError:
            pass
    if methode == "POST":
        en_tetes = {k: v for k, v in en_tetes.items() if k.lower() != "x-pli-general"}   # set here only
        if famille == "video" and refus_general(app):
            en_tetes["X-PLI-General"] = "charge"
        try:
            corps = read_body(handler, limit=LIMITE)
        except RequestError as exc:
            handler._send_json({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
            return
        g = getattr(app, "garde", None) or garde.Garde.depuis_env()
        avis = None
        if g.llm and g.avis is None and hasattr(app, "chat") and general_charge(app):
            avis = garde.avis_general(app, superviseur)          # second opinion from the 27B of the same process (garde.py)
        verdict = g.verifier(famille, corps, handler.headers.get("Content-Type", ""), avis=avis)
        if verdict.refuse:                                   # no word or reason in the response: only the log keeps the reason
            handler._send_json({"error": {"type": "garde", "message": garde.MESSAGE}}, status=400)
            return
    code, en_tetes, donnees = superviseur.relayer(famille, methode, handler.path, en_tetes, corps,
                                                  client_parti=lambda: _client_parti(handler))
    _repondre(handler, code, en_tetes, donnees)


def _client_parti(handler: Any) -> bool:
    """True if the client closed its connection (reading 1 byte without consuming it: end of stream = gone)."""

    try:
        sock = handler.connection
        lisible, _, _ = select.select([sock], [], [], 0)
        return bool(lisible) and sock.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError, AttributeError):
        return False


def _repondre(handler: Any, code: int, en_tetes: dict[str, str], donnees: bytes) -> None:
    handler.send_response(code)
    handler.send_header("Content-Type", en_tetes.get("Content-Type", "application/octet-stream"))
    handler.send_header("Content-Length", str(len(donnees)))
    if handler.close_connection:
        handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(donnees)


__all__ = ["ROUTES", "famille_de", "famille_de_lecture", "general_charge", "refus_general", "servir", "erreur"]
