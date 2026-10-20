"""``kill -USR1`` prints every thread's stack while serving, after a library took the signal as Triton's LLVM does."""

import faulthandler
import http.client
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from tensorfold.server import stacks

HEADER = "most recent call first"


def _dumps(capfd, tries: int) -> bool:
    """Whether USR1 prints a stack within ``tries`` signals (a server re-arms just after writing its response)."""

    for _ in range(tries):
        os.kill(os.getpid(), signal.SIGUSR1)
        time.sleep(0.05)
        if HEADER in capfd.readouterr().err:
            return True
    return False




def test_arming_does_nothing_in_a_process_that_never_started_the_dump(monkeypatch):
    monkeypatch.setattr(stacks, "_started", False)
    calls = []
    monkeypatch.setattr("faulthandler.register", lambda *a, **k: calls.append(a))
    stacks.arm()                                             # an embedding host keeps its own USR1 handler
    assert calls == []
