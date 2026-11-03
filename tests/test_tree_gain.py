"""PLI (30/10/2026): the KFlash tree is kept only if it brings PLI_TREE_GAIN_MIN more expected tokens than the chain."""

import importlib
import os

import numpy as np


def _kf(gain: str):
    os.environ["PLI_TREE_GAIN_MIN"] = gain
    from tensorfold.pli import kflash
    return importlib.reload(kflash)


def _lattice(sure: bool):
    rng = np.random.default_rng(0)
    D, K, R, V = 7, 16, 8, 64
    cands = np.stack([rng.permutation(V)[:K] for _ in range(D)])
    unary = np.full((D, K), -4.0)
    unary[:, 0] = 6.0 if sure else 0.3            # sure: one candidate crushes the others; unsure: two close candidates
    if not sure:
        unary[:, 1] = 0.2
    hproj = np.zeros((D, R))
    pred = np.zeros((V, R), dtype=np.float32)
    succ = np.zeros((V, R), dtype=np.float32)
    return cands, unary, hproj, int(cands[0][0]), pred, succ


def test_chain_kept_when_the_draft_is_sure():
    kf = _kf("0.15")
    t, p, s, n = kf.choose(*_lattice(True), 7, kf.Calibration())
    assert p == list(range(-1, 6)) and kf.CHOICES["chain"] == 1


def test_tree_kept_when_the_draft_hesitates():
    kf = _kf("0.15")
    t, p, s, n = kf.choose(*_lattice(False), 7, kf.Calibration())
    assert p != list(range(-1, 6)) and kf.CHOICES["tree"] == 1


def test_without_the_setting_always_the_tree():
    kf = _kf("")
    a = kf.build(*_lattice(True), 7, kf.Calibration())
    b = kf.choose(*_lattice(True), 7, kf.Calibration())
    assert a[:2] == b[:2]
