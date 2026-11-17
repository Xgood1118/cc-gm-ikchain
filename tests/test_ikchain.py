# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ikchain as mod  # noqa: E402
ikchain = mod

def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_hex(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def test_reach():
    assert abs(ikchain.reach(3.0, 4.0) - 7.0) < 1e-9


def test_solve_keeps_length():
    sol = ikchain.solve_two_bone(ikchain.Vec2(0, 0), ikchain.Vec2(3, 4), 3.0, 4.0)
    assert sol["ok"] is True
    assert abs(sol["error"]) < 1e-6
