# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ikchain as mod  # noqa: E402


def test_reach():
    assert abs(ikchain.reach(3.0, 4.0) - 7.0) < 1e-9


def test_solve_keeps_length():
    sol = ikchain.solve_two_bone(ikchain.Vec2(0, 0), ikchain.Vec2(3, 4), 3.0, 4.0)
    assert sol["ok"] is True
    assert abs(sol["error"]) < 1e-6
