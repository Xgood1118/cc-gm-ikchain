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


# ===== v2 契约测试：初始全部失败，实现后必须通过 =====
import ikchain  # noqa: E402



def test_chain_keeps_lengths_and_converges():
    import math
    r = ikchain.solve_chain((0.0, 0.0), (2.0, 1.0), [1.0, 1.0, 1.0], tol=0.02)
    assert r["end_error"] <= 0.02
    assert r["iterations"] >= 1
    js = r["joints"]
    for a, b in zip(js, js[1:]):
        assert abs(math.hypot(b[0] - a[0], b[1] - a[1]) - 1.0) < 1e-6


def test_chain_infeasible_reported():
    r = ikchain.solve_chain((0.0, 0.0), (10.0, 0.0), [1.0, 1.0, 1.0])
    assert r.get("infeasible") is True
    assert r.get("reason")


def test_frozen_joints_listed():
    limits = [(None, None), (-0.2, 0.2), (None, None)]
    r = ikchain.solve_chain((0.0, 0.0), (0.0, 2.5), [1.0, 1.0, 1.0], limits=limits, tol=0.05)
    assert isinstance(r["frozen"], list)
    assert r["end_error"] >= 0.0


def test_avoid_circle_respected():
    import math
    r = ikchain.solve_chain((0.0, 0.0), (2.0, 0.0), [1.0, 1.0, 1.0], avoid=((1.0, 0.0), 0.3))
    if r.get("infeasible"):
        assert r.get("reason")
    else:
        for p in r["joints"]:
            assert math.hypot(p[0] - 1.0, p[1]) >= 0.3 - 1e-9


def test_limit_degrade_reports_error():
    limits = [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
    r = ikchain.solve_chain((0.0, 0.0), (2.0, 2.0), [1.0, 1.0, 1.0], limits=limits, tol=0.01)
    assert "end_error" in r
    assert r["end_error"] > 0.0
