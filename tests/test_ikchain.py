# -*- coding: utf-8 -*-
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ikchain as mod  # noqa: E402
ikchain = mod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_hex(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# ---------- 基础工具 ----------

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def bone_lengths(sol):
    """从返回位姿反算两根骨的实际长度。"""
    return dist(sol["base"], sol["elbow"]), dist(sol["elbow"], sol["end"])


def test_reach():
    assert abs(ikchain.reach(3.0, 4.0) - 7.0) < 1e-9


def test_joint_limit_clamps_and_validates():
    assert ikchain.joint_limit(0.5, 0.0, 1.0) == 0.5
    assert ikchain.joint_limit(-1.0, 0.0, 1.0) == 0.0
    assert ikchain.joint_limit(2.0, 0.0, 1.0) == 1.0
    try:
        ikchain.joint_limit(0.0, 1.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("非法限制区间应抛出 ValueError")


# ---------- 1. 求解：末端误差与骨长 ----------

def test_solve_keeps_length():
    sol = ikchain.solve_two_bone(ikchain.Vec2(0, 0), ikchain.Vec2(3, 4), 3.0, 4.0)
    assert sol["ok"] is True
    assert abs(sol["error"]) < 1e-6


def test_endpoint_error_converges():
    # 可达范围内扫一圈，末端必须精确落在目标上
    for ang in [i * math.pi / 6 for i in range(12)]:
        for r in (1.5, 3.0, 5.0, 6.9):
            t = ikchain.Vec2(r * math.cos(ang), r * math.sin(ang))
            sol = ikchain.solve_two_bone(ikchain.Vec2(0, 0), t, 3.0, 4.0)
            assert sol["status"] == "ok"
            assert sol["error"] < 1e-9
            assert dist(sol["end"], t) < 1e-9


def test_bone_lengths_never_change():
    l1, l2 = 3.0, 4.0
    for tx, ty in [(3, 4), (30, 40), (0.3, 0.4), (-3, 4), (0, 0), (7, 0)]:
        sol = ikchain.solve_two_bone(
            ikchain.Vec2(0, 0), ikchain.Vec2(tx, ty), l1, l2
        )
        if sol["status"] == "no_solution":
            continue
        a, b = bone_lengths(sol)
        assert abs(a - l1) < 1e-9, (tx, ty, a)
        assert abs(b - l2) < 1e-9, (tx, ty, b)


# ---------- 2. 超范围：最大伸展 ----------

def test_overreach_max_extension():
    sol = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(30, 40), 3.0, 4.0
    )
    assert sol["ok"] is False
    assert sol["status"] == "max_reach"
    # 最大伸展方向与目标方向一致，末端停在半径 r_max 处
    assert abs(dist(sol["base"], sol["end"]) - 7.0) < 1e-9
    assert abs(sol["end"].x - 4.2) < 1e-9
    assert abs(sol["end"].y - 5.6) < 1e-9
    assert abs(sol["error"] - 43.0) < 1e-9
    assert abs(sol["angles"]["joint"] - math.pi) < 1e-9
    # 参考最大伸展位置与实际末端一致
    assert dist(sol["max_extend_position"], sol["end"]) < 1e-9
    # 骨没有被拉长
    a, b = bone_lengths(sol)
    assert abs(a - 3.0) < 1e-9 and abs(b - 4.0) < 1e-9


# ---------- 3. 关节限制 ----------

def test_limit_in_range_solution():
    # 目标要求的关节角落在限制区间内：正常解，不退化
    sol = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(3, 4), 3.0, 4.0,
        limit=(0.0, math.pi),
    )
    assert sol["status"] == "ok"
    assert sol["limited"] is False
    assert 0.0 <= sol["angles"]["joint"] <= math.pi


def test_limit_degrades_to_boundary():
    # 完全伸直才够得着的目标，但关节不许伸直（β <= π/2）
    sol = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(10, 0), 5.0, 5.0,
        limit=(0.0, math.pi / 2),
    )
    assert sol["ok"] is False
    assert sol["status"] == "limited"
    assert sol["limited"] is True
    # 退化到上边界，绝不越界
    assert abs(sol["angles"]["joint"] - math.pi / 2) < 1e-9
    assert abs(dist(sol["base"], sol["end"]) - 5.0 * math.sqrt(2)) < 1e-9
    assert sol["error"] > 0.0
    assert "限制边界" in sol["reason"]
    # 骨长不变
    a, b = bone_lengths(sol)
    assert abs(a - 5.0) < 1e-9 and abs(b - 5.0) < 1e-9


def test_limit_lower_boundary_degradation():
    # 内圈目标要求完全折叠（β=0），但限制要求 β >= 1.0：退化到下边界
    sol = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(0, 0), 5.0, 5.0,
        limit=(1.0, 2.0),
    )
    assert sol["status"] == "limited"
    assert abs(sol["angles"]["joint"] - 1.0) < 1e-9
    assert "限制边界" in sol["reason"]


def test_limit_always_respected_in_sweep():
    # 任意目标，只要给出了位姿，关节角就必须落在限制区间内
    lo, hi = 0.3, 2.5
    for r in (0.0, 0.5, 2.0, 7.0, 20.0):
        for ang in (0.0, 1.1, 3.3, 5.0):
            sol = ikchain.solve_two_bone(
                ikchain.Vec2(0, 0),
                ikchain.Vec2(r * math.cos(ang), r * math.sin(ang)),
                3.0, 4.0, limit=(lo, hi),
            )
            if sol["angles"]["joint"] is not None:
                assert lo - 1e-12 <= sol["angles"]["joint"] <= hi + 1e-12


def test_infeasible_limit_returns_no_solution():
    # 区间颠倒
    sol = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(3, 4), 3.0, 4.0, limit=(2.0, 1.0)
    )
    assert sol["ok"] is False
    assert sol["status"] == "no_solution"
    assert sol["elbow"] is None and sol["end"] is None
    assert sol["error"] is None and sol["angles"]["joint"] is None

    # 区间与关节物理范围 [0, π] 无交集
    sol2 = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(3, 4), 3.0, 4.0, limit=(-3.0, -2.0)
    )
    assert sol2["status"] == "no_solution"
    assert sol2["end"] is None

    # 负骨长同样无解
    sol3 = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(3, 4), -3.0, 4.0
    )
    assert sol3["status"] == "no_solution"
    assert sol3["end"] is None


# ---------- 4. 边界情形 ----------

def test_target_equals_base_unequal_bones():
    sol = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(0, 0), 3.0, 4.0
    )
    # 确定结果：折叠到最小半径 |l1-l2|，方向取 +x
    assert sol["status"] == "min_reach"
    assert sol["ok"] is False
    assert abs(sol["error"] - 1.0) < 1e-9
    assert abs(sol["angles"]["joint"]) < 1e-9
    a, b = bone_lengths(sol)
    assert abs(a - 3.0) < 1e-9 and abs(b - 4.0) < 1e-9


def test_target_equals_base_equal_bones():
    sol = ikchain.solve_two_bone(
        ikchain.Vec2(2, 2), ikchain.Vec2(2, 2), 3.0, 3.0
    )
    # 等长骨折叠后末端恰好回到基座
    assert sol["ok"] is True
    assert sol["error"] < 1e-9
    assert dist(sol["end"], ikchain.Vec2(2, 2)) < 1e-9


def test_zero_length_bones():
    # l1 = 0：退化为长 l2 的单段直骨
    sol = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(10, 0), 0.0, 4.0
    )
    assert sol["status"] == "max_reach"
    assert abs(sol["error"] - 6.0) < 1e-9
    assert abs(sol["end"].x - 4.0) < 1e-9
    assert sol["angles"]["joint"] is None

    # 恰好伸直可达
    sol2 = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(4, 0), 0.0, 4.0
    )
    assert sol2["ok"] is True and sol2["error"] < 1e-9

    # l2 = 0：肘即末端
    sol3 = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(3, 0), 3.0, 0.0
    )
    assert sol3["ok"] is True
    assert dist(sol3["elbow"], sol3["end"]) < 1e-9

    # 两根都为零：末端恒在基座
    sol4 = ikchain.solve_two_bone(
        ikchain.Vec2(1, 1), ikchain.Vec2(5, 5), 0.0, 0.0
    )
    assert dist(sol4["end"], ikchain.Vec2(1, 1)) < 1e-9
    assert abs(sol4["error"] - math.hypot(4, 4)) < 1e-9


def test_exactly_reachable():
    # 外圈边界：d = l1 + l2，恰好完全伸直可达
    sol = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(7, 0), 3.0, 4.0
    )
    assert sol["ok"] is True and sol["error"] < 1e-9
    assert abs(sol["angles"]["joint"] - math.pi) < 1e-9
    assert sol["status"] == "ok"

    # 内圈边界：d = |l1 - l2|，恰好完全折叠可达
    sol2 = ikchain.solve_two_bone(
        ikchain.Vec2(0, 0), ikchain.Vec2(1, 0), 3.0, 4.0
    )
    assert sol2["ok"] is True and sol2["error"] < 1e-9
    assert abs(sol2["angles"]["joint"]) < 1e-9


# ---------- 5. 确定性与肘朝向 ----------

def test_deterministic_two_runs_identical():
    args_list = [
        (ikchain.Vec2(0, 0), ikchain.Vec2(3, 4), 3.0, 4.0),
        (ikchain.Vec2(0, 0), ikchain.Vec2(30, 40), 3.0, 4.0),
        (ikchain.Vec2(1, -1), ikchain.Vec2(0, 0), 3.0, 4.0),
    ]
    for args in args_list:
        a = ikchain.solve_two_bone(*args)
        b = ikchain.solve_two_bone(*args)
        assert a == b


def test_bend_mirror():
    t = ikchain.Vec2(3, 4)
    up = ikchain.solve_two_bone(ikchain.Vec2(0, 0), t, 3.0, 4.0, bend=1.0)
    down = ikchain.solve_two_bone(ikchain.Vec2(0, 0), t, 3.0, 4.0, bend=-1.0)
    assert up["elbow"] != down["elbow"]
    # 两端末端重合、误差与关节角相同
    assert dist(up["end"], down["end"]) < 1e-9
    assert abs(up["error"] - down["error"]) < 1e-12
    assert abs(up["angles"]["joint"] - down["angles"]["joint"]) < 1e-12
    # 两肘关于基座→目标连线对称：中点在该连线上
    mx = (up["elbow"].x + down["elbow"].x) / 2
    my = (up["elbow"].y + down["elbow"].y) / 2
    cross = mx * t.y - my * t.x
    assert abs(cross) < 1e-9


# ---------- 6. CLI 端到端 ----------

def test_cli_end_to_end(tmp_path):
    out = tmp_path / "solution.json"
    proc = subprocess.run(
        [sys.executable, "-m", "ikchain.cli",
         os.path.join(ROOT, "data", "targets.json"), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(read_text(str(out)))
    assert len(data) == 2
    assert data[0]["status"] == "ok" and data[0]["error"] < 1e-9
    assert data[1]["status"] == "max_reach"
    assert data[1]["lengths"] == [3.0, 4.0]


# ===== v2 契约测试：初始全部失败，实现后必须通过 =====
import ikchain as broadphase_ik  # noqa: E402


def test_chain_keeps_bone_lengths():
    import math
    r = broadphase_ik.solve_chain((0.0, 0.0), (2.0, 1.0), [1.0, 1.0, 1.0])
    js = r["joints"]
    for a, b in zip(js, js[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        assert abs(seg - 1.0) < 1e-6


def test_chain_endpoint_error_within_tol():
    r = broadphase_ik.solve_chain((0.0, 0.0), (2.0, 0.5), [1.2, 1.2, 1.2], tol=0.02)
    assert r["end_error"] <= 0.02
    assert r["iterations"] >= 1


def test_chain_infeasible_reported():
    r = broadphase_ik.solve_chain((0.0, 0.0), (10.0, 0.0), [1.0, 1.0, 1.0])
    assert r.get("infeasible") is True
    assert r.get("reason")


def test_frozen_joints_listed():
    limits = [(None, None), (-0.2, 0.2), (None, None)]
    r = broadphase_ik.solve_chain((0.0, 0.0), (0.0, 2.5), [1.0, 1.0, 1.0], limits=limits, tol=0.05)
    assert "frozen" in r
    assert isinstance(r["frozen"], list)


def test_avoid_circle_not_entered():
    import math
    r = broadphase_ik.solve_chain((0.0, 0.0), (2.0, 0.0), [1.0, 1.0, 1.0],
                                  avoid=((1.0, 0.0), 0.3))
    if r.get("infeasible"):
        assert r.get("reason")
    else:
        for p in r["joints"]:
            assert math.hypot(p[0] - 1.0, p[1] - 0.0) >= 0.3 - 1e-9
