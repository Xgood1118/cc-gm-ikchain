# -*- coding: utf-8 -*-
"""ikchain 契约测试：两骨解析解、关节限位、边界退化、链式 CCD、规避。"""

import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ikchain  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOL = 1e-9


# ---------------------------------------------------------------------------
# 基础函数
# ---------------------------------------------------------------------------

def test_reach():
    assert abs(ikchain.reach(3.0, 4.0) - 7.0) < TOL
    assert ikchain.reach(0.0, 5.0) == 5.0
    assert ikchain.reach(0.0, 0.0) == 0.0


def test_joint_limit_clamps_both_sides():
    assert ikchain.joint_limit(0.5, 0.0, 1.0) == 0.5
    assert ikchain.joint_limit(-0.2, 0.0, 1.0) == 0.0
    assert ikchain.joint_limit(1.5, 0.0, 1.0) == 1.0
    # None 表示该侧无界
    assert ikchain.joint_limit(-3.0, None, 0.0) == -3.0
    assert ikchain.joint_limit(3.0, 0.0, None) == 3.0


# ---------------------------------------------------------------------------
# 两骨：可达解与骨长守恒
# ---------------------------------------------------------------------------

def test_solve_keeps_length():
    sol = ikchain.solve_two_bone(ikchain.Vec2(0, 0), ikchain.Vec2(3, 4), 3.0, 4.0)
    assert sol["ok"] is True
    assert sol["status"] == "ok"
    assert abs(sol["error"]) < 1e-6
    assert abs(math.hypot(*sol["elbow"]) - 3.0) < 1e-9
    e = sol["elbow"]
    assert abs(math.hypot(sol["end"][0] - e[0], sol["end"][1] - e[1]) - 4.0) < 1e-9
    # 末端精确到达目标
    assert abs(sol["end"][0] - 3.0) < 1e-9
    assert abs(sol["end"][1] - 4.0) < 1e-9


def test_solve_general_point_with_offset_base():
    base = ikchain.Vec2(1.0, -2.0)
    target = ikchain.Vec2(4.0, 2.0)
    sol = ikchain.solve_two_bone(base, target, 2.0, 3.0)
    assert sol["ok"] is True
    assert abs(math.hypot(sol["elbow"][0] - 1.0, sol["elbow"][1] + 2.0) - 2.0) < 1e-9
    assert math.hypot(
        sol["end"][0] - sol["elbow"][0], sol["end"][1] - sol["elbow"][1]
    ) - 3.0 < 1e-9
    assert sol["error"] < 1e-9


def test_bend_mirror_solutions_both_valid():
    a = ikchain.solve_two_bone((0, 0), (3, 0), 2.0, 2.0, bend=1.0)
    b = ikchain.solve_two_bone((0, 0), (3, 0), 2.0, 2.0, bend=-1.0)
    assert a["ok"] and b["ok"]
    # 两侧关于目标方向（x 轴）镜像
    assert b["elbow"][1] == -a["elbow"][1]
    assert a["elbow"][0] == b["elbow"][0]
    assert abs(a["elbow"][1]) > 0.0


# ---------------------------------------------------------------------------
# 两骨：超距 -> 最大伸展，不拉长骨
# ---------------------------------------------------------------------------

def test_beyond_reach_max_extend_no_stretch():
    sol = ikchain.solve_two_bone((0, 0), (30, 40), 3.0, 4.0)
    assert sol["status"] == "max_reach"
    assert sol["ok"] is False
    # 末端停在最大伸展位置（沿目标方向 7.0）
    assert abs(math.hypot(*sol["end"]) - 7.0) < 1e-9
    assert math.hypot(*sol["elbow"]) - 3.0 < 1e-9
    assert math.hypot(
        sol["end"][0] - sol["elbow"][0], sol["end"][1] - sol["elbow"][1]
    ) - 4.0 < 1e-9
    # 残余误差 = 目标距离 - 最大可达
    assert abs(sol["error"] - (50.0 - 7.0)) < 1e-9
    assert abs(math.pi - sol["angles"]["joint"]) < 1e-9
    assert sol["max_extend_position"] == sol["end"]


def test_too_close_max_fold():
    # l1=5, l2=3，最近可达 2；目标距离 1 -> 最大折叠
    sol = ikchain.solve_two_bone((0, 0), (1, 0), 5.0, 3.0)
    assert sol["status"] == "min_reach"
    assert sol["ok"] is False
    assert abs(sol["end"][0] - 2.0) < 1e-9
    assert math.hypot(*sol["elbow"]) - 5.0 < 1e-9
    assert math.hypot(
        sol["end"][0] - sol["elbow"][0], sol["end"][1] - sol["elbow"][1]
    ) - 3.0 < 1e-9
    assert abs(sol["error"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 两骨：关节限位
# ---------------------------------------------------------------------------

def test_joint_limit_respected_and_degrades_to_boundary():
    # l1=l2=2, 目标 (3,0)：自然内角 acos(-1/8) ≈ 1.696 rad
    natural = math.acos(-1.0 / 8.0)
    sol_free = ikchain.solve_two_bone((0, 0), (3, 0), 2.0, 2.0)
    assert abs(sol_free["angles"]["joint"] - natural) < 1e-9

    sol = ikchain.solve_two_bone((0, 0), (3, 0), 2.0, 2.0, limit=(None, 1.0))
    assert sol["status"] == "limited"
    assert sol["limited"] is True
    assert abs(sol["angles"]["joint"] - 1.0) < 1e-9
    assert sol["error"] > 0.0
    # 骨长仍守恒
    assert math.hypot(*sol["elbow"]) - 2.0 < 1e-9
    assert math.hypot(
        sol["end"][0] - sol["elbow"][0], sol["end"][1] - sol["elbow"][1]
    ) - 2.0 < 1e-9


def test_joint_limit_lo_boundary():
    # 完全伸展内角 ≈ pi，强制内角 >= 3.0
    sol = ikchain.solve_two_bone((0, 0), (1, 0), 1.0, 1.0, limit=(3.0, None))
    assert sol["status"] == "limited"
    assert abs(sol["angles"]["joint"] - 3.0) < 1e-9
    assert sol["error"] > 0.0


def test_limit_infeasible_no_out_of_range_pose():
    # 内角不可能 > pi
    sol = ikchain.solve_two_bone((0, 0), (3, 0), 2.0, 2.0, limit=(4.0, None))
    assert sol["status"] == "infeasible"
    assert sol["ok"] is False
    assert sol["elbow"] is None and sol["end"] is None
    assert sol["reason"]
    # 空限位区间
    sol2 = ikchain.solve_two_bone((0, 0), (3, 0), 2.0, 2.0, limit=(1.0, 0.5))
    assert sol2["status"] == "infeasible"
    # 内角不可能为负
    sol3 = ikchain.solve_two_bone((0, 0), (3, 0), 2.0, 2.0, limit=(None, -0.1))
    assert sol3["status"] == "infeasible"


# ---------------------------------------------------------------------------
# 两骨：边界与退化
# ---------------------------------------------------------------------------

def test_target_coincides_with_base_equal_bones():
    sol = ikchain.solve_two_bone((0, 0), (0, 0), 2.0, 2.0)
    assert sol["ok"] is True
    assert sol["status"] == "ok"
    assert sol["error"] == 0.0
    assert math.hypot(*sol["elbow"]) - 2.0 < 1e-9
    assert math.hypot(
        sol["end"][0] - sol["elbow"][0], sol["end"][1] - sol["elbow"][1]
    ) - 2.0 < 1e-9


def test_target_coincides_with_base_unequal_bones():
    sol = ikchain.solve_two_bone((0, 0), (0, 0), 3.0, 1.0)
    # 最近可达距离 2 > 0，按最大折叠处理，但仍给出确定姿态
    assert sol["status"] == "min_reach"
    assert abs(sol["end"][0] - 2.0) < 1e-9
    assert math.hypot(*sol["elbow"]) - 3.0 < 1e-9
    assert math.hypot(
        sol["end"][0] - sol["elbow"][0], sol["end"][1] - sol["elbow"][1]
    ) - 1.0 < 1e-9


def test_zero_length_first_bone():
    sol = ikchain.solve_two_bone((0, 0), (3, 4), 0.0, 5.0)
    assert sol["ok"] is True
    assert sol["elbow"] == (0.0, 0.0)
    assert abs(math.hypot(*sol["end"]) - 5.0) < 1e-9
    assert sol["error"] < 1e-9


def test_zero_length_second_bone():
    sol = ikchain.solve_two_bone((0, 0), (3, 4), 5.0, 0.0)
    assert sol["ok"] is True
    assert abs(math.hypot(*sol["elbow"]) - 5.0) < 1e-9
    assert sol["end"] == sol["elbow"]


def test_both_bones_zero_length():
    sol = ikchain.solve_two_bone((0, 0), (0, 0), 0.0, 0.0)
    assert sol["status"] == "ok"
    assert sol["elbow"] == (0.0, 0.0) and sol["end"] == (0.0, 0.0)
    assert sol["error"] == 0.0


def test_exact_max_reach_deterministic():
    # 目标距离恰好等于 l1+l2
    sol = ikchain.solve_two_bone((0, 0), (7, 0), 3.0, 4.0)
    assert sol["status"] == "ok"
    assert sol["error"] < 1e-9
    assert abs(sol["elbow"][0] - 3.0) < 1e-9
    assert abs(abs(sol["angles"]["joint"]) - math.pi) < 1e-9


def test_exact_min_reach_deterministic():
    # 目标距离恰好等于 |l1-l2|
    sol = ikchain.solve_two_bone((0, 0), (2, 0), 5.0, 3.0)
    assert sol["status"] == "ok"
    assert sol["error"] < 1e-9
    assert abs(sol["angles"]["joint"]) < 1e-9


def test_deterministic_two_calls_identical():
    kw = dict(bend=1.0)
    a = ikchain.solve_two_bone((1, 2), (6, -3), 3.5, 4.5, **kw)
    b = ikchain.solve_two_bone((1, 2), (6, -3), 3.5, 4.5, **kw)
    assert a == b


# ---------------------------------------------------------------------------
# 链式求解
# ---------------------------------------------------------------------------

def _chain_bone_lengths(js):
    return [
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(js[:-1], js[1:])
    ]


def test_chain_keeps_lengths_and_converges():
    r = ikchain.solve_chain((0.0, 0.0), (2.0, 1.0), [1.0, 1.0, 1.0], tol=0.02)
    assert r.get("infeasible") is False
    assert r["end_error"] <= 0.02
    assert r["residual"] == r["end_error"]
    assert r["iterations"] >= 1
    js = r["joints"]
    assert js[0] == (0.0, 0.0)
    for got, want in zip(_chain_bone_lengths(js), [1.0, 1.0, 1.0]):
        assert abs(got - want) < 1e-9


def test_chain_four_bone_chain():
    r = ikchain.solve_chain((0.0, 0.0), (1.5, -2.0),
                            [0.8, 1.2, 1.0, 0.7], tol=0.01, max_iter=128)
    assert r["infeasible"] is False
    assert r["end_error"] <= 0.01
    for got, want in zip(_chain_bone_lengths(r["joints"]),
                         [0.8, 1.2, 1.0, 0.7]):
        assert abs(got - want) < 1e-9


def test_chain_infeasible_reported():
    r = ikchain.solve_chain((0.0, 0.0), (10.0, 0.0), [1.0, 1.0, 1.0])
    assert r.get("infeasible") is True
    assert r.get("reason")
    assert r.get("joints") is None


def test_chain_zero_length_bone():
    r = ikchain.solve_chain((0.0, 0.0), (2.0, 0.0),
                            [1.0, 0.0, 1.0], tol=0.01)
    assert r["infeasible"] is False
    assert r["end_error"] <= 0.01
    for got, want in zip(_chain_bone_lengths(r["joints"]), [1.0, 0.0, 1.0]):
        assert abs(got - want) < 1e-9


def test_chain_deterministic():
    kw = dict(tol=0.01)
    a = ikchain.solve_chain((0.0, 0.0), (2.0, 1.0), [1.0, 1.0, 1.0], **kw)
    b = ikchain.solve_chain((0.0, 0.0), (2.0, 1.0), [1.0, 1.0, 1.0], **kw)
    assert a["joints"] == b["joints"]
    assert a["iterations"] == b["iterations"]
    assert a["end_error"] == b["end_error"]
    assert a["frozen"] == b["frozen"]


# ---------------------------------------------------------------------------
# 链式：限位、冻结、降级
# ---------------------------------------------------------------------------

def test_frozen_joints_listed():
    limits = [(None, None), (-0.2, 0.2), (None, None)]
    r = ikchain.solve_chain((0.0, 0.0), (0.0, 2.5),
                            [1.0, 1.0, 1.0], limits=limits, tol=0.05)
    assert r["infeasible"] is False
    assert isinstance(r["frozen"], list)
    assert 1 in r["frozen"]
    assert r["end_error"] >= 0.0
    for got in _chain_bone_lengths(r["joints"]):
        assert abs(got - 1.0) < 1e-9
    # 冻结关节的实际弯折角必须在限位内
    p = r["joints"]
    a0 = math.atan2(p[1][1] - p[0][1], p[1][0] - p[0][0])
    a1 = math.atan2(p[2][1] - p[1][1], p[2][0] - p[1][0])
    rel = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    assert -0.2 - 1e-6 <= rel <= 0.2 + 1e-6


def test_limit_degrade_reports_error():
    limits = [(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)]
    r = ikchain.solve_chain((0.0, 0.0), (2.0, 2.0),
                            [1.0, 1.0, 1.0], limits=limits, tol=0.01)
    assert r["infeasible"] is False
    assert "end_error" in r
    assert r["end_error"] > 0.0
    assert r["converged"] is False
    assert r["frozen"]  # 降级时至少报告一个冻结关节
    assert r["reason"]


def test_multiple_frozen_joints_priority_order():
    # 三个关节全部严格限位：冻结下标按优先级（从基座起）有序列出
    limits = [(-0.05, 0.05), (-0.05, 0.05), (-0.05, 0.05)]
    r = ikchain.solve_chain((0.0, 0.0), (1.0, 2.0),
                            [1.0, 1.0, 1.0], limits=limits, tol=0.01)
    assert r["infeasible"] is False
    assert r["frozen"] == sorted(r["frozen"])
    assert len(r["frozen"]) >= 2
    assert all(0 <= i < 3 for i in r["frozen"])
    assert r["end_error"] > 0.0  # 冻结后误差放大，但给出实际值


def test_chain_empty_limit_range_infeasible():
    limits = [(1.0, 0.5), (None, None), (None, None)]
    r = ikchain.solve_chain((0.0, 0.0), (2.0, 1.0),
                            [1.0, 1.0, 1.0], limits=limits)
    assert r["infeasible"] is True
    assert r["reason"]


# ---------------------------------------------------------------------------
# 链式：禁入圆规避
# ---------------------------------------------------------------------------

def test_avoid_circle_respected():
    r = ikchain.solve_chain((0.0, 0.0), (2.0, 0.0), [1.0, 1.0, 1.0],
                            avoid=((1.0, 0.0), 0.3))
    if r.get("infeasible"):
        assert r.get("reason")
    else:
        js = r["joints"]
        # 关节点
        for p in js:
            assert math.hypot(p[0] - 1.0, p[1]) >= 0.3 - 1e-9
        # 每一段骨也不得穿圆
        for a, b in zip(js, js[1:]):
            assert _seg_dist((a[0], a[1]), (b[0], b[1]), (1.0, 0.0)) >= 0.3 - 1e-9
        assert r["end_error"] <= 0.02
        for got in _chain_bone_lengths(js):
            assert abs(got - 1.0) < 1e-9


def test_avoid_target_inside_circle_infeasible():
    r = ikchain.solve_chain((0.0, 0.0), (1.0, 0.0), [1.0, 1.0, 1.0],
                            avoid=((1.0, 0.0), 0.3))
    assert r["infeasible"] is True
    assert r["joints"] is None
    assert r["reason"]


def test_avoid_base_inside_circle_infeasible():
    r = ikchain.solve_chain((0.0, 0.0), (2.0, 0.0), [1.0, 1.0, 1.0],
                            avoid=((0.0, 0.0), 0.2))
    assert r["infeasible"] is True
    assert r["reason"]


def _seg_dist(p, q, c):
    vx, vy = q[0] - p[0], q[1] - p[1]
    l2 = vx * vx + vy * vy
    if l2 == 0:
        return math.hypot(p[0] - c[0], p[1] - c[1])
    t = max(0.0, min(1.0,
                     ((c[0] - p[0]) * vx + (c[1] - p[1]) * vy) / l2))
    return math.hypot(p[0] + vx * t - c[0], p[1] + vy * t - c[1])


# ---------------------------------------------------------------------------
# Vec2 与确定性
# ---------------------------------------------------------------------------

def test_vec2_operations():
    a = ikchain.Vec2(3.0, 4.0)
    assert a.length() == 5.0
    assert a.angle() == math.atan2(4.0, 3.0)
    b = ikchain.Vec2(1.0, 2.0)
    assert (a + b).tuple() == (4.0, 6.0)
    assert (a - b).tuple() == (2.0, 2.0)
    assert (a * 2).tuple() == (6.0, 8.0)
    assert (2 * a).tuple() == (6.0, 8.0)
    assert ikchain.Vec2(0, 0).normalize().tuple() == (0.0, 0.0)


def test_negative_bone_length_rejected():
    with __import__("pytest").raises(ValueError):
        ikchain.reach(-1.0, 1.0)
    r = ikchain.solve_chain((0, 0), (1, 0), [1.0, -1.0])
    assert r["infeasible"] is True


# ---------------------------------------------------------------------------
# CLI 验收
# ---------------------------------------------------------------------------

def test_cli_end_to_end(tmp_path):
    out = tmp_path / "solution.json"
    proc = subprocess.run(
        [sys.executable, "-m", "ikchain.cli",
         os.path.join(ROOT, "data", "targets.json"), "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "case 0" in proc.stdout and "case 1" in proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["status"] == "ok"
    assert abs(data[0]["error"]) < 1e-9
    assert data[1]["status"] == "max_reach"
    assert abs(data[1]["error"] - 43.0) < 1e-9

    # 再跑一次，输出必须逐字节一致（确定性输出）
    out2 = tmp_path / "solution2.json"
    subprocess.run(
        [sys.executable, "-m", "ikchain.cli",
         os.path.join(ROOT, "data", "targets.json"), "--out", str(out2)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert out.read_bytes() == out2.read_bytes()
