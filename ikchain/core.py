# -*- coding: utf-8 -*-
"""两骨 IK 求解（仅使用 Python 标准库）。

角度约定
--------
- ``base`` 角（θ1）：第一根骨相对 +x 轴的绝对方向，弧度。
- ``joint`` 角（β）：中间关节（肘）的**内角**，即两根骨之间的夹角，
  取值区间 [0, π]：π 表示完全伸直，0 表示完全折叠。
- ``bone2`` 角（θ2）：第二根骨的绝对方向，弧度。

``limit=(lo, hi)`` 约束的就是中间关节内角 β。求解结果保证：只要返回了
位姿，β 一定落在限制区间内；若限制区间本身不可行，则返回
``status="no_solution"`` 且不给出任何越界位姿。
"""

import math
from dataclasses import dataclass

#: 几何判定容差
EPS = 1e-10
#: 末端误差小于该值视为精确到达
ERR_EPS = 1e-9


@dataclass
class Vec2:
    """二维向量。"""
    x: float
    y: float

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, k):
        return Vec2(self.x * k, self.y * k)

    def length(self):
        return math.hypot(self.x, self.y)

    def normalized(self):
        d = math.hypot(self.x, self.y)
        if d <= EPS:
            return Vec2(1.0, 0.0)
        return Vec2(self.x / d, self.y / d)


__all__ = ["Vec2", "solve_two_bone", "joint_limit", "reach"]


def reach(l1, l2):
    """最大可达距离：两骨完全伸直时末端到基座的距离。"""
    return l1 + l2


def joint_limit(angle, lo, hi):
    """把角度钳制在 [lo, hi] 内（弧度，不做周期卷绕）。

    越过下界返回 ``lo``，越过上界返回 ``hi``，区间内原样返回。
    区间本身非法（``lo > hi``）时抛出 ``ValueError``——由调用方
    （``solve_two_bone``）转成明确的无解标记，而不是悄悄给出越界值。
    """
    if lo > hi:
        raise ValueError("无效关节限制区间: lo=%r > hi=%r" % (lo, hi))
    if angle < lo:
        return lo
    if angle > hi:
        return hi
    return angle


def _norm_angle(a):
    """归一化到 (-π, π]。"""
    return math.atan2(math.sin(a), math.cos(a))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _no_solution(base, target, l1, l2, bend, reason, r_max, extend_pos):
    return {
        "ok": False,
        "status": "no_solution",
        "reason": reason,
        "limited": False,
        "base": base,
        "target": target,
        "elbow": None,
        "end": None,
        "error": None,
        "angles": {"base": None, "joint": None, "bone2": None},
        "lengths": [l1, l2],
        "max_reach": r_max,
        "max_extend_position": extend_pos,
        "bend": 1.0 if bend >= 0 else -1.0,
    }


def solve_two_bone(base, target, l1, l2, bend=1.0, limit=None):
    """两骨 IK 解析求解。

    参数
    ----
    base, target: Vec2
        基座与目标位置。
    l1, l2: float
        两根骨的长度，必须非负。
    bend: float
        肘的弯折方向：>=0 逆时针（肘在基座→目标连线左侧），<0 镜像。
    limit: None 或 (lo, hi)
        中间关节内角 β 的弧度限制区间。

    返回
    ----
    dict，关键字段：

    - ``ok``: 末端误差收敛到 ``ERR_EPS`` 内且所有约束满足时为 True。
    - ``status``: ``ok`` / ``max_reach``（超范围最大伸展）/
      ``min_reach``（落入最小半径内，完全折叠）/ ``limited``
      （关节角越界，已退化到限制边界）/ ``no_solution``（限制不可行）。
    - ``elbow`` / ``end``: 肘与末端位置；无解时为 None。
    - ``error``: 末端到目标的距离；无解时为 None。
    - ``angles``: ``{"base", "joint", "bone2"}`` 三个角（弧度）。
    - ``max_extend_position``: 朝目标方向完全伸直时末端的参考位置。
    """
    sgn = 1.0 if bend >= 0 else -1.0
    r_max = l1 + l2

    delta = target - base
    d = delta.length()
    phi = math.atan2(delta.y, delta.x) if d > EPS else 0.0
    ux, uy = math.cos(phi), math.sin(phi)
    extend_pos = Vec2(base.x + r_max * ux, base.y + r_max * uy)

    # ---- 输入合法性：负骨长无解 ----
    if l1 < 0.0 or l2 < 0.0:
        return _no_solution(
            base, target, l1, l2, bend,
            "骨长不能为负（l1=%r, l2=%r），无可行构型。" % (l1, l2),
            r_max, extend_pos,
        )

    # ---- 关节限制区间可行性 ----
    if limit is None:
        lo_eff, hi_eff = 0.0, math.pi
        raw_lo, raw_hi = 0.0, math.pi
    else:
        raw_lo, raw_hi = limit
        if raw_lo > raw_hi:
            return _no_solution(
                base, target, l1, l2, bend,
                "关节限制区间非法: lo=%.6f > hi=%.6f，不存在满足限制的构型。"
                % (raw_lo, raw_hi),
                r_max, extend_pos,
            )
        lo_eff, hi_eff = max(0.0, raw_lo), min(math.pi, raw_hi)
        if lo_eff > hi_eff + EPS:
            return _no_solution(
                base, target, l1, l2, bend,
                "关节限制区间 [%.6f, %.6f] 与关节物理范围 [0, π] 无交集，无解。"
                % (raw_lo, raw_hi),
                r_max, extend_pos,
            )

    # ---- 零长度骨：关节角无定义，退化为单段直骨 ----
    if l1 == 0.0 or l2 == 0.0:
        length = r_max
        end = Vec2(base.x + length * ux, base.y + length * uy)
        if l1 == 0.0:
            elbow = Vec2(base.x, base.y)
        else:  # l2 == 0，肘即末端
            elbow = Vec2(end.x, end.y)
        error = math.hypot(end.x - target.x, end.y - target.y)
        if d > r_max + EPS:
            status, reason = "max_reach", (
                "目标超出最大可达距离 %.6f，按最大伸展处理；"
                "存在零长度骨，关节角无定义。" % r_max)
        elif error <= ERR_EPS:
            status, reason = "ok", "精确到达；存在零长度骨，关节角无定义。"
        else:
            status, reason = "min_reach", (
                "零长度骨使可达距离退化为单点 %.6f，目标在其内侧，"
                "按折叠处理。" % length)
        return {
            "ok": error <= ERR_EPS,
            "status": status,
            "reason": reason,
            "limited": False,
            "base": base,
            "target": target,
            "elbow": elbow,
            "end": end,
            "error": error,
            "angles": {"base": None, "joint": None, "bone2": None},
            "lengths": [l1, l2],
            "max_reach": r_max,
            "max_extend_position": extend_pos,
            "bend": sgn,
        }

    r_min = abs(l1 - l2)

    # ---- 无约束的关节内角 β* ----
    if d >= r_max - EPS:
        beta_star = math.pi          # 恰好可达或超范围：完全伸直
    elif d <= r_min + EPS:
        beta_star = 0.0              # 落入内圈：完全折叠
    else:
        cos_beta = (l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2)
        beta_star = math.acos(_clamp(cos_beta, -1.0, 1.0))

    # ---- 关节限制：越界则退化到边界 ----
    beta = joint_limit(beta_star, lo_eff, hi_eff)
    limited = beta_star < lo_eff - EPS or beta_star > hi_eff + EPS
    if limited:
        boundary = lo_eff if beta_star < lo_eff else hi_eff
        reason = (
            "无约束关节角 β=%.6f rad（%.4f°）越过限制 [%.6f, %.6f]，"
            "已退化为限制边界 β=%.6f rad（%.4f°），末端为限制下的最近可达位置。"
            % (beta_star, math.degrees(beta_star), raw_lo, raw_hi,
               boundary, math.degrees(boundary))
        )
    else:
        reason = ""

    # 固定 β 后末端相对基座的实际可达距离
    d_eff = math.sqrt(max(
        0.0,
        l1 * l1 + l2 * l2 - 2.0 * l1 * l2 * math.cos(beta),
    ))

    # ---- 按 β 与朝向摆位 ----
    if d_eff <= EPS:
        # 等长骨完全折叠：末端回到基座，第一骨方向取 +x 保证结果确定
        theta1 = 0.0
        theta2 = math.pi
        elbow = Vec2(base.x + l1, base.y)
        end = Vec2(base.x, base.y)
    else:
        cos_alpha = (l1 * l1 + d_eff * d_eff - l2 * l2) / (2.0 * l1 * d_eff)
        alpha = math.acos(_clamp(cos_alpha, -1.0, 1.0))
        theta1 = phi - sgn * alpha
        theta2 = theta1 + sgn * (math.pi - beta)
        elbow = Vec2(
            base.x + l1 * math.cos(theta1),
            base.y + l1 * math.sin(theta1),
        )
        end = Vec2(
            elbow.x + l2 * math.cos(theta2),
            elbow.y + l2 * math.sin(theta2),
        )

    error = math.hypot(end.x - target.x, end.y - target.y)

    if limited:
        status = "limited"
    elif d > r_max + EPS:
        status = "max_reach"
        reason = (
            "目标距离 %.6f 超出最大可达距离 %.6f，按最大伸展处理，"
            "两根骨长保持不变，残余误差 %.6f。" % (d, r_max, error)
        )
    elif d < r_min - EPS:
        status = "min_reach"
        reason = (
            "目标距离 %.6f 小于最小可达距离 %.6f，按完全折叠处理，"
            "残余误差 %.6f。" % (d, r_min, error)
        )
    else:
        status = "ok"
        reason = "目标在可达范围内，解析解满足全部约束。"

    return {
        "ok": error <= ERR_EPS,
        "status": status,
        "reason": reason,
        "limited": limited,
        "base": base,
        "target": target,
        "elbow": elbow,
        "end": end,
        "error": error,
        "angles": {
            "base": _norm_angle(theta1),
            "joint": beta,
            "bone2": _norm_angle(theta2),
        },
        "lengths": [l1, l2],
        "max_reach": r_max,
        "max_extend_position": extend_pos,
        "bend": sgn,
    }
