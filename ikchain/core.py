# -*- coding: utf-8 -*-
"""二维 IK 求解器（仅依赖 Python 标准库）。

- :func:`solve_two_bone`：两骨解析解，支持中间关节内角限位、超距最大伸展、
  目标与基座重合 / 零长度骨 / 恰好可达等退化场景。
- :func:`solve_chain`：三骨以上链的 CCD 迭代求解，支持逐关节角度限位、
  冻结关节报告、禁入圆规避，以及不可行场景的显式标记。

所有位置对外统一以 ``(x, y)`` 元组返回，角度均为弧度。
"""

import math
from dataclasses import dataclass

EPS = 1e-12
# 限位判定容差：落在边界该容差内即视为被限位冻结
LIM_EPS = 1e-7
# 禁入圆外侧预留安全余量，避免浮点误差把点放回圆内
AVOID_MARGIN = 1.001


# ---------------------------------------------------------------------------
# 基础类型与小工具
# ---------------------------------------------------------------------------

@dataclass
class Vec2:
    """二维向量。"""

    x: float
    y: float

    def __add__(self, other):
        o = _as_vec(other)
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, other):
        o = _as_vec(other)
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k):
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def length(self):
        return math.hypot(self.x, self.y)

    def normalize(self):
        d = self.length()
        if d <= EPS:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / d, self.y / d)

    def angle(self):
        """该向量的方向角，零向量约定为 0。"""
        if abs(self.x) <= EPS and abs(self.y) <= EPS:
            return 0.0
        return math.atan2(self.y, self.x)

    def tuple(self):
        return (self.x, self.y)


def _as_vec(p):
    """把 Vec2 / (x, y) 序列统一成 :class:`Vec2`。"""
    if isinstance(p, Vec2):
        return Vec2(p.x, p.y)
    return Vec2(float(p[0]), float(p[1]))


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _wrap_pi(a):
    """把角度规整到 (-pi, pi]。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _rotate_about(p, pivot, ang):
    """把点 p 绕 pivot 旋转 ang 弧度。"""
    c, s = math.cos(ang), math.sin(ang)
    d = p - pivot
    return Vec2(
        pivot.x + c * d.x - s * d.y,
        pivot.y + s * d.x + c * d.y,
    )


# ---------------------------------------------------------------------------
# 保留签名的三个基础函数
# ---------------------------------------------------------------------------

def reach(l1, l2):
    """两骨完全伸展时的最大可达距离。"""
    if l1 < 0 or l2 < 0:
        raise ValueError("骨长不能为负")
    return float(l1) + float(l2)


def joint_limit(angle, lo, hi):
    """把角度钳制到限位区间 ``[lo, hi]``。

    ``lo`` / ``hi`` 为 ``None`` 时表示该侧无界；返回被钳制后的角度，
    未越界时原样返回。
    """
    angle = float(angle)
    if lo is not None and angle < lo:
        return float(lo)
    if hi is not None and angle > hi:
        return float(hi)
    return angle


# ---------------------------------------------------------------------------
# 两骨解析解
# ---------------------------------------------------------------------------

def _two_bone_infeasible(reason, b, t, l1, l2, rmax, max_pos, bend):
    return {
        "ok": False,
        "status": "infeasible",
        "reason": reason,
        "limited": True,
        "base": b.tuple(),
        "target": t.tuple(),
        "elbow": None,
        "end": None,
        "error": _dist(b, t),
        "angles": None,
        "lengths": [l1, l2],
        "max_reach": rmax,
        "max_extend_position": max_pos.tuple(),
        "bend": bend,
    }


def _elbow_geometry(b, direction, l1, l2, r, bend):
    """给定基座、朝向、两骨长与末端距离 r，计算肘关节与末端位置。"""
    end = b + direction * r
    if l1 <= EPS:
        return b, end
    if r <= EPS:
        # 末端落在基座上：肘关节垂直于参考方向放置，朝向由 bend 决定
        side = Vec2(direction.y, -direction.x)
        return b + side * (bend * l1), end
    a = (r * r + l1 * l1 - l2 * l2) / (2.0 * r)
    a = _clamp(a, -l1, l1)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    foot = b + direction * a
    side = Vec2(direction.y, -direction.x)  # 顺时针法向
    return foot + side * (bend * h), end


def solve_two_bone(base, target, l1, l2, *, bend=1.0, limit=None, tol=1e-9):
    """两骨 IK 解析解。

    参数
    ----
    base, target:
        基座与末端目标，接受 :class:`Vec2` 或 ``(x, y)``。
    l1, l2:
        两根骨的长度。
    bend:
        肘关节弯折方向，``+1`` 为顺时针侧（默认），``-1`` 为镜像侧。
    limit:
        中间关节内角（弧度，区间 ``[0, pi]``）的限位 ``(lo, hi)``，
        元素可为 ``None`` 表示该侧无界；``None`` 表示不限位。
    tol:
        判定 ``ok`` 的末端误差容差。

    返回
    -------
    dict，``status`` 取值：

    - ``ok``：末端到达目标；
    - ``max_reach``：目标超出最大可达距离，按最大伸展处理；
    - ``min_reach``：目标比最近可达距离更近（含与基座重合但骨长不等），
      按最大折叠处理；
    - ``limited``：自然解越过关节限制，已退化为限制边界上的最近姿态；
    - ``infeasible``：限位在物理上不可满足，不返回任何越界姿态。
    """
    b = _as_vec(base)
    t = _as_vec(target)
    l1, l2 = float(l1), float(l2)
    bend = 1.0 if bend >= 0 else -1.0
    rmax = reach(l1, l2)

    if l1 < 0 or l2 < 0:
        return _two_bone_infeasible(
            "骨长不能为负。", b, t, l1, l2, rmax, b, bend,
        )

    # 限位规范化与物理可行性检查
    lo = hi = None
    if limit is not None:
        lo, hi = limit
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        if lo is not None and hi is not None and lo > hi:
            return _two_bone_infeasible(
                "关节限位区间为空：lo=%r > hi=%r。" % (lo, hi),
                b, t, l1, l2, rmax, b, bend,
            )
        if hi is not None and hi < 0.0:
            return _two_bone_infeasible(
                "关节内角不可能小于 0，限位 hi=%r 不可满足。" % hi,
                b, t, l1, l2, rmax, b, bend,
            )
        if lo is not None and lo > math.pi:
            return _two_bone_infeasible(
                "关节内角不可能大于 pi，限位 lo=%r 不可满足。" % lo,
                b, t, l1, l2, rmax, b, bend,
            )

    d = _dist(b, t)
    if d > EPS:
        direction = (t - b).normalize()
    else:
        # 目标与基座重合：取 +x 作为确定性参考方向
        direction = Vec2(1.0, 0.0)

    rmin = abs(l1 - l2)
    max_pos = b + direction * rmax

    # 第一步：忽略限位，求距离约束下的自然姿态
    if d > rmax + EPS:
        r_nat, status = rmax, "max_reach"
    elif d < rmin - EPS:
        r_nat, status = rmin, "min_reach"
    else:
        r_nat, status = _clamp(d, rmin, rmax), "ok"

    # 自然关节内角
    if l1 <= EPS or l2 <= EPS:
        gamma_nat = 0.0
    else:
        cos_g = (l1 * l1 + l2 * l2 - r_nat * r_nat) / (2.0 * l1 * l2)
        gamma_nat = math.acos(_clamp(cos_g, -1.0, 1.0))

    # 第二步：施加关节限位
    limited = False
    gamma = gamma_nat
    if lo is not None and gamma < lo:
        gamma, limited = float(lo), True
    if hi is not None and gamma > hi:
        gamma, limited = float(hi), True

    if limited and (l1 <= EPS or l2 <= EPS):
        return _two_bone_infeasible(
            "存在零长度骨，无法表达非零关节角 %r。" % gamma,
            b, t, l1, l2, rmax, max_pos, bend,
        )

    if limited:
        # 限制边界上的最近姿态：末端沿目标方向停在该弯折对应的距离
        r = math.sqrt(
            max(0.0, l1 * l1 + l2 * l2 - 2.0 * l1 * l2 * math.cos(gamma))
        )
        status = "limited"
    else:
        r = r_nat

    elbow, end = _elbow_geometry(b, direction, l1, l2, r, bend)
    error = _dist(end, t)

    theta1 = (elbow - b).angle()
    theta2 = (end - elbow).angle()

    reason = {
        "ok": "目标在可达范围内，解析解满足全部约束。",
        "max_reach": (
            "目标距离 %.6f 超出最大可达距离 %.6f，按最大伸展处理，"
            "两根骨长保持不变，残余误差 %.6f。" % (d, rmax, error)
        ),
        "min_reach": (
            "目标距离 %.6f 小于最近可达距离 %.6f，按最大折叠处理，"
            "两根骨长保持不变，残余误差 %.6f。" % (d, rmin, error)
        ),
        "limited": (
            "自然解的关节内角 %.6f 越过限位 [%s, %s]，"
            "已退化为限制边界上的关节角 %.6f，残余误差 %.6f。"
            % (
                gamma_nat,
                "-inf" if lo is None else "%.6f" % lo,
                "+inf" if hi is None else "%.6f" % hi,
                gamma,
                error,
            )
        ),
    }[status]

    return {
        "ok": error <= tol,
        "status": status,
        "reason": reason,
        "limited": limited,
        "base": b.tuple(),
        "target": t.tuple(),
        "elbow": elbow.tuple(),
        "end": end.tuple(),
        "error": error,
        "angles": {
            "base": theta1,
            "joint": gamma,
            "bone2": theta2,
        },
        "lengths": [l1, l2],
        "max_reach": rmax,
        "max_extend_position": max_pos.tuple(),
        "bend": bend,
    }


# ---------------------------------------------------------------------------
# 多骨链 CCD 迭代求解
# ---------------------------------------------------------------------------

def _bone_angles(P, lengths):
    """每节骨的绝对方向角；零长度骨沿用前一节方向。"""
    angles = []
    prev = 0.0
    for i, l in enumerate(lengths):
        if l > EPS:
            prev = (P[i + 1] - P[i]).angle()
        angles.append(prev)
    return angles


def _seg_circle_dist(p, q, c):
    """圆心 c 到线段 pq 的最短距离。"""
    v = q - p
    l2 = v.x * v.x + v.y * v.y
    if l2 <= EPS:
        return _dist(p, c)
    t = _clamp(((c.x - p.x) * v.x + (c.y - p.y) * v.y) / l2, 0.0, 1.0)
    closest = Vec2(p.x + v.x * t, p.y + v.y * t)
    return _dist(closest, c)


def _push_out(P, j, C, R, base_dir):
    """把第 j 个关节推到禁入圆边界外侧；圆心处按固定法向推出。"""
    d = _dist(P[j], C)
    if d <= EPS:
        perp = Vec2(-base_dir.y, base_dir.x)
        P[j] = C + perp * (R * AVOID_MARGIN)
    else:
        P[j] = C + (P[j] - C) * (R * AVOID_MARGIN / d)


def _avoid_nudge(P, lengths, C, R, base_dir):
    """把侵入禁入圆的关节 / 线段推出圆外，返回是否发生过推动。"""
    n = len(lengths)
    moved = False

    # 关节点在圆内：沿径向投到圆外
    for j in range(1, n):
        if _dist(P[j], C) < R:
            _push_out(P, j, C, R, base_dir)
            moved = True

    # 端点在圆外但线段穿圆：推动离穿透点最近的可动内部关节
    for i in range(n):
        if _seg_circle_dist(P[i], P[i + 1], C) >= R - 1e-12:
            continue
        v = P[i + 1] - P[i]
        l2 = v.x * v.x + v.y * v.y
        if l2 <= EPS:
            t = 0.0
        else:
            t = _clamp(
                ((C.x - P[i].x) * v.x + (C.y - P[i].y) * v.y) / l2,
                0.0,
                1.0,
            )
        closest = Vec2(P[i].x + v.x * t, P[i].y + v.y * t)
        candidates = [j for j in (i, i + 1) if 0 < j < n]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: _dist(P[k], closest))
        if _dist(P[j], C) < R * AVOID_MARGIN:
            _push_out(P, j, C, R, base_dir)
            moved = True
    return moved


def _limits_satisfied(P, lengths, limits):
    """检查当前姿态的全部关节角是否落在限位内。"""
    angles = _bone_angles(P, lengths)
    for i, (lo, hi) in enumerate(limits):
        rel = angles[i] if i == 0 else _wrap_pi(angles[i] - angles[i - 1])
        if lo is not None and rel < lo - LIM_EPS:
            return False
        if hi is not None and rel > hi + LIM_EPS:
            return False
    return True


def _circle_clear(P, lengths, C, R):
    return all(
        _seg_circle_dist(P[i], P[i + 1], C) >= R - 1e-9
        for i in range(len(lengths))
    )


def _fabrik_forward(P, lengths, target):
    """前向 reaching：把末端钉到目标，逐关节向回拉，骨长保持不变。"""
    n = len(lengths)
    P[n] = Vec2(target.x, target.y)
    for i in range(n - 1, -1, -1):
        d = _dist(P[i + 1], P[i])
        if d <= EPS:
            continue
        u = (P[i] - P[i + 1]).normalize()
        P[i] = P[i + 1] + u * lengths[i]


def _fabrik_backward(P, lengths, base):
    """后向 reaching：把基座钉回，逐关节向外推，骨长保持不变。"""
    n = len(lengths)
    P[0] = Vec2(base.x, base.y)
    for i in range(n):
        d = _dist(P[i + 1], P[i])
        if d <= EPS:
            P[i + 1] = Vec2(P[i].x, P[i].y)
            continue
        u = (P[i + 1] - P[i]).normalize()
        P[i + 1] = P[i] + u * lengths[i]


def _apply_limits(P, lengths, limits):
    """逐关节把相对弯折角钳制到限位内，返回被钳制的关节下标集合。"""
    n = len(lengths)
    clamped = set()
    for i in range(n):
        angles = _bone_angles(P, lengths)
        lo, hi = limits[i]
        if lo is None and hi is None:
            continue
        ref = 0.0 if i == 0 else angles[i - 1]
        rel = angles[i] if i == 0 else _wrap_pi(angles[i] - ref)
        rel_new = joint_limit(rel, lo, hi)
        if abs(rel_new - rel) > LIM_EPS:
            clamped.add(i)
        delta = _wrap_pi((ref + rel_new) - angles[i])
        if abs(delta) > EPS:
            for k in range(i + 1, n + 1):
                P[k] = _rotate_about(P[k], P[i], delta)
    return clamped


def _chain_infeasible(reason, end_error, iterations=0, frozen=None):
    return {
        "infeasible": True,
        "reason": reason,
        "end_error": float(end_error),
        "residual": float(end_error),
        "iterations": iterations,
        "frozen": sorted(frozen or []),
        "joints": None,
        "converged": False,
    }


def solve_chain(base, target, lengths, limits=None, tol=0.01, avoid=None, max_iter=64):
    """多骨链 FABRIK 求解：返回 dict(joints, end_error, iterations, frozen)。

    参数
    ----
    base, target:
        基座与末端目标。
    lengths:
        骨长序列，长度即骨数（支持三骨以上，允许零长度骨）。
    limits:
        每节骨的方向限位 ``(lo, hi)``（弧度），元素可为 ``None``；
        第 0 节相对世界坐标系，其余节相对前一节的弯折角（规整到 (-pi, pi]）。
        传 ``None`` 表示该节不受限。
    tol:
        末端误差收敛容差。
    avoid:
        禁入圆 ``((cx, cy), radius)``；链的任何一段都不得进入该圆。
    max_iter:
        最大迭代轮数。

    返回
    -------
    可行 / 降级时返回::

        dict(infeasible=False, joints=[(x,y),...], end_error=..., residual=...,
             iterations=..., frozen=[...], converged=..., reason=...)

    不可行时返回 ``dict(infeasible=True, reason=...)``，不返回侵入禁区或
    违反物理约束的姿态。``frozen`` 为最终姿态上被限位冻结在边界的关节下标，
    按下标（离基座的优先级）有序列出。
    """
    B = _as_vec(base)
    T = _as_vec(target)
    ls = [float(x) for x in lengths]
    n = len(ls)
    if n < 1:
        return _chain_infeasible("骨长序列为空，无法构成链。", _dist(B, T))
    if any(l < 0 for l in ls):
        return _chain_infeasible("骨长不能为负。", _dist(B, T))

    total = sum(ls)
    if limits is None:
        norm_limits = [(None, None)] * n
    else:
        if len(limits) != n:
            return _chain_infeasible(
                "限位数量 %d 与骨数 %d 不一致。" % (len(limits), n), _dist(B, T)
            )
        norm_limits = []
        for i, lim in enumerate(limits):
            if lim is None:
                norm_limits.append((None, None))
                continue
            lo, hi = lim
            lo = None if lo is None else float(lo)
            hi = None if hi is None else float(hi)
            if lo is not None and hi is not None and lo > hi:
                return _chain_infeasible(
                    "关节 %d 的限位区间为空：lo=%r > hi=%r。" % (i, lo, hi),
                    _dist(B, T),
                )
            norm_limits.append((lo, hi))

    C = R = None
    if avoid is not None:
        (cc, R) = avoid
        C = _as_vec(cc)
        R = float(R)
        if _dist(B, C) < R - 1e-9:
            return _chain_infeasible(
                "基座位于禁入圆内，链无法在不进入禁区的情况下存在。",
                _dist(B, T),
            )
        if _dist(T, C) < R - 1e-9:
            return _chain_infeasible(
                "末端目标位于禁入圆内，可达与规避无法同时满足。",
                _dist(B, T),
            )

    d = _dist(B, T)
    if d > total + 1e-9:
        return _chain_infeasible(
            "目标距离 %.6f 超出链最大伸展 %.6f，无可达解。" % (d, total),
            d - total,
        )

    # 确定性初始姿态：按方向角逐节前向构造，骨长天生精确。
    # 目标不在最大伸展点上时给内部关节一个固定弯折，避免完全伸直的退化构型；
    # 零长度骨不参与弯折，直接叠在上一关节上。
    base_dir = (T - B).normalize() if d > EPS else Vec2(1.0, 0.0)
    base_angle = base_dir.angle()
    need_bend = d < total - 1e-9
    P = [Vec2(B.x, B.y)]
    for j, l in enumerate(ls):
        if l <= EPS:
            P.append(Vec2(P[-1].x, P[-1].y))
            continue
        ang = base_angle + (0.35 * j if need_bend else 0.0)
        P.append(P[-1] + Vec2(math.cos(ang), math.sin(ang)) * l)

    iterations = 0
    clamped_last = set()
    converged = False

    for it in range(1, max_iter + 1):
        iterations = it
        # FABRIK 双向 reaching（每一步都严格保持骨长）
        _fabrik_forward(P, ls, T)
        _fabrik_backward(P, ls, B)
        # 关节限位钳制
        clamped_last = _apply_limits(P, ls, norm_limits)

        # 禁入圆规避
        if C is not None:
            if _avoid_nudge(P, ls, C, R, base_dir):
                # 推动会破坏骨长，做一次后向 reaching 恢复
                _fabrik_backward(P, ls, B)

        err = _dist(P[-1], T)
        if (
            err <= tol
            and _limits_satisfied(P, ls, norm_limits)
            and (C is None or _circle_clear(P, ls, C, R))
        ):
            converged = True
            break

    # 最后再钳一次限位（规避恢复可能轻微越界）
    if not _limits_satisfied(P, ls, norm_limits):
        clamped_last |= _apply_limits(P, ls, norm_limits)

    err = _dist(P[-1], T)

    if C is not None and not _circle_clear(P, ls, C, R):
        return _chain_infeasible(
            "迭代 %d 轮后链仍侵入禁入圆，可达与规避无法同时满足。" % iterations,
            err,
            iterations,
            clamped_last,
        )

    # 冻结关节：迭代中被钳制，或最终姿态贴在限位边界上；按下标有序报告
    angles = _bone_angles(P, ls)
    frozen = set(clamped_last)
    for i, (lo, hi) in enumerate(norm_limits):
        rel = angles[i] if i == 0 else _wrap_pi(angles[i] - angles[i - 1])
        if lo is not None and abs(rel - lo) <= 1e-6:
            frozen.add(i)
        if hi is not None and abs(rel - hi) <= 1e-6:
            frozen.add(i)

    if converged:
        reason = "迭代 %d 轮后末端误差 %.6f 收敛到容差 %.6f 以内。" % (
            iterations,
            err,
            tol,
        )
    else:
        reason = (
            "迭代 %d 轮后未达到容差 %.6f（关节限位 / 禁入圆约束下降级），"
            "实际末端误差 %.6f，冻结关节 %s。"
            % (iterations, tol, err, sorted(frozen) or "无")
        )

    return {
        "infeasible": False,
        "joints": [(p.x, p.y) for p in P],
        "end_error": err,
        "residual": err,
        "iterations": iterations,
        "frozen": sorted(frozen),
        "converged": converged,
        "reason": reason,
    }

