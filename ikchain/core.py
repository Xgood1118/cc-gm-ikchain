# -*- coding: utf-8 -*-
"""两骨 IK 求解"""

from dataclasses import dataclass


@dataclass
class Vec2:
    """二维向量。"""
    x: float
    y: float




def solve_two_bone(base, target, l1, l2):
    """两骨求解"""
    raise NotImplementedError("solve_two_bone 尚未实现")


def joint_limit(angle, lo, hi):
    """角度限制"""
    raise NotImplementedError("joint_limit 尚未实现")


def reach(l1, l2):
    """最大可达距离"""
    raise NotImplementedError("reach 尚未实现")



def solve_chain(base, target, lengths, limits=None, tol=0.01, avoid=None, max_iter=64):
    """多骨链求解：返回 dict(joints, end_error, iterations, frozen)。

    frozen 为被限位冻结的关节下标列表；不可行时返回 dict(infeasible=True, reason=...)。
    """
    raise NotImplementedError("solve_chain 尚未实现")
