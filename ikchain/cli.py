# -*- coding: utf-8 -*-
"""两骨 IK 命令行入口。

用法::

    python -m ikchain.cli data/targets.json --out solution.json
"""

import argparse
import json
import math

import ikchain as m


def _vec2(v):
    return m.Vec2(float(v[0]), float(v[1]))


def _point(v):
    if v is None:
        return None
    return [round(v.x, 12), round(v.y, 12)]


def _serialize(sol):
    return {
        "ok": sol["ok"],
        "status": sol["status"],
        "reason": sol["reason"],
        "limited": sol["limited"],
        "base": _point(sol["base"]),
        "target": _point(sol["target"]),
        "elbow": _point(sol["elbow"]),
        "end": _point(sol["end"]),
        "error": None if sol["error"] is None else round(sol["error"], 12),
        "angles": {
            k: (None if a is None else round(a, 12))
            for k, a in sol["angles"].items()
        },
        "lengths": list(sol["lengths"]),
        "max_reach": sol["max_reach"],
        "max_extend_position": _point(sol["max_extend_position"]),
        "bend": sol["bend"],
    }


def solve_one(item):
    base = _vec2(item["base"]) if "base" in item else m.Vec2(0.0, 0.0)
    target = _vec2(item["target"])
    l1 = float(item["l1"])
    l2 = float(item["l2"])
    kwargs = {}
    if "bend" in item:
        kwargs["bend"] = float(item["bend"])
    if "limit" in item:
        kwargs["limit"] = (float(item["limit"][0]), float(item["limit"][1]))
    return m.solve_two_bone(base, target, l1, l2, **kwargs)


def main():
    ap = argparse.ArgumentParser(description="两骨 IK 求解")
    ap.add_argument("targets", help="目标 JSON 文件路径")
    ap.add_argument("--out", required=True, help="解算结果输出 JSON 路径")
    a = ap.parse_args()

    with open(a.targets, encoding="utf-8") as f:
        items = json.load(f)

    solutions = []
    print("%-3s %-9s %12s %14s %14s %-22s"
          % ("#", "status", "error", "joint(rad)", "joint(deg)", "max_extend"))
    for i, item in enumerate(items):
        # 解算两次，确认结果一致
        sol_a = solve_one(item)
        sol_b = solve_one(item)
        assert _serialize(sol_a) == _serialize(sol_b), "两次解算结果不一致"
        solutions.append(_serialize(sol_a))

        joint = sol_a["angles"]["joint"]
        ext = sol_a["max_extend_position"]
        print("%-3d %-9s %12s %14s %14s (%.6f, %.6f)"
              % (
                  i,
                  sol_a["status"],
                  "n/a" if sol_a["error"] is None
                  else "%.6e" % sol_a["error"],
                  "n/a" if joint is None else "%.6f" % joint,
                  "n/a" if joint is None else "%.4f" % math.degrees(joint),
                  ext.x, ext.y,
              ))

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(solutions, f, ensure_ascii=False, indent=2)

    print("\n已写入 %s（共 %d 个目标）" % (a.out, len(solutions)))


if __name__ == "__main__":
    main()
