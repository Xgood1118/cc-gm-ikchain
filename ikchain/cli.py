# -*- coding: utf-8 -*-
"""ikchain 命令行入口：读取 targets.json，逐条求解并写出 solution.json。"""

import argparse
import json
import os

import ikchain as m


def _round_obj(obj, ndigits=9):
    """把结果里的浮点值统一截断到 ndigits 位，保证两次输出逐字节一致。"""
    if isinstance(obj, dict):
        return {k: _round_obj(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_obj(v, ndigits) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj


def solve_record(rec):
    base = m.Vec2(float(rec.get("base", (0.0, 0.0))[0]),
                  float(rec.get("base", (0.0, 0.0))[1])) if "base" in rec \
        else m.Vec2(0.0, 0.0)
    target = m.Vec2(float(rec["target"][0]), float(rec["target"][1]))
    l1 = float(rec["l1"])
    l2 = float(rec["l2"])
    limit = rec.get("limit")
    bend = rec.get("bend", 1.0)
    return m.solve_two_bone(base, target, l1, l2, bend=bend, limit=limit)


def main():
    ap = argparse.ArgumentParser(description="两骨 IK 求解")
    ap.add_argument("input", help="targets.json 路径")
    ap.add_argument("--out", required=True, help="结果输出 JSON 路径")
    ap.add_argument("--digits", type=int, default=9, help="浮点保留位数")
    a = ap.parse_args()

    with open(a.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = [_round_obj(solve_record(rec), a.digits) for rec in records]

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")

    for i, r in enumerate(results):
        print(
            "case %d: status=%s ok=%s error=%.9f elbow=%s end=%s"
            % (i, r["status"], r["ok"], r["error"], r["elbow"], r["end"])
        )
    print("已写出 %d 条结果到 %s" % (len(results), a.out))


if __name__ == "__main__":
    main()
