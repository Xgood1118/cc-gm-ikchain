# 两骨 IK 求解

一个能在关节限制下解两骨 IK、并在无解时给出最大可达位置的求解器

## 保留签名

- `Vec2`
- `solve_two_bone(base, target, l1, l2)`
- `joint_limit(angle, lo, hi)`
- `reach(l1, l2)`

## 自测

```
python -m pytest tests -q
python -m ikchain.cli data/targets.json --out solution.json
```
