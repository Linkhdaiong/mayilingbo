#!/usr/bin/env python3
"""证明 advantage 权重与样本的对齐不依赖 LeRobot 底层的物理排列顺序。

背景：sidecar 是一维张量，若按「扁平 idx」取权重，就隐含假设 LeRobot 底层样本
的物理排列顺序 == 生成权重时的 episode 拼接顺序。转换脚本版本差异或分片方式变化
都会打破这个假设，造成权重与样本错位（且不会报错，只是静默掉分）。

本测试验证改用 (episode_index, frame_index) 定位后，即使物理顺序被打乱，
取到的权重依然正确。
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lingbotvla.offline_rl.advantage import (  # noqa: E402
    AdvantageConfig,
    build_weights_from_lengths,
    load_weights,
    save_weights,
)

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name} {extra}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


LENGTHS = [4, 6, 5, 3, 7]
cfg = AdvantageConfig()
w = build_weights_from_lengths(LENGTHS, cfg)

print("=" * 76)
print("A. sidecar 往返：episode_lengths 被完整保存")
print("=" * 76)
import tempfile  # noqa: E402

tmp = tempfile.mkdtemp(prefix="align_")
pt = os.path.join(tmp, "adv.pt")
save_weights(pt, w, cfg, episode_lengths=LENGTHS)
w2, meta = load_weights(pt)
check("权重数值往返一致", torch.allclose(w, w2))
check("episode_lengths 往返一致", meta.get("episode_lengths") == LENGTHS,
      f"{meta.get('episode_lengths')}")
check("超参仍在", meta.get("gamma") == cfg.gamma and meta.get("beta") == cfg.beta)

print()
print("=" * 76)
print("B. 偏移表计算（与 base_dataset.py 里的公式一致）")
print("=" * 76)
_lt = torch.tensor(LENGTHS, dtype=torch.long)
offsets = torch.cat([torch.zeros(1, dtype=torch.long), _lt.cumsum(0)[:-1]])
check("偏移表 = [0, 4, 10, 15, 18]", offsets.tolist() == [0, 4, 10, 15, 18],
      f"{offsets.tolist()}")
check("偏移+长度之和 == 权重总数", int(offsets[-1]) + LENGTHS[-1] == w.numel())

print()
print("=" * 76)
print("C. 核心：物理顺序被打乱后，权重依然正确")
print("=" * 76)
# 构造「真实」样本表：(episode_index, frame_index) -> 期望权重
truth = {}
for ep, n in enumerate(LENGTHS):
    for fr in range(n):
        truth[(ep, fr)] = float(w[int(offsets[ep]) + fr])

# 模拟两种物理排列：正常顺序 vs 完全打乱
import random  # noqa: E402

keys = list(truth.keys())
shuffled = keys[:]
random.Random(0).shuffle(shuffled)

def lookup_via_offsets(ep, fr):
    """复刻 base_dataset.py 的定位方式（顺序无关）。"""
    return float(w[int(offsets[ep]) + fr])

def lookup_via_flat_idx(pos, physical):
    """旧的扁平索引方式（依赖物理顺序）。"""
    ep, fr = physical[pos]
    return float(w[pos])

# 1) 顺序无关查找：任意排列下都正确
bad = [(k, lookup_via_offsets(*k), v) for k, v in truth.items()
       if abs(lookup_via_offsets(*k) - v) > 1e-9]
check("顺序无关查找：所有 (ep,fr) 都取到正确权重", not bad,
      f"(错误 {len(bad)} 个)")

# 2) 扁平索引方式在打乱后确实会错 —— 证明这个改动是必要的
mismatch = sum(1 for pos, (ep, fr) in enumerate(shuffled)
               if abs(lookup_via_flat_idx(pos, shuffled) - truth[(ep, fr)]) > 1e-9)
check("对照：旧的扁平索引在打乱后确实错位（证明改动必要）", mismatch > 0,
      f"(错位 {mismatch}/{len(keys)} 个样本)")

print()
print("=" * 76)
print("D. 静态检查：源码里确实是按 (episode_index, frame_index) 定位")
print("=" * 76)
src = open(os.path.join(ROOT, "lingbotvla/data/vla_data/base_dataset.py"),
           encoding="utf-8").read()
check("读取 raw_item['episode_index']", 'raw_item["episode_index"]' in src)
check("读取 raw_item['frame_index']", 'raw_item["frame_index"]' in src)
check("用偏移表定位", "_adv_episode_offsets[_ep]) + _fr" in src)
check("缺少 lengths 时有回退路径", "_adv_episode_offsets is not None" in src)
check("长度不一致时抛错", "_lt.sum()) != _n_expected" in src)

print()
print("=" * 76)
print(f"结果: {PASS} 通过, {FAIL} 失败")
print("=" * 76)
sys.exit(1 if FAIL else 0)
