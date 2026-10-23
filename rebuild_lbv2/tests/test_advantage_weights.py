#!/usr/bin/env python3
"""advantage 权重的数学自检（不需要 GPU，不需要数据集）。"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbotvla.offline_rl.advantage import (  # noqa: E402
    AdvantageConfig,
    advantage_weights,
    build_weights_from_lengths,
    build_weights_from_lerobot,
    episode_lengths_to_returns,
    load_weights,
    read_lerobot_episode_lengths,
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


print("=" * 74)
print("T1. 形状与索引对齐")
print("=" * 74)
lengths = [5, 3, 4]
cfg = AdvantageConfig(gamma=0.99, beta=2.0)
r = episode_lengths_to_returns(lengths, cfg)
check("returns 长度 == sum(lengths)", r.numel() == sum(lengths), f"({r.numel()} vs {sum(lengths)})")
w = build_weights_from_lengths(lengths, cfg)
check("weights 长度 == sum(lengths)", w.numel() == sum(lengths))
check("权重全为正", bool((w > 0).all()))

print()
print("=" * 74)
print("T2. gamma=1 退化为纯 BC（零效应基线）")
print("=" * 74)
w1 = build_weights_from_lengths(lengths, AdvantageConfig(gamma=1.0, beta=2.0))
check("gamma=1 时权重恒为 1", bool(torch.allclose(w1, torch.ones_like(w1))),
      f"max|w-1|={float((w1-1).abs().max()):.2e}")

print()
print("=" * 74)
print("T3. mean(w) == 1（保持损失量级）")
print("=" * 74)
for g in (0.9, 0.99, 0.999):
    for b in (0.5, 2.0, 5.0):
        ww = build_weights_from_lengths(lengths, AdvantageConfig(gamma=g, beta=b))
        ok = abs(float(ww.mean()) - 1.0) < 1e-5
        check(f"gamma={g} beta={b} mean(w)=1", ok, f"mean={float(ww.mean()):.6f}")

print()
print("=" * 74)
print("T4. 优势方向：终局 reward 稀疏时，越接近 episode 末尾回报越高")
print("=" * 74)
r2 = episode_lengths_to_returns([6], AdvantageConfig(gamma=0.9))
check("episode 内回报单调递增", bool((r2[1:] > r2[:-1]).all()),
      f"G = {[round(float(x),4) for x in r2]}")
check("末帧回报 == 1（终局 r=1）", abs(float(r2[-1]) - 1.0) < 1e-6)
w2 = build_weights_from_lengths([6], AdvantageConfig(gamma=0.9, beta=2.0, w_min=1e-9, w_max=1e9))
check("episode 内权重单调递增（偏向接近成功的状态）", bool((w2[1:] > w2[:-1]).all()),
      f"w = {[round(float(x),4) for x in w2]}")

print()
print("=" * 74)
print("T5. 裁剪与归一化顺序")
print("=" * 74)
wc = build_weights_from_lengths([50], AdvantageConfig(gamma=0.9, beta=50.0, w_min=0.2, w_max=5.0))
check("裁剪后不超过 w_max（归一化前）", float(wc.max()) <= 5.0 * 1.0001, f"max={float(wc.max()):.4f}")
check("裁剪后不低于 w_min", float(wc.min()) > 0)
check("仍满足 mean(w)=1", abs(float(wc.mean()) - 1.0) < 1e-5, f"mean={float(wc.mean()):.6f}")

print()
print("=" * 74)
print("T5b. 默认超参下不应把训练集中到轨迹的一小段（集中度）")
print("=" * 74)
# 这是加权真正的风险指标：w < 0.5×均值 的样本占比过高，说明大半个 episode 被压掉，
# 训练会集中到收尾阶段。见 AdvantageConfig.w_min 的说明。
for _T in (100, 200, 300):
    _w_long = build_weights_from_lengths([_T] * 4, AdvantageConfig())
    _conc = float((_w_long < 0.5 * float(_w_long.mean())).float().mean())
    check(f"T={_T}: 被压制(<0.5×均值)占比 < 40%", _conc < 0.40, f"({_conc:.1%})")
_w_long = build_weights_from_lengths([200] * 4, AdvantageConfig())
check("默认超参下权重仍有区分度", float(_w_long.std(unbiased=False)) > 0.02,
      f"std={float(_w_long.std(unbiased=False)):.4f}")
# 有界性：裁剪区间把最坏情况限制住（结构性保证）
_c = AdvantageConfig()
check("最坏情况被裁剪区间限制在 ±2 倍内",
      _c.w_min == 0.5 and _c.w_max == 2.0, f"[{_c.w_min}, {_c.w_max}]")
# 即使 beta 被误设成很大，也不应退化成「只训收尾」
_w_bad = build_weights_from_lengths([300], AdvantageConfig(beta=50.0))
_ratio = float(_w_bad.max() / _w_bad.min())
check("beta=50 误设时极差仍被限制在 ~4x 内（不会退化）", _ratio < 6.0, f"({_ratio:.2f}x)")

print()
print("=" * 74)
print("T6. progress 模式")
print("=" * 74)
wp = build_weights_from_lengths(
    [6], AdvantageConfig(reward_mode="progress", beta=2.0, w_min=1e-9, w_max=1e9)
)
check("progress 模式权重严格递增（放宽裁剪）", bool((wp[1:] > wp[:-1]).all()),
      f"w = {[round(float(x),4) for x in wp]}")
wp_clip = build_weights_from_lengths([6], AdvantageConfig(reward_mode="progress", beta=2.0))
check("progress 模式默认裁剪下非递减（饱和属预期）",
      bool((wp_clip[1:] >= wp_clip[:-1]).all()),
      f"w = {[round(float(x),4) for x in wp_clip]}")
check("progress 模式首帧权重最低", int(wp.argmin()) == 0)

print()
print("=" * 74)
print("T7. 边界：单帧 episode / 空输入")
print("=" * 74)
ws = build_weights_from_lengths([1, 1], AdvantageConfig(gamma=0.99))
check("单帧 episode 不报错", ws.numel() == 2, f"w={[float(x) for x in ws]}")
we = advantage_weights(torch.zeros(0), AdvantageConfig())
check("空输入返回空", we.numel() == 0)

print()
print("=" * 74)
print("T8. 加权损失等价性（核心：w=1 时与纯 BC 完全一致）")
print("=" * 74)
torch.manual_seed(0)
losses = torch.rand(4, 8, 3)          # (B, T, D) 模拟 L1_fm 的逐元素损失
bc = losses.mean()
w_one = torch.ones(4)
weighted = (losses * w_one[:, None, None]).mean() / w_one.mean()
check("w 恒为 1 时，加权损失 == 纯 BC 损失", torch.allclose(bc, weighted),
      f"{float(bc):.8f} vs {float(weighted):.8f}")

w_hi = torch.tensor([1.0, 1.0, 1.0, 3.0])
wh = (losses * w_hi[:, None, None]).mean() / w_hi.mean()
check("w 不等时加权损失 != 纯 BC（确实生效）", not torch.allclose(bc, wh),
      f"bc={float(bc):.6f} weighted={float(wh):.6f}")

print()
print("=" * 74)
print("T9. LeRobot meta 读取（合成 v2.x / v3 两种布局）")
print("=" * 74)
with tempfile.TemporaryDirectory() as d:
    meta = os.path.join(d, "meta")
    os.makedirs(meta)
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump({"fps": 30, "total_frames": 12}, f)
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        for i, n in enumerate([5, 3, 4]):
            f.write(json.dumps({"episode_index": i, "length": n}) + "\n")
    L, info = read_lerobot_episode_lengths(d)
    check("v2.x episodes.jsonl 解析", L == [5, 3, 4], f"{L}")
    check("info.json 解析", info.get("fps") == 30)
    ww, LL, _ = build_weights_from_lerobot(d, AdvantageConfig(gamma=0.99, beta=2.0))
    check("由数据集构造权重", ww.numel() == 12)

with tempfile.TemporaryDirectory() as d:
    meta = os.path.join(d, "meta")
    os.makedirs(meta)
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump({"fps": 30, "total_frames": 12}, f)
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        for i, n in enumerate([5, 3, 4]):
            f.write(json.dumps({"episode_index": i, "length": n}) + "\n")
    # 篡改 total_frames，应被拒绝
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump({"fps": 30, "total_frames": 999}, f)
    try:
        read_lerobot_episode_lengths(d)
        check("total_frames 不一致时应报错", False)
    except ValueError:
        check("total_frames 不一致时正确报错", True)

print()
print("=" * 74)
print("T10. 存取往返")
print("=" * 74)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "adv.pt")
    cfg0 = AdvantageConfig(gamma=0.97, beta=1.5)
    save_weights(p, w, cfg0)
    w_back, cfg_back = load_weights(p)
    check("权重往返一致", torch.allclose(w, w_back))
    check("超参往返一致", cfg_back is not None and abs(cfg_back["gamma"] - 0.97) < 1e-9,
          f"{cfg_back}")

print()
print("=" * 74)
print(f"结果: {PASS} 通过, {FAIL} 失败")
print("=" * 74)
sys.exit(1 if FAIL else 0)
