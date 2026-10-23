#!/usr/bin/env python3
# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""为 LeRobot 训练集生成 advantage 权重 sidecar（offline RL / advantage-weighted flow matching）。

**这一步不训练、不调用环境**，只读数据集的 ``meta/`` 目录，因此不违反规则 3.1 的
环境调用轴。产出的 ``.pt`` 会被数据集按样本索引读取，训练时作为 flow-matching
损失的逐样本权重。

用法::

    # 单数据集
    python tools/build_adv_weights.py --dataset /path/to/lerobot_ds --out adv.pt \
        --gamma 0.99 --beta 2.0 --verify

    # 多数据集（推荐）：与 data.train_path 同格式，保证拼接顺序与训练完全一致
    python tools/build_adv_weights.py --train-path assets/training_data/robotwin_clean.txt \
        --out adv_weights_g099_b2.pt --verify

    # 零效应基线（w 恒为 1，等价纯 BC），用于 A/B 对照
    python tools/build_adv_weights.py --train-path ... --out adv_baseline.pt --gamma 1.0

``--verify`` 会打印权重分布并做自检（mean(w)==1、权重有区分度等）。
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbotvla.offline_rl.advantage import (  # noqa: E402
    AdvantageConfig,
    build_weights_from_lerobot,
    save_weights,
)


def parse_train_path(train_path: str) -> list[tuple[str, str]]:
    """解析训练清单，语义与 lingbotvla.data.vla_data.multi_vla_dataset.get_all_tasks 完全一致。

    格式：每行 ``<data_name><空格><repo_id>``；多个 txt 用逗号分隔。
    返回顺序即数据集拼接顺序，也是权重必须遵循的顺序。
    """
    entries: list[tuple[str, str]] = []
    for tf in train_path.split(","):
        tf = tf.strip()
        if not tf:
            continue
        if not tf.lower().endswith(".txt"):
            raise ValueError(f"train_path 项必须以 .txt 结尾：{tf}")
        with open(tf, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    raise ValueError(f"{tf}:{lineno} 期望 '<data_name> <repo_id>'，实际 {line!r}")
                entries.append((parts[0], parts[1]))
    if not entries:
        raise ValueError(f"训练清单为空：{train_path}")
    return entries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="由 LeRobot 数据集的 episode 边界推导 advantage 权重（零环境调用）"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="单个 LeRobot 数据集根目录（含 meta/）")
    src.add_argument(
        "--train-path",
        help="训练清单 txt（'<data_name> <repo_id>' 每行一条），与 data.train_path 同格式",
    )
    p.add_argument("--out", required=True, help="输出 .pt 路径")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子；1.0 = 退化为纯 BC")
    p.add_argument("--beta", type=float, default=0.25,
                   help="优势缩放（AWAC 温度倒数）；权重按优势 std 标准化，故该值尺度无关。"
                        "刻意取小，避免加权退化成只训收尾")
    p.add_argument("--w-min", type=float, default=0.5,
                   help="权重下裁剪。决定最坏情况下某个阶段被压制的上限，别调太宽")
    p.add_argument("--w-max", type=float, default=2.0, help="权重上裁剪，同上")
    p.add_argument(
        "--reward-mode",
        choices=["terminal_gamma", "progress"],
        default="terminal_gamma",
        help="terminal_gamma: gamma**(T-1-t)；progress: t/(T-1)",
    )
    p.add_argument("--no-normalize-mean", action="store_true", help="不强制 mean(w)=1")
    p.add_argument("--verify", action="store_true", help="打印分布统计并做自检")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = AdvantageConfig(
        gamma=args.gamma,
        beta=args.beta,
        w_min=args.w_min,
        w_max=args.w_max,
        reward_mode=args.reward_mode,
        normalize_mean=not args.no_normalize_mean,
    )
    cfg.validate()

    print("=" * 76)
    print("advantage 权重构造（零环境调用；reward 由 episode 边界推导）")
    print("=" * 76)
    print(f"  reward_mode   : {cfg.reward_mode}")
    print(f"  gamma={cfg.gamma}  beta={cfg.beta}  clip=[{cfg.w_min}, {cfg.w_max}]")
    print(f"  normalize_mean: {cfg.normalize_mean}")
    print()

    info: dict = {}
    if args.train_path:
        entries = parse_train_path(args.train_path)
        print(f"  训练清单: {args.train_path}  （{len(entries)} 个数据集）")
        print("  " + "-" * 72)
        chunks = []
        lengths: list[int] = []
        for name, root in entries:
            try:
                w, L, ds_info = build_weights_from_lerobot(root, cfg)
            except Exception as e:
                print(f"    ❌ {name}: {type(e).__name__}: {e}")
                print(f"       (repo_id = {root})")
                return 1
            chunks.append(w)
            lengths.extend(L)
            print(f"    {name:56s} {len(L):4d} eps {w.numel():8d} 帧")
            if not info:
                info = ds_info
        print("  " + "-" * 72)
        weights = torch.cat(chunks) if chunks else torch.zeros(0)
    else:
        print(f"  数据集: {args.dataset}")
        weights, lengths, info = build_weights_from_lerobot(args.dataset, cfg)

    n_ep = len(lengths)
    n_sample = int(weights.numel())
    print()
    print(f"  episode 数 : {n_ep}")
    print(f"  样本数     : {n_sample}")
    if n_ep:
        print(f"  帧数 min/mean/max : {min(lengths)} / {sum(lengths)/n_ep:.1f} / {max(lengths)}")
    if info.get("fps"):
        print(f"  fps        : {info['fps']}")
    print()

    if n_sample != sum(lengths):
        print(f"  ❌ 权重数 ({n_sample}) != 帧数之和 ({sum(lengths)})")
        return 1

    w = weights
    print("  权重分布:")
    print(f"    min    {w.min():.4f}")
    print(f"    p25    {w.quantile(0.25):.4f}")
    print(f"    median {w.median():.4f}")
    print(f"    p75    {w.quantile(0.75):.4f}")
    print(f"    max    {w.max():.4f}")
    print(f"    mean   {w.mean():.6f}")
    print(f"    std    {w.std(unbiased=False):.4f}")
    # ---- 集中度诊断（比裁剪饱和率重要得多）----
    # 加权真正的风险不是「饱和」，而是「把训练集中到轨迹的某一小段」。
    # 下面按 episode 位置统计权重剖面，直接暴露这个问题。
    if cfg.reward_mode == "terminal_gamma" and len(lengths) > 0:
        _scale = float(w.mean())
        suppressed = float((w < 0.5 * _scale).float().mean())
        print("    集中度诊断（关键指标）:")
        print(f"      权重 < 0.5×均值 的样本占比: {suppressed:.1%}")
        if suppressed > 0.40:
            print("      ⚠️  超过 40% 的样本被压掉一半以上 —— 加权把训练集中到了轨迹的一小段。")
            print("          对需要全程视觉鲁棒性的任务通常是负收益；请调小 --beta，")
            print("          或收窄 --w-min/--w-max 把最坏情况限制住。")
        elif suppressed > 0.20:
            print("      ⚠️  有集中趋势但尚可接受；务必与 --gamma 1.0 基线做 A/B。")
        else:
            print("      ✅ 加权温和，未出现明显集中。")
        # 逐 episode 剖面（取第一个 episode 作代表）
        _n = int(lengths[0])
        if _n >= 8:
            _idx = [0, _n // 4, _n // 2, 3 * _n // 4, _n - 1]
            print(f"      首条 episode (T={_n}) 剖面 [0,25%,50%,75%,末]: "
                  f"{[round(float(w[i]) / _scale, 3) for i in _idx]}")
    print()

    if args.verify:
        print("  自检:")
        ok = True
        if cfg.normalize_mean and abs(float(w.mean()) - 1.0) > 1e-4:
            print(f"    ❌ mean(w) = {float(w.mean()):.6f} != 1")
            ok = False
        else:
            print("    ✅ mean(w) == 1（损失量级与纯 BC 可比）")
        if bool((w <= 0).any()):
            print("    ❌ 存在非正权重")
            ok = False
        else:
            print("    ✅ 所有权重 > 0")
        if float(w.std(unbiased=False)) < 1e-6:
            print("    ⚠️  权重几乎恒定 —— 等价于纯 BC（gamma=1 时属预期）")
        else:
            print("    ✅ 权重有区分度（加权确实生效）")
        if not ok:
            return 1

    save_weights(args.out, weights, cfg, episode_lengths=lengths)
    print()
    print(f"  ✅ 已写出: {args.out}")
    print("     训练时把 data.adv_weight_path 指向它即可启用 advantage 加权。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
