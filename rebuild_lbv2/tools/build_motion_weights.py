#!/usr/bin/env python3
"""生成「低运动窗口降权」sidecar。

思路
----
50 个任务 × 50 条 clean 轨迹里，有大量"几乎不动"的帧（抓稳之后的等待、任务完成后的
静止、以及对齐前的微调）。这些帧占掉大量梯度权重，却教不出任何推进动作；实测评测侧
4,611 个失败 episode **100% 是跑满 400 步超时**（不是早失败），说明模型学到了"待着不动"
这个安全动作。本脚本按 release 数据**自身**的动作位移统计，给静止窗口降权。

合规
----
* 训练全程零环境调用；
* 权重只由 release 数据集的动作张量自身统计得到（不涉及成功率/奖励/评测结果）；
* 属于赛题规则 2.1「按数据统计量重加权」；
* 产出格式与 ``tools/build_adv_weights.py`` 完全一致，因此训练侧复用既有
  ``--data.adv_weight_path`` 通路，**不需要改数据加载代码**。

权重定义
--------
对第 e 条轨迹的动作序列 ``A ∈ R^{T×D}``：
1. 逐维用全体数据的标准差 ``s_d`` 归一化（避免大量纲维度主导）；
2. 逐帧位移 ``δ_t = mean_d |A[t+1,d] − A[t,d]| / s_d``（末帧复制前一帧）；
3. 平滑 ``m_t = mean(δ_t … δ_{t+W−1})``，W = ``--window``（默认 10）；
4. ``w_t = clamp((m_t / median(m)) ** 0.5, lo, hi)``（默认 lo=0.6, hi=1.0）；
5. 全库归一化到 ``mean(w) = 1``（与 advantage 权重的归约口径一致，保持损失量级可比）。

输出
----
**每个任务一份** sidecar：``<output-dir>/<data_name>.pt``，内容为
``torch.save({"weights", "config", "episode_lengths", "num_samples"})``。
逐任务输出是必须的：``VLADataset`` 会校验 sidecar 样本数 == 本任务样本数，
而多任务训练会把同一份 config 传给每个任务的 dataset。
``episode_lengths`` 让训练侧用 ``(episode_index, frame_index)`` 定位权重。

用法
----
    python tools/build_motion_weights.py \
        --data-root   /path/datasets/robotwin/lerobot \
        --manifest    /path/datasets/robotwin/clean_training_data.txt \
        --output-dir  /path/datasets/robotwin/motion_weights_clean50

    # 训练：--data.adv_weight_path=/path/datasets/robotwin/motion_weights_clean50

``--manifest`` 与训练的 ``data.train_path`` 必须是同一份清单（逐任务顺序一致）；
脚本会按清单顺序拼接各任务的权重，并在写盘前断言总帧数一致。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #
def _read_episode_lengths(root: str) -> Tuple[List[int], dict]:
    """读每个 episode 的帧数（v2.x episodes.jsonl 与 v3 episodes/*.parquet 都支持）。"""
    meta_dir = os.path.join(root, "meta")
    info = {}
    info_path = os.path.join(meta_dir, "info.json")
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

    lengths: List[int] = []
    episodes_dir = os.path.join(meta_dir, "episodes")
    if os.path.isdir(episodes_dir):
        try:
            import pyarrow.parquet as pq

            files = sorted(glob.glob(os.path.join(episodes_dir, "**", "*.parquet"), recursive=True))
            rows: List[Tuple[int, int]] = []
            for fp in files:
                table = pq.read_table(fp, columns=["episode_index", "length"])
                rows.extend(zip(table.column("episode_index").to_pylist(),
                                table.column("length").to_pylist()))
            rows.sort(key=lambda r: r[0])
            lengths = [int(r[1]) for r in rows]
        except Exception:
            lengths = []

    if not lengths:
        jsonl = os.path.join(meta_dir, "episodes.jsonl")
        if os.path.isfile(jsonl):
            rows = []
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        rows.append((int(rec["episode_index"]), int(rec["length"])))
            rows.sort(key=lambda r: r[0])
            lengths = [r[1] for r in rows]

    if not lengths:
        raise FileNotFoundError(f"在 {meta_dir} 下找不到 episode 元数据")
    return lengths, info


def _read_actions(root: str, lengths: Sequence[int]) -> np.ndarray:
    """按 episode 顺序读出动作，返回 (sum(lengths), D) 的 float32 数组。"""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"在 {root}/data 下找不到 parquet 文件")

    chunks = []
    for fp in files:
        table = pq.read_table(fp, columns=["episode_index", "frame_index", "action"])
        ep = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
        fr = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        act = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        if act.ndim == 1:  # 单维动作
            act = act[:, None]
        chunks.append((ep, fr, act))

    ep = np.concatenate([c[0] for c in chunks])
    fr = np.concatenate([c[1] for c in chunks])
    act = np.concatenate([c[2] for c in chunks], axis=0)

    order = np.lexsort((fr, ep))
    ep, fr, act = ep[order], fr[order], act[order]

    # 校验：episode 连续、帧号从 0 连续递增、总帧数与 meta 一致
    n_expected = int(sum(lengths))
    if act.shape[0] != n_expected:
        raise ValueError(
            f"数据帧数 ({act.shape[0]}) 与 episode 长度之和 ({n_expected}) 不一致，"
            "数据集可能不完整"
        )
    for e, ln in enumerate(lengths):
        mask = ep == e
        got = int(mask.sum())
        if got != int(ln):
            raise ValueError(f"episode {e} 帧数 {got} != meta 声明的 {ln}")
        if got and not np.array_equal(fr[mask], np.arange(got)):
            raise ValueError(f"episode {e} 的 frame_index 不是 0..{got-1} 连续")
    return act


# --------------------------------------------------------------------------- #
# 权重
# --------------------------------------------------------------------------- #
def motion_weights(
    actions: np.ndarray,
    lengths: Sequence[int],
    window: int = 10,
    lo: float = 0.6,
    hi: float = 1.0,
    power: float = 0.5,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, dict]:
    if actions.ndim != 2 or not actions.size or not np.isfinite(actions).all():
        raise ValueError("actions 必须是非空、有限的二维数组")
    if not lengths or any(int(n) != n or n <= 0 for n in lengths) or sum(lengths) != len(actions):
        raise ValueError("episode 长度必须为正整数且总和等于动作帧数")
    if not isinstance(window, int) or window < 1 or not (0 < lo <= hi) or power <= 0:
        raise ValueError("要求 window>=1、0<lo<=hi、power>0")
    a = actions.astype(np.float32, copy=False)
    scale = a.std(axis=0)
    scale = np.where(scale < eps, 1.0, scale).astype(np.float32)

    delta = np.zeros(a.shape[0], dtype=np.float32)
    off = 0
    for ln in lengths:
        seg = a[off: off + ln]
        if ln >= 2:
            d = np.abs(seg[1:] - seg[:-1]) / scale
            d = d.mean(axis=1)
            delta[off: off + ln - 1] = d
            delta[off + ln - 1] = d[-1]
        off += ln

    # 每条轨迹独立平滑，窗口不得跨 episode 边界。
    if window > 1:
        pad = window // 2
        kernel = np.ones(window, dtype=np.float32) / window
        motion = np.empty_like(delta)
        off = 0
        for ln in lengths:
            padded = np.pad(delta[off:off + ln], (pad, window - 1 - pad), mode="edge")
            motion[off:off + ln] = np.convolve(padded, kernel, mode="valid")
            off += ln
    else:
        motion = delta

    ref = float(np.median(motion)) if motion.size else 1.0
    ref = max(ref, eps)
    w = np.power(motion / ref, power)
    w = np.clip(w, lo, hi).astype(np.float32)
    mean_pre = float(w.mean())
    w = w / max(mean_pre, eps)  # 归一化到 mean(w)=1（与 advantage 权重同口径）

    stats = {
        "kind": "motion_window_downweight",
        "window": window,
        "clip": [lo, hi],
        "power": power,
        "motion_median": ref,
        "frac_at_lo": float((w <= (lo / max(mean_pre, eps)) * 1.001).mean()),
        "mean_pre_norm": mean_pre,
        "mean_after_norm": float(w.mean()),
        "num_samples": int(w.size),
    }
    return w, stats


def _iter_manifest(path: str) -> List[Tuple[str, str]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"清单行格式异常（需要 '<data_name> <repo_path>'）：{line!r}")
            if parts[0] in {name for name, _ in out}:
                raise ValueError(f"清单任务名重复：{parts[0]}")
            out.append((parts[0], parts[1]))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="生成低运动窗口降权 sidecar（每任务一份）")
    p.add_argument("--data-root", required=True,
                   help="LeRobot 数据集根目录；若 --manifest 里是绝对路径则忽略此项")
    p.add_argument("--manifest", required=True,
                   help="与训练 data.train_path 完全一致的清单（只有 data_name 顺序需要一致）")
    p.add_argument("--output-dir", required=True,
                   help="输出目录；会为每个任务写 <data_name>.pt（多任务训练必须用目录）")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--lo", type=float, default=0.6)
    p.add_argument("--hi", type=float, default=1.0)
    p.add_argument("--power", type=float, default=0.5)
    p.add_argument("--dry-run", action="store_true", help="只打印统计，不写盘")
    args = p.parse_args(argv)

    entries = _iter_manifest(args.manifest)
    print(f"[motion] 清单任务数: {len(entries)}")

    per_task = {}
    summary = []
    for i, (name, rel) in enumerate(entries):
        root = rel if os.path.isabs(rel) else os.path.join(args.data_root, rel)
        if not os.path.isdir(root):
            print(f"[motion] 跳过（目录不存在）：{root}", file=sys.stderr)
            return 2
        lengths, _info = _read_episode_lengths(root)
        actions = _read_actions(root, lengths)
        w, stats = motion_weights(actions, lengths, window=args.window,
                                  lo=args.lo, hi=args.hi, power=args.power)
        payload = {
            "weights": torch.from_numpy(w.astype(np.float32)),
            "config": {
                "kind": "motion_window_downweight",
                "data_name": name,
                "window": args.window,
                "clip": [args.lo, args.hi],
                "power": args.power,
                "manifest": os.path.abspath(args.manifest),
                **stats,
            },
            "episode_lengths": [int(x) for x in lengths],
            "num_samples": int(w.size),
        }
        per_task[name] = stats
        out_path = os.path.join(args.output_dir, f"{name}.pt")
        if not args.dry_run:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(payload, out_path)
        summary.append((name, w.size, float(w.mean()), stats["frac_at_lo"]))
        print(f"  [{i+1:2d}/{len(entries)}] {name:28s} episodes={len(lengths):3d} "
              f"frames={w.size:6d} 静止占比={stats['frac_at_lo']*100:5.1f}% "
              f"mean={w.mean():.4f}")

    total = sum(s[1] for s in summary)
    print(f"[motion] 任务数 {len(summary)}，样本合计 {total}")
    if args.dry_run:
        print("[motion] dry-run，未写盘")
        return 0
    print(f"[motion] 已写出 {len(summary)} 份 sidecar 到：{args.output_dir}")
    print("[motion] 训练时传 --data.adv_weight_path=<该目录>（目录形式，按 <data_name>.pt 逐任务匹配）")

    print("[motion] 注意 yaml 里不要写 null（会被解析成字符串 'null' 导致找文件崩溃）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
