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
"""Offline RL: advantage-weighted flow matching (AWAC-style).

合规依据（比赛规则 3.4）
--------------------------------------------------------------------------
规则 3.4 不按方法名（RL / BC / 蒸馏）划分禁界，一律以**监督信号的获取方式**判定。
明确允许「训练全程零环境调用、且监督信号（含 reward、标签、样本权重）全部可自
RoboTwin release 数据内部获得或推导」的学习方法，**包括但不限于 offline RL
（IQL / CQL / AWAC 等）、价值或优势加权、advantage-weighted BC**。

本模块严格落在该允许范围内：

  * **零环境调用** —— 不渲染、不交互采样、不调用 ``check_success()``；
  * **reward 自 release 数据内部推导** —— clean demonstration 是专家脚本采集、
    且采集时通过 ``check_success()`` 的**成功轨迹**（这是数据集的构造性质，
    不是我们运行环境得来的）。因此每条 episode 的最后一步 ``r = 1``，其余 ``r = 0``，
    该 reward 完全由 episode 边界推出；
  * **不涉及官方 robotwin 后训练权重** —— 只用到数据自身的长度信息。

数学形式
--------------------------------------------------------------------------
设某 episode 长度为 ``T``（帧数），第 ``t`` 帧（0-based）的折扣回报
（return-to-go）在稀疏终局 reward 下有闭式解::

    G_t = gamma ** (T - 1 - t)

优势与权重::

    A_t = G_t - mean(G)
    w_t = exp(beta * A_t / std(G))      再裁剪到 [w_min, w_max]
    w_t = w_t / mean(w)                 归一化，保持损失量级不变

``gamma == 1`` 时 ``G_t`` 恒为 1、``A_t`` 恒为 0、``w_t`` 恒为 1 —— 即退化为纯 BC，
可作为 A/B 消融的零效应基线。

也提供 ``progress`` 模式（``G_t = t / (T - 1)``，线性进度），同样是纯数据推导量。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "AdvantageConfig",
    "episode_lengths_to_returns",
    "advantage_weights",
    "build_weights_from_lengths",
    "read_lerobot_episode_lengths",
    "build_weights_from_lerobot",
    "save_weights",
    "load_weights",
]


@dataclass
class AdvantageConfig:
    """构造 advantage 权重的超参。"""

    #: 折扣因子。``1.0`` 表示不引入折扣（此时权重恒为 1，等价纯 BC）。
    gamma: float = 0.99
    #: 优势缩放系数，对应 AWAC 里的温度倒数 ``1/lambda``。越小越接近纯 BC。
    #:
    #: 权重按优势的**标准差**标准化后再乘 beta，因此该值是尺度无关的。
    #: 默认 0.25 是**刻意取小**的：见下面 ``w_min``/``w_max`` 关于集中度的说明。
    beta: float = 0.25
    #: 权重裁剪区间。**这是防止加权退化的关键**：它直接决定了最坏情况下
    #: 某个训练阶段被压制/放大的上限。
    #:
    #: 为什么必须收窄：``reward_mode="terminal_gamma"`` 下回报随「离终点的距离」
    #: 单调变化，若裁剪区间放到 [0.1, 10]，实测典型 300 帧 episode 会呈现
    #:
    #:     t=0 -> w≈0.20    t=150 -> w≈0.39    t=225 -> w≈1.03    t=299 -> w≈5.58
    #:     w < 0.5 的帧占比 58.7%
    #:
    #: 即**超过一半的 episode（接近、抓取、运输阶段）被压掉 2 倍以上**，训练几乎
    #: 只集中在每条轨迹的最后四分之一。对 clean2random 这类要求全程视觉鲁棒性的
    #: 任务，这通常是**负收益**。
    #:
    #: 默认 [0.5, 2.0] 把最坏情况硬性限制在 ±2 倍以内：即使 ``beta`` 被误设成
    #: 很大的值，也只会退化成「轻微重加权」而不会退化成「只训收尾」。
    w_min: float = 0.5
    #: 见 ``w_min``。
    w_max: float = 2.0
    #: ``"terminal_gamma"`` 用 gamma**(T-1-t)；``"progress"`` 用 t/(T-1)。
    reward_mode: str = "terminal_gamma"
    #: 归一化后是否强制 ``mean(w) == 1``（保持损失量级与纯 BC 可比）。
    normalize_mean: bool = True

    def validate(self) -> None:
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError(f"gamma 必须在 (0, 1]，收到 {self.gamma}")
        if self.beta < 0:
            raise ValueError(f"beta 必须 >= 0，收到 {self.beta}")
        if not (0.0 < self.w_min <= self.w_max):
            raise ValueError(f"要求 0 < w_min <= w_max，收到 {self.w_min}, {self.w_max}")
        if self.reward_mode not in ("terminal_gamma", "progress"):
            raise ValueError(f"未知 reward_mode: {self.reward_mode}")


def episode_lengths_to_returns(
    lengths: Sequence[int],
    cfg: AdvantageConfig,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """把每个 episode 的帧数展开成**逐帧**的回报 ``G_t``。

    Args:
        lengths: 每个 episode 的帧数，按数据集中的先后顺序。
        cfg: 见 :class:`AdvantageConfig`。
        dtype: 输出 dtype。

    Returns:
        一维张量，长度等于 ``sum(lengths)``，与数据集样本索引一一对应
        （LeRobot 数据集按 episode 顺序、episode 内按 frame 顺序存放）。
    """
    cfg.validate()
    chunks: List[torch.Tensor] = []
    for n in lengths:
        n = int(n)
        if n <= 0:
            # 空 episode 不产生样本；保持与数据集长度一致，跳过即可。
            continue
        t = torch.arange(n, dtype=dtype)
        if cfg.reward_mode == "terminal_gamma":
            # 末帧 t = n-1 的 return-to-go 为 1（终局 reward=1）。
            g = torch.pow(torch.tensor(cfg.gamma, dtype=dtype), (n - 1) - t)
        else:  # progress
            denom = max(n - 1, 1)
            g = t / denom
        chunks.append(g)
    if not chunks:
        return torch.zeros(0, dtype=dtype)
    return torch.cat(chunks, dim=0)


def advantage_weights(returns: torch.Tensor, cfg: AdvantageConfig) -> torch.Tensor:
    """由逐帧回报算出 advantage 权重。

    Args:
        returns: :func:`episode_lengths_to_returns` 的输出，形状 ``(N,)``。
        cfg: 见 :class:`AdvantageConfig`。

    Returns:
        形状 ``(N,)`` 的权重张量。``cfg.normalize_mean`` 为真时 ``mean(w) == 1``。
    """
    cfg.validate()
    if returns.numel() == 0:
        return returns.clone()

    g = returns.to(torch.float32)
    adv = g - g.mean()
    scale = adv.std(unbiased=False)
    if float(scale) < 1e-12:
        # 回报完全一致（例如 gamma=1）→ 优势为 0 → 权重恒为 1，退化为纯 BC。
        w = torch.ones_like(g)
    else:
        w = torch.exp(cfg.beta * adv / scale)
    w = w.clamp(min=cfg.w_min, max=cfg.w_max)
    if cfg.normalize_mean:
        w = w / w.mean().clamp(min=1e-12)
    return w


def build_weights_from_lengths(
    lengths: Sequence[int], cfg: AdvantageConfig
) -> torch.Tensor:
    """``lengths -> returns -> weights`` 的便捷封装。"""
    returns = episode_lengths_to_returns(lengths, cfg)
    return advantage_weights(returns, cfg)


# --------------------------------------------------------------------------- #
# LeRobot 数据集读取
# --------------------------------------------------------------------------- #
def read_lerobot_episode_lengths(root: str) -> Tuple[List[int], dict]:
    """从 LeRobot 数据集根目录读出每个 episode 的帧数。

    兼容两种布局：
      * v3：``meta/episodes/**/*.parquet``（含 ``length`` 列）
      * v2.x：``meta/episodes.jsonl``（每行 ``{"episode_index": i, "length": n}``）

    Returns:
        ``(lengths, info)``，``lengths`` 按 ``episode_index`` 升序；``info`` 是
        ``meta/info.json`` 的内容（含 ``fps``、``total_frames`` 等）。
    """
    meta_dir = os.path.join(root, "meta")
    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(f"找不到 LeRobot meta 目录: {meta_dir}")

    info = {}
    info_path = os.path.join(meta_dir, "info.json")
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

    lengths: List[int] = []

    # --- v3: meta/episodes/**/*.parquet ---
    episodes_dir = os.path.join(meta_dir, "episodes")
    if os.path.isdir(episodes_dir):
        try:
            import glob

            import pyarrow.parquet as pq

            files = sorted(glob.glob(os.path.join(episodes_dir, "**", "*.parquet"), recursive=True))
            if files:
                rows: List[Tuple[int, int]] = []
                for fp in files:
                    table = pq.read_table(fp, columns=["episode_index", "length"])
                    idx = table.column("episode_index").to_pylist()
                    ln = table.column("length").to_pylist()
                    rows.extend(zip(idx, ln))
                rows.sort(key=lambda r: r[0])
                lengths = [int(r[1]) for r in rows]
        except Exception:
            lengths = []

    # --- v2.x: meta/episodes.jsonl ---
    if not lengths:
        jsonl = os.path.join(meta_dir, "episodes.jsonl")
        if os.path.isfile(jsonl):
            rows = []
            with open(jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rows.append((int(rec["episode_index"]), int(rec["length"])))
            rows.sort(key=lambda r: r[0])
            lengths = [r[1] for r in rows]

    if not lengths:
        raise FileNotFoundError(
            f"在 {meta_dir} 下找不到 episodes 元数据（既无 episodes/**/*.parquet 也无 episodes.jsonl）"
        )

    total_frames = info.get("total_frames")
    if total_frames is not None and int(total_frames) != sum(lengths):
        raise ValueError(
            f"episode 帧数之和 ({sum(lengths)}) 与 info.json 的 total_frames ({total_frames}) 不一致，"
            "数据集可能不完整；拒绝生成权重。"
        )
    return lengths, info


def build_weights_from_lerobot(
    root: str, cfg: AdvantageConfig
) -> Tuple[torch.Tensor, List[int], dict]:
    """从 LeRobot 数据集直接构造逐样本权重。

    Returns:
        ``(weights, lengths, info)``。
    """
    lengths, info = read_lerobot_episode_lengths(root)
    weights = build_weights_from_lengths(lengths, cfg)
    return weights, lengths, info


# --------------------------------------------------------------------------- #
# 存取
# --------------------------------------------------------------------------- #
def save_weights(
    path: str,
    weights: torch.Tensor,
    cfg: Optional[AdvantageConfig] = None,
    episode_lengths: Optional[Sequence[int]] = None,
) -> None:
    """保存权重 sidecar（含超参与 episode 长度，便于复现与审计）。

    ``episode_lengths`` 会被一并写入：训练侧据此用
    ``(episode_index, frame_index)`` 定位权重，而不是依赖 LeRobot 底层样本的
    物理排列顺序。这消除了「转换脚本版本差异导致权重与样本错位」的风险。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "weights": weights.to(torch.float32).cpu(),
        "config": asdict(cfg) if cfg is not None else None,
        "episode_lengths": (
            [int(x) for x in episode_lengths] if episode_lengths is not None else None
        ),
        "num_samples": int(weights.numel()),
    }
    torch.save(payload, path)


def load_weights(path: str) -> Tuple[torch.Tensor, Optional[dict]]:
    """读取权重 sidecar，返回 ``(weights, meta_dict)``。

    ``meta_dict`` 含超参（``gamma``/``beta``/...）与 ``episode_lengths``。
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "weights" in payload:
        meta = dict(payload.get("config") or {})
        if payload.get("episode_lengths") is not None:
            meta["episode_lengths"] = payload["episode_lengths"]
        return payload["weights"].to(torch.float32), (meta or None)
    # 兼容裸张量
    return payload.to(torch.float32), None
