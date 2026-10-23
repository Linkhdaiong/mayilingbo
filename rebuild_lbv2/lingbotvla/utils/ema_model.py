"""EMA 影子模块（补官方 `lingbotvla/utils/ema.py` 缺失的容器）。

为什么单独一个文件
------------------
官方仓库**已经**把 EMA 的存取接线做完了：
  * ``Checkpointer.save``：``state["ema"]`` 非 None 时写入 ``checkpoints/global_step_N/ema/``；
  * ``AsyncHFCheckpointSaver``：``save_ema=True`` 时额外导出 ``ema_hf_ckpt/``；
  * ``ckpt_to_state_dict(..., ema=True)`` 负责读回。
但 ``tasks/vla/train_lingbotvla.py`` 与 ``train_pi0.py`` 里都只写了 ``"ema": None``，
从未创建影子模型，所以整条通路是死的。本模块只补“容器 + 更新”这一块，
**不改官方 ``lingbotvla/utils/ema.py``**（那个 ``ema_update`` 要求目标参数是 fp32，
6.4B 参数即 25.5 GB 显存，不可用）。

设计
----
* 影子参数逐层镜像主模型，**参数名与主模型完全一致**（不额外加前缀），因此官方的
  DCP 存取与 HF 导出可以直接使用；
* 影子参数用 bf16 保存（约 12.8 GB），主模型精度不变；
* ``decay <= 0`` 时不创建对象，训练行为与改动前逐位一致；
* 合规依据：赛题规则 2.3「训练期 EMA（产出仍为一份权重）」。EMA 不引入任何新的
  监督信号，也零环境调用。

关于 bf16 影子与随机舍入
----------------------
bf16 只有 8 位尾数，相对 ULP ≈ 0.39%、半 ULP ≈ 0.195%；而 ``decay=0.999`` 时单步
更新量 ≈ ``1e-3·|src| ≈ 0.1%·|dst|`` **小于半 ULP**，直接用
``dst.mul_(decay).add_(src, alpha=alpha)`` 会让两次舍入都把结果送回原值，影子权重被
"冻结"在初始化值上且不报错。因此 ``update()`` 改为逐张量用 fp32 中间量计算 +
随机舍入（stochastic rounding）写回 bf16，详见 :class:`EMAModel` docstring。

用法（train_lingbotvla.py 内）
-----------------------------
    ema_model = None
    if getattr(args.train, "ema_decay", 0.0) > 0:
        ema_model = EMAModel(model, decay=args.train.ema_decay)
    ...
    state = {"model": model, "ema": ema_model, "optimizer": optimizer, "extra_state": {...}}
    ...
    optimizer.step()
    if ema_model is not None:
        ema_model.update(model)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

__all__ = ["EMAModel"]

logger = logging.getLogger(__name__)

# 随机舍入 / 距离计算的分块元素数：避免为最大的单张量额外申请整块 fp32 临时量
# （16M 元素 = 64 MB fp32），峰值额外显存只与分块大小有关，而不是 12.8 GB 级。
_SR_CHUNK = 1 << 24


def _local_view(t: torch.Tensor) -> torch.Tensor:
    """DTensor → 本地分片张量（与 DTensor 共享存储）；普通张量原样返回。

    FSDP2 下主模型参数是 DTensor，而 bit-view（``view(torch.int32)``）不在 DTensor 的
    分片传播规则里，因此随机舍入统一在本地视图上做（单卡 1 rank 时本地视图就是全量）。
    """
    to_local = getattr(t, "to_local", None)
    if callable(to_local) and hasattr(t, "device_mesh"):
        return to_local()
    return t


def _stochastic_round_bf16_(dst: torch.Tensor, val: torch.Tensor, generator: Optional[torch.Generator] = None) -> None:
    """把 fp32 值 ``val`` 以**随机舍入**写回 bf16 张量 ``dst``（原地）。

    bf16 的位模式就是 fp32 的高 16 位，所以"舍入到 bf16"只取决于低 16 位是否进位：
    给 fp32 位模式加上 ``U(0, 2^16)`` 的随机整数再截断低 16 位，即"以概率
    ``low16 / 2^16`` 向上进位"，期望上无偏。低 16 位为 0 的值加随机数后若不跨过
    2^16 边界则不进位，因此不动点不会被系统性打破；而 0.1% 量级的增量（低 16 位
    ≈ 256）能以 ≈0.4% 的概率进位，期望增量正好等于原值。符号/量级由位模式本身保证
    （符号位与阶码位于高 16 位，加法只影响尾数低位与少量进位）。

    NaN/Inf 元素不做舍入，直接保留 ``dst`` 原值（跳过）。
    """
    dst = _local_view(dst)
    val = _local_view(val)
    if dst.shape != val.shape:
        raise RuntimeError(
            "EMA 随机舍入：影子与主模型的本地分片形状不一致（DTensor 布局不同），"
            "无法安全原地写回；请把 ema_dtype 设为 float32 或关闭 EMA。"
        )
    if not dst.is_contiguous():
        # 影子参数来自 clone，正常一定 contiguous；万一不是，reshape 会拿到副本
        # 导致写入丢失（等于又一次"静默失效"），这里显式走临时张量
        tmp = dst.contiguous()
        _stochastic_round_bf16_(tmp, val, generator)
        dst.copy_(tmp)
        return
    dst_flat = dst.reshape(-1)
    val_flat = val.reshape(-1)
    n = val_flat.numel()
    for start in range(0, n, _SR_CHUNK):
        end = min(start + _SR_CHUNK, n)
        chunk = val_flat[start:end].contiguous()
        # 用 int64 承载位模式，避免极大指数（bits 接近 0x7F7Fxxxx）时 int32 加法溢出
        bits = chunk.view(torch.int32).to(torch.int64)
        noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int64, device=bits.device, generator=generator)
        # & -65536 == 丢弃低 16 位，只保留 fp32 的高 16 位（即 bf16）
        rounded = ((bits + noise) & -65536).to(torch.int32).view(torch.float32).to(torch.bfloat16)
        finite = torch.isfinite(chunk)
        dst_flat[start:end].copy_(torch.where(finite, rounded, dst_flat[start:end]))


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    """``||a - b||_2 / ||b||_2``；分块累加，本地形状不一致时返回 None。"""
    a_flat = _local_view(a).detach().reshape(-1)
    b_flat = _local_view(b).detach().reshape(-1)
    if a_flat.numel() != b_flat.numel():
        return None
    num = 0.0
    den = 0.0
    n = a_flat.numel()
    for start in range(0, n, _SR_CHUNK):
        end = min(start + _SR_CHUNK, n)
        x = a_flat[start:end].to(torch.float32)
        y = b_flat[start:end].to(torch.float32)
        num += float((x - y).pow(2).sum().item())
        den += float(y.pow(2).sum().item())
    return (num ** 0.5) / (den ** 0.5 + 1e-12)


class EMAModel(nn.Module):
    """与主模型同构的影子模块（bf16 影子参数）。

    为什么必须随机舍入（stochastic rounding）
    ----------------------------------------
    bf16 尾数 8 位 ⇒ 相对 ULP ≈ 0.39%、半 ULP ≈ 0.195%；而 ``decay=0.999`` 时单步
    增量 = ``(1-decay)·|src| ≈ 0.1%·|dst|`` **小于半 ULP** —— ``0.999·dst`` 舍入回
    ``dst``、``+0.001·src`` 也舍入回 ``dst``，影子权重永远停在初始化值（整条 EMA
    轨道变空操作，却不报错）。随机舍入以概率 ``low16/2^16`` 向上进位，使 0.1% 量级
    的增量在期望上按比例累积，从而既保留 bf16 影子的显存优势又不冻结。

    Args:
        model: 主模型（可在 FSDP2 / DTensor 之下；单卡时语义等价于本地张量）。
        decay: 滑动平均系数，推荐 0.999。
        dtype: 影子参数精度，默认 bfloat16；``torch.float32`` 时跳过随机舍入、精确计算。
        update_buffers: 浮点 buffer 是否也做 EMA；整型 buffer（如计数器）一律直拷。
        log_interval: 每多少次 update 打印一次影子与主模型的相对距离（0 表示关闭）。
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        dtype: torch.dtype = torch.bfloat16,
        update_buffers: bool = True,
        log_interval: int = 100,
    ) -> None:
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"ema decay 必须在 (0, 1) 区间内，收到 {decay}")
        self.decay = float(decay)
        self.ema_dtype = dtype
        self.update_buffers_flag = bool(update_buffers)
        self.log_interval = int(log_interval)
        self.num_updates = 0
        self._param_names: List[str] = []
        self._buffer_names: List[str] = []

        # ---------- 用 self 作为根，逐层镜像主模型的模块树 ----------
        modules: Dict[str, nn.Module] = {"": self}
        for name, _mod in model.named_modules():
            if name == "":
                continue
            parent, _, leaf = name.rpartition(".")
            child = nn.Module()
            setattr(modules[parent], leaf, child)
            modules[name] = child

        for name, p in model.named_parameters():
            parent, _, leaf = name.rpartition(".")
            shadow = p.detach().to(dtype).clone()
            modules[parent].register_parameter(leaf, nn.Parameter(shadow, requires_grad=False))
            self._param_names.append(name)

        for name, b in model.named_buffers():
            parent, _, leaf = name.rpartition(".")
            holder = modules[parent]
            if leaf in dict(holder.named_parameters()) or leaf in dict(holder.named_buffers()):
                continue
            holder.register_buffer(leaf, b.detach().clone())
            self._buffer_names.append(name)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _ema_update_tensor_(self, dst: torch.Tensor, src: torch.Tensor, decay: float, alpha: float) -> None:
        """单个张量的 EMA 更新：fp32 中间量计算，再按影子精度写回。"""
        # new = decay·dst + alpha·src，全部在 fp32 里算（峰值额外显存 ≈ 2× 本张量；
        # 随机舍入内部再按 64 MB 分块，不会为最大张量整块申请临时量）
        new = dst.to(torch.float32).mul(decay).add_(src.to(torch.float32), alpha=alpha)
        if dst.dtype == torch.bfloat16:
            # bf16 写回必须随机舍入，否则 0.1% 量级的增量会被半 ULP 吞掉
            _stochastic_round_bf16_(dst, new)
        elif dst.dtype == torch.float32:
            # fp32 影子：跳过随机舍入，直接精确写回
            dst.copy_(new)
        else:
            # 其它低精度（本仓库未使用 ema_dtype=fp16 等）：常规就近舍入
            dst.copy_(new.to(dst.dtype))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """在 ``optimizer.step()`` 之后调用一次。"""
        decay = self.decay
        alpha = 1.0 - decay
        src_params = dict(model.named_parameters())
        for name, dst in self.named_parameters():
            src = src_params.get(name)
            if src is None:
                continue
            self._ema_update_tensor_(dst, src.detach(), decay, alpha)

        if self.update_buffers_flag:
            src_bufs = dict(model.named_buffers())
            for name, dst in self.named_buffers():
                src = src_bufs.get(name)
                if src is None:
                    continue
                src = src.detach()
                if not torch.is_floating_point(dst):
                    dst.copy_(src.to(dst.device))
                    continue
                self._ema_update_tensor_(dst, src, decay, alpha)

        self.num_updates += 1
        self._maybe_log(model)

    @torch.no_grad()
    def ema_distance(self, model: nn.Module, max_tensors: int = 8) -> float:
        """抽样若干张量，返回 ``||ema - model||_2 / ||model||_2`` 的平均值。

        可观测性入口（也供测试调用）：bf16 影子被舍入吞掉时该值恒为 0。
        抽样按参数名等间距取，避免只看第一层（embedding）而误判整体。
        """
        src_params = dict(model.named_parameters())
        dst_params = dict(self.named_parameters())
        names = [n for n in self._param_names if n in src_params and n in dst_params]
        if not names:
            return 0.0
        max_tensors = max(1, int(max_tensors))
        if len(names) > max_tensors:
            step = len(names) / max_tensors
            names = [names[int(i * step)] for i in range(max_tensors)]
        dists: List[float] = []
        for name in names:
            d = _rel_l2(dst_params[name], src_params[name])
            if d is not None:
                dists.append(d)
        if not dists:
            return 0.0
        return sum(dists) / len(dists)

    @torch.no_grad()
    def _maybe_log(self, model: nn.Module) -> None:
        """每 ``log_interval`` 次更新（以及第 1 次更新）打印一次相对距离。"""
        if self.log_interval <= 0:
            return
        if self.num_updates != 1 and self.num_updates % self.log_interval != 0:
            return
        dist = self.ema_distance(model)
        logger.info(
            "[EMA] update=%d decay=%.4f dtype=%s rel_dist(ema vs model)=%.3e",
            self.num_updates,
            self.decay,
            self.ema_dtype,
            dist,
        )
        if dist == 0.0:
            logger.warning(
                "[EMA] 第 %d 次更新后 ||ema-model||/||model|| 仍为 0：EMA 未生效"
                "（影子权重没跟上主模型，低精度写回可能又被舍入吞掉了）。",
                self.num_updates,
            )

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """把 EMA 权重写回主模型（仅调试/导出验证用；正式流程走 ``ema_hf_ckpt``）。"""
        src_params = dict(self.named_parameters())
        for name, dst in model.named_parameters():
            src = src_params.get(name)
            if src is not None:
                dst.data.copy_(src.data.to(dst.dtype))

    def extra_repr(self) -> str:
        return (
            f"decay={self.decay}, dtype={self.ema_dtype}, num_updates={self.num_updates}, "
            f"params={len(self._param_names)}, buffers={len(self._buffer_names)}"
        )
