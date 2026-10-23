"""stage-2 / DVAC 新增件的单元测试（在训练环境里跑：pytest -q tests/test_stage2_extra.py）。

这些测试**不需要数据集、不需要 GPU、不需要模型权重**，可在训练启动前 1 分钟内跑完。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# 1) 增广：形状/ dtype 不变、关闭时不改变像素、开启时不产生 NaN
# --------------------------------------------------------------------------- #
def test_aug_preserves_shape_and_dtype():
    from lingbotvla.data.vla_data.transform import (
        sample_visual_augmentation_params,
        apply_visual_augmentation,
    )

    torch.manual_seed(0)
    img = torch.rand(3, 224, 224)
    for _ in range(20):
        params = sample_visual_augmentation_params(img)
        out = apply_visual_augmentation(img, params)
        assert out.shape == img.shape
        assert out.dtype == img.dtype
        assert torch.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0

    # 视频形态 (T,C,H,W) 也要支持
    vid = torch.rand(2, 3, 224, 224)
    params = sample_visual_augmentation_params(vid)
    out = apply_visual_augmentation(vid, params)
    assert out.shape == vid.shape


def test_aug_identity_params_is_identity():
    from lingbotvla.data.vla_data.transform import apply_visual_augmentation

    img = torch.rand(3, 64, 64)
    params = {
        "brightness": torch.tensor(1.0),
        "contrast": torch.tensor(1.0),
        "saturation": torch.tensor(1.0),
    }
    out = apply_visual_augmentation(img, params)
    assert torch.allclose(out, img, atol=1e-5)


def test_aug_disable_geo_and_erase_via_env(monkeypatch):
    from lingbotvla.data.vla_data import transform as T

    monkeypatch.setitem(T.AUG_CFG, "p_geo", 0.0)
    monkeypatch.setitem(T.AUG_CFG, "p_erase", 0.0)
    monkeypatch.setitem(T.AUG_CFG, "p_noise", 0.0)
    img = torch.rand(3, 64, 64)
    for _ in range(10):
        p = T.sample_visual_augmentation_params(img)
        assert "geo" not in p and "erase" not in p and "noise_sigma" not in p


# --------------------------------------------------------------------------- #
# 2) 任务均衡采样：槽位 → 帧下标的映射
# --------------------------------------------------------------------------- #
def test_task_balanced_slot_mapping():
    from lingbotvla.data.vla_data.multi_vla_dataset import MultiVLADataset

    ds = object.__new__(MultiVLADataset)
    ds.dataset_start_index = [0, 100, 100 + 500]     # 三个任务，帧数 100 / 500 / 50
    ds._frames_per_task = [100, 500, 50]
    ds._n_tasks = 3
    ds.task_balanced = True

    seen_tasks = [0, 0, 0]
    for slot in range(3000):
        idx = ds._slot_to_index(slot)
        task = 0 if idx < 100 else (1 if idx < 600 else 2)
        bound = ds.dataset_start_index[task], ds.dataset_start_index[task] + ds._frames_per_task[task]
        assert bound[0] <= idx < bound[1], (slot, idx, task)
        seen_tasks[task] += 1
    # 每个任务拿到完全相同数量的槽位
    assert seen_tasks == [1000, 1000, 1000], seen_tasks


def test_task_balanced_disabled_is_identity():
    from lingbotvla.data.vla_data.multi_vla_dataset import MultiVLADataset

    ds = object.__new__(MultiVLADataset)
    ds.dataset_start_index = [0, 100]
    ds._frames_per_task = [100, 500]
    ds._n_tasks = 2
    ds.task_balanced = False
    for slot in (0, 7, 123):
        assert ds._slot_to_index(slot) == slot


# --------------------------------------------------------------------------- #
# 3) EMA：影子参数名与主模型完全一致、能收敛、state_dict 可往返
# --------------------------------------------------------------------------- #
class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.block = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        return self.block(self.a(x))


def test_ema_names_match_and_copies_buffers():
    from lingbotvla.utils.ema_model import EMAModel

    m = _Tiny()
    ema = EMAModel(m, decay=0.9, dtype=torch.float32)
    assert [n for n, _ in ema.named_parameters()] == [n for n, _ in m.named_parameters()]
    assert [n for n, _ in ema.named_buffers()] == [n for n, _ in m.named_buffers()]
    # 初始即等于主模型
    for (n1, p1), (n2, p2) in zip(ema.named_parameters(), m.named_parameters()):
        assert n1 == n2 and torch.allclose(p1, p2.to(p1.dtype))


def test_ema_moves_towards_model_and_roundtrips():
    from lingbotvla.utils.ema_model import EMAModel

    torch.manual_seed(0)
    m = _Tiny()
    ema = EMAModel(m, decay=0.9, dtype=torch.float32)
    before = {n: p.clone() for n, p in ema.named_parameters()}

    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    for _ in range(30):
        ema.update(m)

    for n, p in ema.named_parameters():
        assert torch.allclose(p, m.state_dict()[n], atol=1e-2), n
        assert not torch.allclose(p, before[n]), n

    sd = ema.state_dict()
    ema2 = EMAModel(m, decay=0.9, dtype=torch.float32)
    ema2.load_state_dict(sd)
    for n, p in ema2.named_parameters():
        assert torch.allclose(p, sd[n])


def test_ema_rejects_bad_decay():
    from lingbotvla.utils.ema_model import EMAModel

    with pytest.raises(ValueError):
        EMAModel(_Tiny(), decay=0.0)
    with pytest.raises(ValueError):
        EMAModel(_Tiny(), decay=1.0)


def test_ema_bf16_shadow_is_not_frozen():
    """bf16 影子 + decay=0.999：单步增量(0.1%) < 半 ULP(0.195%)。

    只有随机舍入才能让增量在期望上累积；旧的 ``dst.mul_(decay).add_(src)``
    会让影子权重永远等于初始化值（本用例会失败）。
    """
    from lingbotvla.utils.ema_model import EMAModel

    torch.manual_seed(0)
    m = _Tiny()
    ema = EMAModel(m, decay=0.999, dtype=torch.bfloat16, log_interval=0)
    before = {n: p.clone() for n, p in ema.named_parameters()}

    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    for _ in range(500):
        ema.update(m)

    moved = {n: not torch.equal(p, before[n]) for n, p in ema.named_parameters()}
    assert any(moved.values()), f"bf16 EMA 影子完全没动：{moved}"
    # 500 步后应跟随 1 - 0.999**500 ≈ 39% 的位移，distance 因此明显大于 0
    dist = ema.ema_distance(m)
    assert 0.1 < dist < 1.0, dist


# --------------------------------------------------------------------------- #
# 4) DVAC：执行长度决策（τ 只用历史 chunk；冷启动不截断）
# --------------------------------------------------------------------------- #
class _FakeServer:
    dvac_alpha = 1.0
    dvac_n_min = 5
    dvac_n_max = 50
    dvac_min_history = 2
    dvac_verbose = False
    _dvac_buf_maxlen = 8

    def __init__(self):
        self._dvac_buf = []
        self._dvac_trace = []


def _call(server, var, max_len=50):
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    return LingbotVLAv2Server._dvac_exec_len(server, var, max_len)


def _chunk(value, length=50):
    return torch.full((1, length), float(value), dtype=torch.float32)


def _warmup(server, value=0.01):
    """先喂满 dvac_min_history 个历史 chunk（冷启动段一律返回 max_len）。"""
    for _ in range(server.dvac_min_history):
        assert _call(server, _chunk(value)) == 50


def test_dvac_cold_start_returns_full_chunk():
    """历史不足 dvac_min_history 时不做截断，n = max_len（退回原版行为）。"""
    s = _FakeServer()
    # 第 1 个 chunk（历史为空）即便整段高方差，也不截断
    assert _call(s, _chunk(100.0)) == 50
    assert len(s._dvac_buf) == 1
    # 第 2 个 chunk 时历史只有 1 个 (< dvac_min_history=2) → 仍然不截断
    assert _call(s, _chunk(100.0)) == 50
    assert len(s._dvac_buf) == 2

    # 对照组：同样这段高方差 chunk，历史用低方差填满后就会被截断到 n_min
    s2 = _FakeServer()
    _warmup(s2, value=0.01)             # 历史 [0.01, 0.01] ⇒ τ=0.01
    assert _call(s2, _chunk(100.0)) == s2.dvac_n_min


def test_dvac_cuts_at_first_high_variance_index():
    s = _FakeServer()
    _warmup(s, value=0.01)              # 历史 μ≈0.01, σ=0 ⇒ τ=0.01
    # 前 12 步方差低，其余高 → 第一个 V(k)>τ 在 k=12
    v = torch.cat([torch.full((12,), 0.01), torch.full((38,), 10.0)]).unsqueeze(0)
    n = _call(s, v)
    assert 5 <= n <= 50
    assert n == 12, n


def test_dvac_all_low_takes_full_chunk():
    s = _FakeServer()
    _warmup(s, value=0.02)
    # 与历史持平 → 不触发截断（等于 τ 也不算 >）
    assert _call(s, _chunk(0.02)) == 50


def test_dvac_respects_bounds():
    s = _FakeServer()
    _warmup(s)
    v = torch.cat([torch.full((1,), 100.0), torch.zeros((1, 49))], dim=1)
    assert _call(s, v) == s.dvac_n_min
    s2 = _FakeServer()
    s2.dvac_n_max = 20
    _warmup(s2)
    # 整段低于 τ → n = max_len = 50，再被 n_max 压到 20
    assert _call(s2, torch.zeros(1, 50)) == 20


def test_dvac_buffer_is_bounded():
    s = _FakeServer()
    for _ in range(50):
        _call(s, _chunk(0.01))
    assert len(s._dvac_buf) <= s._dvac_buf_maxlen
    assert len(s._dvac_buf) == s._dvac_buf_maxlen


def test_dvac_threshold_uses_history_only():
    """τ 只用历史：当前 chunk 在算完 τ 之后才入缓冲区。

    历史 = [0.01, 0.03] ⇒ μ=0.02, σ=0.01；α=1 ⇒ τ=0.03。
    当前 chunk 恒为 0.035：0.035 > τ ⇒ 第 0 步即异常 ⇒ n = n_min = 5。
    （旧实现先把 0.035 入缓冲区：μ=0.025、σ≈0.0108、τ≈0.0358 > 0.035 ⇒ n=50，
      也就是"拿自己的均值当阈值"，这正是修复前的退化。）
    """
    s = _FakeServer()
    s._dvac_buf = [0.01, 0.03]
    assert _call(s, _chunk(0.035)) == s.dvac_n_min

    # 同一份历史、同一段曲线：α 现在真的能改变判决（旧实现里 α 几乎无效）
    s_a1 = _FakeServer()
    s_a1._dvac_buf = [0.01, 0.03]
    assert _call(s_a1, _chunk(0.025)) == 50        # τ=0.03，整段不超阈值
    s_a0 = _FakeServer()
    s_a0._dvac_buf = [0.01, 0.03]
    s_a0.dvac_alpha = 0.0
    assert _call(s_a0, _chunk(0.025)) == s_a0.dvac_n_min   # τ=0.02，第 0 步即截断


def test_dvac_reset_clears_state(monkeypatch):
    """reset() 必须清空跨 episode 的 DVAC 统计量，否则历史污染 τ。"""
    import deploy.lingbot_vla_v2_policy as mod

    class _DummyFT:
        def __init__(self, *args, **kwargs):
            self.org_features = {"actions": ["action"]}

    class _DummyVLA:
        feature_transform = None

    monkeypatch.setattr(mod, "FeatureTransform", _DummyFT)

    s = object.__new__(mod.LingbotVLAv2Server)
    s.use_bf16 = True
    s.use_length = 20
    s.vla = _DummyVLA()
    s.data_config = object()
    s.config = SimpleNamespace(chunk_size=50)
    s.processor = None
    s.robot_norm_path = None
    s.action_key = "action"
    s.global_step = 123
    s.last_action_chunk = {"action": torch.zeros(1, 20, 4)}
    s.last_normalized_action_chunk = None
    s._dvac_buf = [0.01, 0.02, 0.03]
    s._dvac_trace = [(0.01, 0.02, 5)]
    s._dvac_var = torch.zeros(1, 50)
    s._exec_len = 7

    s.reset(robo_name="robotwin")

    assert s._dvac_buf == []
    assert s._dvac_trace == []
    assert s._dvac_var is None
    assert s._exec_len == s.use_length == 20
    assert s.global_step == 0


def test_dvac_rejects_chunk_ret_false():
    """dvac_alpha>0 + chunk_ret=False 会错位（_eff_len 被改写），必须直接报错。"""
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    with pytest.raises(ValueError):
        LingbotVLAv2Server(path_to_pi_model="", chunk_ret=False, dvac_alpha=1.0)


# --------------------------------------------------------------------------- #
# 5) 采样接口：DVAC 需要的那两个 kwarg 必须存在
# --------------------------------------------------------------------------- #
def test_sample_actions_has_dvac_kwargs():
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        LingbotVlaV2Policy,
    )

    inner = [m for _, m in LingbotVlaV2Policy.__dict__.items()]
    _ = inner  # policy 只是转发，真正的实现在内部 action expert
    import lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 as mod

    found = False
    for name in dir(mod):
        obj = getattr(mod, name)
        fn = getattr(obj, "sample_actions", None)
        if fn is not None and hasattr(fn, "__code__"):
            code = fn.__code__
            if "return_denoise_var" in code.co_varnames:
                found = True
                sig = inspect.signature(fn)
                assert "denoise_var_tail_steps" in sig.parameters
    assert found, "没有找到带 return_denoise_var 的 sample_actions 实现"
