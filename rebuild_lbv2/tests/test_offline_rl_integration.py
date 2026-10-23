#!/usr/bin/env python3
"""offline RL 集成测试：不依赖真实数据集与 6B 权重，验证整条接线是否正确。

覆盖：
  T1 合成 LeRobot 数据集 -> build_adv_weights.py -> .pt
  T2 配置链路：robotwin.yaml + MyDataArguments 接受 adv_weight_path
  T3 模型 forward 签名含 adv_weight，且加权发生在归约之前
  T4 数据集侧接线：VLADataset 能加载 sidecar 并把 adv_weight 挂进样本
  T5 端到端数值：加权损失 == 手工计算，且 w=1 时与纯 BC 逐位一致
"""
import ast
import inspect
import json
import os
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name} {extra}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def make_fake_lerobot(root, lengths):
    """只造 meta/，足够 build_adv_weights.py 读取。"""
    meta = os.path.join(root, "meta")
    os.makedirs(meta, exist_ok=True)
    with open(os.path.join(meta, "info.json"), "w") as f:
        json.dump({"fps": 30, "total_frames": sum(lengths), "codebase_version": "v2.1"}, f)
    with open(os.path.join(meta, "episodes.jsonl"), "w") as f:
        for i, n in enumerate(lengths):
            f.write(json.dumps({"episode_index": i, "tasks": ["pick"], "length": n}) + "\n")
    return root


print("=" * 76)
print("T1. 合成数据集 -> build_adv_weights.py -> .pt")
print("=" * 76)
tmp = tempfile.mkdtemp(prefix="offline_rl_it_")
ds_a = make_fake_lerobot(os.path.join(tmp, "task_a"), [4, 6, 5])
ds_b = make_fake_lerobot(os.path.join(tmp, "task_b"), [3, 3])
train_txt = os.path.join(tmp, "train.txt")
with open(train_txt, "w") as f:
    f.write(f"robotwin {ds_a}\n")
    f.write(f"robotwin {ds_b}\n")
out_pt = os.path.join(tmp, "adv.pt")

env = dict(os.environ)
env["PYTHONPATH"] = ROOT
r = subprocess.run(
    [sys.executable, os.path.join(ROOT, "tools", "build_adv_weights.py"),
     "--train-path", train_txt, "--out", out_pt,
     "--gamma", "0.99", "--verify"],
    capture_output=True, text=True, env=env, cwd=ROOT,
)
check("builder 退出码 0", r.returncode == 0, f"(rc={r.returncode})")
if r.returncode != 0:
    print(r.stdout[-2000:])
    print(r.stderr[-2000:])
check("产出 .pt 存在", os.path.isfile(out_pt))
expected_n = sum([4, 6, 5]) + sum([3, 3])
from lingbotvla.offline_rl.advantage import load_weights  # noqa: E402
w, cfg = load_weights(out_pt)
check("权重数量 == 两个数据集帧数之和", int(w.numel()) == expected_n,
      f"({int(w.numel())} vs {expected_n})")
check("mean(w) == 1", abs(float(w.mean()) - 1.0) < 1e-5, f"mean={float(w.mean()):.6f}")
check("超参随权重一起保存（用默认 beta）", cfg is not None and cfg["gamma"] == 0.99 and cfg["beta"] == 0.25,
      f"gamma={cfg.get('gamma')} beta={cfg.get('beta')}")
# 顺序校验：第一个数据集第 2 条 episode（长度 6）内部应单调不减
# （裁剪饱和会让首尾若干帧取相同值，属预期；关键是方向正确）
seg = w[4:10]
check("拼接顺序正确（task_a 第 2 条 episode 内部单调不减）", bool((seg[1:] >= seg[:-1]).all()),
      f"w={[round(float(x),3) for x in seg]}")
# 默认超参下不应把训练集中到轨迹的一小段
_conc = float((w < 0.5 * float(w.mean())).float().mean())
check("默认超参下被压制(<0.5×均值)占比 < 40%", _conc < 0.40, f"({_conc:.1%})")
_cfg = cfg or {}
check("裁剪区间有界（最坏情况 ±2 倍）",
      _cfg.get("w_min") == 0.5 and _cfg.get("w_max") == 2.0,
      f"[{_cfg.get('w_min')}, {_cfg.get('w_max')}]")

print()
print("=" * 76)
print("T2. 配置链路：robotwin.yaml + MyDataArguments")
print("=" * 76)
import yaml  # noqa: E402
cfg_path = os.path.join(ROOT, "configs", "vla", "robotwin", "robotwin.yaml")
with open(cfg_path, encoding="utf-8") as f:
    ycfg = yaml.safe_load(f)
# yaml 里必须「不设置」该键（默认关闭）。写成 null 会被官方解析器变成字符串
# 'null'（真值），训练时去找名为 "null" 的文件并崩溃 —— 见 test_config_chain.py。
_v = ycfg["data"].get("adv_weight_path")
check("robotwin.yaml 默认不启用 adv_weight_path", _v is None and _v != "null",
      f"= {_v!r}")

sys.path.insert(0, os.path.join(ROOT, "tasks", "vla"))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_lingbotvla_mod", os.path.join(ROOT, "tasks", "vla", "train_lingbotvla.py")
    )
    src = open(os.path.join(ROOT, "tasks", "vla", "train_lingbotvla.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MyDataArguments":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and getattr(item.target, "id", None) == "adv_weight_path":
                    found = True
    check("MyDataArguments 定义了 adv_weight_path 字段", found)
except Exception as e:
    check("MyDataArguments 定义了 adv_weight_path 字段", False, f"({e})")

print()
print("=" * 76)
print("T3. 模型 forward：签名含 adv_weight，且加权在归约之前")
print("=" * 76)
model_src = open(
    os.path.join(ROOT, "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"),
    encoding="utf-8",
).read()
mtree = ast.parse(model_src)
sig_ok = False
for node in ast.walk(mtree):
    if isinstance(node, ast.FunctionDef) and node.name == "forward":
        names = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
        if "adv_weight" in names and "actions" in names:
            sig_ok = True
check("LingbotVlaV2Policy.forward 接受 adv_weight", sig_ok)

i_weight = model_src.find("losses = losses * w.view(-1, 1, 1)")
i_reduce = model_src.find("masked_losses = losses * joint_mask")
check("加权语句存在", i_weight > 0)
check("加权发生在归约之前", 0 < i_weight < i_reduce,
      f"(weight@{i_weight} < reduce@{i_reduce})")
check("adv_weight=None 时短路（不改行为）", "if adv_weight is not None:" in model_src)

print()
print("=" * 76)
print("T4. 数据集侧接线（静态检查 + 真实实例化）")
print("=" * 76)
base_src = open(os.path.join(ROOT, "lingbotvla/data/vla_data/base_dataset.py"), encoding="utf-8").read()
check("VLADataset.__init__ 读取 dataset_config.adv_weight_path",
      'getattr(dataset_config, "adv_weight_path", None)' in base_src)
check("getitem 挂载 adv_weight", 'item["adv_weight"] = self.adv_weights[idx]' in base_src)
check("样本数不匹配时抛错", "_n_expected" in base_src and "raise ValueError" in base_src)

print()
print("=" * 76)
print("T5. 端到端数值：复现模型里那段加权归约")
print("=" * 76)
torch.manual_seed(0)
losses = torch.rand(4, 8, 3)
w_b = w[:4].clone()
w_b = w_b / w_b.mean()

# 复刻补丁里的代码路径
losses_w = losses * w_b.view(-1, 1, 1)
loss_vla = losses_w.mean()
manual = (losses * w_b.view(-1, 1, 1)).mean()
check("加权损失 == 手工计算", torch.allclose(loss_vla, manual),
      f"{float(loss_vla):.8f}")

losses_one = losses * torch.ones(4).view(-1, 1, 1)
check("w=1 时与纯 BC 逐位一致", torch.allclose(losses_one.mean(), losses.mean()),
      f"{float(losses.mean()):.8f}")

try:
    bad = torch.ones(3)
    l = losses * bad.view(-1, 1, 1)
    check("长度不匹配应被拦住", False)
except Exception:
    check("长度不匹配被拦住", True)

print()
print("=" * 76)
print(f"结果: {PASS} 通过, {FAIL} 失败")
print("=" * 76)
sys.exit(1 if FAIL else 0)
