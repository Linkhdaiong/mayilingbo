#!/usr/bin/env python3
"""验证 adv_weight_path 能走通官方配置链路：yaml -> parse_args -> MyDataArguments。

这是最关键的接线检查：官方解析器用 parse_known_args + remaining_args 报错，
任何未在 dataclass 里声明的 yaml 键都会直接抛 ValueError。
"""
import os
import sys

# arguments.py 的 __post_init__ 会读这些分布式环境变量；测试进程里没有，
# 必须显式给上，否则 parse_args 直接 TypeError。
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tasks", "vla"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "train_lingbotvla_mod", os.path.join(ROOT, "tasks", "vla", "train_lingbotvla.py")
)
mod = importlib.util.module_from_spec(spec)
sys.modules["train_lingbotvla_mod"] = mod
spec.loader.exec_module(mod)

from lingbotvla.utils.arguments import parse_args  # noqa: E402

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name} {extra}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


yaml_path = os.path.join(ROOT, "configs", "vla", "robotwin", "robotwin.yaml")

print("=" * 76)
print("A. 官方解析器能接受含 adv_weight_path 的 yaml")
print("=" * 76)
sys.argv = [
    "train_lingbotvla.py", yaml_path,
    "--train.global_batch_size=32", "--train.micro_batch_size=32",
]
try:
    args = parse_args(mod.Arguments)
    check("parse_args 未抛异常", True)
    check("args.data 有 adv_weight_path 字段", hasattr(args.data, "adv_weight_path"))
    check("默认关闭（None = 纯 BC）",
          args.data.adv_weight_path is None,
          f"= {args.data.adv_weight_path!r}")
    # 回归护栏：绝不能是字符串 'null' —— 那会让 `if adv_weight_path:` 为真，
    # 训练时去找名为 "null" 的文件并崩溃。
    check("不是字符串 'null'（已知的 yaml null 陷阱）",
          args.data.adv_weight_path != "null")
except ValueError as e:
    check("parse_args 未抛异常", False, f"ValueError: {str(e)[:300]}")
    raise SystemExit(1)

print()
print("=" * 76)
print("B. 命令行覆盖 yaml（训练时实际用法）")
print("=" * 76)
sys.argv = [
    "train_lingbotvla.py", yaml_path,
    "--train.global_batch_size=32", "--train.micro_batch_size=32",
    "--data.adv_weight_path=/tmp/adv_weights.pt",
]
try:
    args2 = parse_args(mod.Arguments)
    check("命令行可覆盖 adv_weight_path",
          args2.data.adv_weight_path == "/tmp/adv_weights.pt",
          f"= {args2.data.adv_weight_path!r}")
except ValueError as e:
    check("命令行可覆盖 adv_weight_path", False, f"ValueError: {str(e)[:300]}")

print()
print("=" * 76)
print("C. 回归：去掉该字段的 yaml 仍能解析（确认没破坏原有字段）")
print("=" * 76)
import tempfile  # noqa: E402

import yaml  # noqa: E402

with open(yaml_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["data"].pop("adv_weight_path", None)
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
    yaml.safe_dump(cfg, tf, allow_unicode=True)
    tmp_yaml = tf.name
sys.argv = [
    "train_lingbotvla.py", tmp_yaml,
    "--train.global_batch_size=32", "--train.micro_batch_size=32",
]
try:
    args3 = parse_args(mod.Arguments)
    check("无该字段的 yaml 仍可解析", True)
    check("缺省时 adv_weight_path 为 None", args3.data.adv_weight_path is None)
except ValueError as e:
    check("无该字段的 yaml 仍可解析", False, f"ValueError: {str(e)[:300]}")
os.unlink(tmp_yaml)

print()
print("=" * 76)
print(f"结果: {PASS} 通过, {FAIL} 失败")
print("=" * 76)
sys.exit(1 if FAIL else 0)
