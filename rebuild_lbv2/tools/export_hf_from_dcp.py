#!/usr/bin/env python3
"""从 DCP checkpoint 导出 HuggingFace 权重（可选导出 EMA 分支）。

为什么需要它
------------
stage-1 的每一次 DCP 保存都会顺带导出 `hf_ckpt/`（25.5 GB，fp32）。stage-2 如果
每 2000 步保存一次，同时导出 `hf_ckpt` + `ema_hf_ckpt`，磁盘占用会到 600 GB 量级。
因此 stage-2 默认**只存 DCP**（`--train.save_hf_weights=false`），训练结束后用本脚本
按需把「某个 step 的普通权重 / EMA 权重」导出成可评测的 hf_ckpt。

它走的是官方同一条通路（`ckpt_to_state_dict` + `save_model_weights`），不引入新的
权重格式。

用法
----
    # 导出第 24000 步的普通权重
    python tools/export_hf_from_dcp.py \
        --dcp /out/checkpoints/global_step_24000 \
        --assets-from /models/lingbot_hf_45000/hf_ckpt \
        --out /out/hf/step_24000_plain

    # 导出同一步的 EMA 权重
    python tools/export_hf_from_dcp.py \
        --dcp /out/checkpoints/global_step_24000 --ema \
        --assets-from /models/lingbot_hf_45000/hf_ckpt \
        --out /out/hf/step_24000_ema

`--assets-from` 指向任意一份已有的 hf_ckpt（含 config.json / tokenizer 等），
本脚本只替换里面的权重分片，其余文件原样拷贝。
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lingbotvla.checkpoint.format_utils import ckpt_to_state_dict  # noqa: E402
from lingbotvla.models.module_utils import save_model_weights  # noqa: E402

NON_WEIGHT_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.py",
    "*.jinja",
    "*.tiktoken",
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="导出 DCP（可选 EMA）为 hf_ckpt")
    p.add_argument("--dcp", required=True, help="checkpoints/global_step_N 目录")
    p.add_argument("--out", required=True, help="输出 hf_ckpt 目录（不得已存在）")
    p.add_argument("--assets-from", required=True, help="已有 hf_ckpt 目录，用于拷贝 config/tokenizer")
    p.add_argument("--ema", action="store_true", help="导出 ema/ 分支（需训练时开了 EMA）")
    p.add_argument("--ckpt-manager", default="dcp", choices=["dcp", "bytecheckpoint", "native"])
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="导出精度；官方提交用 F32（≈25.5 GB）")
    p.add_argument("--shard-size", type=int, default=5_000_000_000)
    args = p.parse_args(argv)

    if os.path.exists(args.out):
        print(f"输出目录已存在，拒绝覆盖：{args.out}", file=sys.stderr)
        return 2
    if args.ema:
        ema_dir = os.path.join(args.dcp, "ema")
        if not os.path.isdir(ema_dir):
            print(f"未找到 EMA 分支：{ema_dir}\n"
                  f"（训练时未开 --train.ema_decay，或该 step 保存时 EMA 尚未创建）",
                  file=sys.stderr)
            return 3

    tmp = args.out + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    print(f"[export] DCP={args.dcp} ema={args.ema} dtype={args.dtype}")
    state_dict = ckpt_to_state_dict(
        save_checkpoint_path=args.dcp,
        output_dir=tmp,
        ckpt_manager=args.ckpt_manager,
        ema=args.ema,
    )
    print(f"[export] 取到 {len(state_dict)} 个张量")

    save_model_weights(
        tmp,
        state_dict,
        save_dtype=getattr(torch, args.dtype),
        shard_size=args.shard_size,
    )
    del state_dict

    # 拷贝 config / tokenizer 等非权重文件
    n_copied = 0
    for pattern in NON_WEIGHT_PATTERNS:
        for src in glob.glob(os.path.join(args.assets_from, pattern)):
            dst = os.path.join(tmp, os.path.basename(src))
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                n_copied += 1
    print(f"[export] 拷贝非权重文件 {n_copied} 个")

    os.replace(tmp, args.out)
    total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(args.out, "*.safetensors")))
    n_shard = len(glob.glob(os.path.join(args.out, "*.safetensors")))
    print(f"[export] 完成：{args.out}（{n_shard} 分片，{total/1e9:.2f} GB）")
    print("[export] 下次启动 deploy 前先跑 verify_hf_ckpt.sh 做静态校验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
