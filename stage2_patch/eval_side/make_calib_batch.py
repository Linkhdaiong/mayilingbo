#!/usr/bin/env python3
"""构造「少 episode 的配对评测批次」——用于 checkpoint 选型与 DVAC A/B。

为什么要这个脚本
----------------
官方 `eval_matrix.py init` 是给正式 10k 评测用的：它**写死** 50 任务 / 100 episodes /
且只接受 `global_step_45000/hf_ckpt`。而冲刺期需要的是：
  * 任意 checkpoint（含 EMA 导出的目录）；
  * 每任务 1~2 个 episode 的快速配对比较；
  * 与 `eval_parallel.py` / `eval_matrix.py report` 完全兼容的批次目录。

本脚本复用官方 `eval_matrix.py` 的 `inputs()`/`jobs()` 口径（直接 import 同目录模块），
只放宽 episode 数与 checkpoint 选择，产出 schema 完全一致的 `manifest.json`。

用法
----
    # 在 02_代码材料/src/eval_orchestration/tools/checkpoint-eval-45000-100/ 下
    python make_calib_batch.py \
        --root /path/cyber \
        --batch /path/cyber/eval/calib_step45000_base \
        --model /path/cyber/models/hf/step45000 --step 45000 \
        --episodes 1 --tasks all --note "baseline alpha=0"

    # 然后（与正式评测同一条工具链）：
    python ../eval_parallel/eval_parallel.py validate --batch <batch目录>
    python ../eval_parallel/eval_parallel.py run --batch <batch目录> --concurrency 8
    python scripts/calibrate.py --run base=<batch目录> --run cand=<另一批次> --json r.json

注意：**一个批次只放一个 checkpoint**。这样每个批次目录就是一个「run」，可以直接喂给
`scripts/calibrate.py --run name=<dir>` 做配对比较；DVAC 开关不同的两次评测也应各自
成为独立批次（因为开关通过环境变量 `DVAC_ARGS` 传给 eval.sh，不写进 manifest）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _import_eval_matrix():
    """定位官方 eval_matrix.py：优先同目录，其次 tools/checkpoint-eval-45000-100。"""
    cands = [BASE, BASE.parent / "checkpoint-eval-45000-100",
             BASE.parent.parent / "checkpoint-eval-45000-100"]
    _env = os.environ.get("EVAL_MATRIX_DIR")
    if _env:
        cands.insert(0, Path(_env))
    for d in cands:
        if (d / "eval_matrix.py").is_file():
            sys.path.insert(0, str(d))
            import importlib
            return importlib.import_module("eval_matrix"), d
    raise SystemExit(
        "找不到官方 eval_matrix.py。请把本脚本放到 "
        "02_代码材料/src/eval_orchestration/tools/checkpoint-eval-45000-100/ 下运行。"
    )


em, _EM_DIR = _import_eval_matrix()


def read_tasks(spec: str) -> list:
    if spec == "all":
        tasks = (_EM_DIR / "tasks50.txt").read_text().split()
    else:
        p = Path(spec)
        if p.is_file():
            tasks = p.read_text().split()
        else:
            tasks = [t.strip() for t in spec.split(",") if t.strip()]
    if not tasks:
        raise SystemExit("任务列表为空")
    if len(set(tasks)) != len(tasks):
        raise SystemExit("任务列表有重复")
    return tasks


def check_checkpoint(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"checkpoint 目录不存在：{path}")
    shards = list(path.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"没有 safetensors 分片：{path}")
    if any(p.stat().st_size == 0 for p in shards):
        raise SystemExit(f"存在空分片：{path}")
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        wm = set(json.loads(idx.read_text())["weight_map"].values())
        missing = [n for n in wm if not (path / n).is_file()]
        if missing:
            raise SystemExit(f"缺少分片：{missing}")
    if not (path / "config.json").is_file():
        raise SystemExit(f"缺少 config.json（deploy 会加载失败）：{path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="构造少 episode 的配对评测批次")
    ap.add_argument("--root", required=True, help="评测根目录（含 eval.sh / repos / models）")
    ap.add_argument("--batch", required=True, help="新批次目录（必须不存在）")
    ap.add_argument("--model", required=True, help="该 checkpoint 的 hf_ckpt 目录")
    ap.add_argument("--step", type=int, required=True, help="批次里显示的 step 标签")
    ap.add_argument("--episodes", type=int, default=1, help="每个 (任务, setting) 的 episode 数")
    ap.add_argument("--tasks", default="all", help="all | 逗号列表 | 任务清单文件")
    ap.add_argument("--job-timeout", type=int, default=7200)
    ap.add_argument("--health-timeout", type=int, default=1200)
    ap.add_argument("--port", type=int, default=9330)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--note", default="", help="写进 manifest 的人类备注（如 alpha=1.0）")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    batch = Path(args.batch).resolve()
    if batch.exists():
        raise SystemExit(f"批次目录已存在，拒绝覆盖：{batch}")

    tasks = read_tasks(args.tasks)
    model = Path(args.model).resolve()
    check_checkpoint(model)

    missing_envs = [t for t in tasks if not (root / "repos/RoboTwin_pi0" / f"envs/{t}.py").is_file()]
    if missing_envs:
        raise SystemExit(f"缺少任务环境文件：{missing_envs}")

    ckpts = [{"step": int(args.step), "path": str(model)}]
    settings = {
        "clean2clean": {"config": "demo_clean", "instruction": "seen"},
        "clean2random": {"config": "demo_randomized", "instruction": "unseen"},
    }
    n_jobs = len(tasks) * len(settings)
    m = {
        "created": em.now(),
        "root": str(root),
        "episodes": int(args.episodes),
        "video": bool(args.video),
        "tasks": tasks,
        "checkpoints": ckpts,
        "jobs": n_jobs,
        "planned_episodes": n_jobs * int(args.episodes),
        "health_timeout": int(args.health_timeout),
        "job_timeout": int(args.job_timeout),
        "port": int(args.port),
        "settings": settings,
        "format_notice": "INTERNAL ONLY; not official submission JSON",
        "purpose": "calibration/selection batch (subset episodes)",
        "note": args.note,
        "dvac_env_hint": "DVAC 通过环境变量 DVAC_ARGS 传给 eval.sh，不在 manifest 内；记录在 note",
        "inputs": em.inputs(root, ckpts),
    }

    batch.mkdir(parents=True)
    em.write(batch / "manifest.json", m)

    snap = batch / "snapshot"
    snap.mkdir()
    for src, name in (
        (root / "eval.sh", "eval.sh"),
        (root / "repos/RoboTwin_pi0/script/eval_policy_client_lingbotvla.py", "client.py"),
        (_EM_DIR / "tasks50.txt", "tasks50.txt"),
    ):
        if src.is_file():
            shutil.copy2(src, snap / name)
    for repo in ("lingbot-vla-v2", "RoboTwin_pi0"):
        for cmd, name in ((['rev-parse', 'HEAD'], 'commit.txt'), (['status', '--short'], 'status.txt')):
            try:
                r = subprocess.run(["git", "-C", str(root / "repos" / repo), *cmd],
                                   capture_output=True, text=True)
                (snap / f"{repo}_{name}").write_text(r.stdout + r.stderr)
            except OSError:
                pass
    (snap / "note.txt").write_text(
        f"purpose=calibration\nstep={args.step}\nmodel={model}\nepisodes={args.episodes}\n"
        f"tasks={len(tasks)}\ndvac={args.note or '<unset>'}\n"
        f"hint: DVAC_ARGS 在 run 时通过环境变量传入，例如 DVAC_ARGS='--dvac_alpha 1.0'\n"
    )

    est_min = n_jobs / max(1, int(os.environ.get("CALIB_CONC", "8"))) * (2.0 + args.episodes * 0.5)
    print("已创建校准批次:", batch)
    print(f"  checkpoint : step{args.step} {model}")
    print(f"  任务/设置   : {len(tasks)} × {len(settings)} = {n_jobs} jobs, {n_jobs*args.episodes} episodes")
    print(f"  预计机时    : ≈{est_min:.0f} min（并发 {os.environ.get('CALIB_CONC','8')}，含每 job 服务启动开销）")
    print("  下一步      : eval_parallel.py validate → run；然后用 scripts/calibrate.py 做配对比较")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
