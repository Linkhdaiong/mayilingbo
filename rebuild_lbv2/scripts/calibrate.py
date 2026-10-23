#!/usr/bin/env python3
"""配对评测汇总 / 选型比较。

用途
----
1. checkpoint 选型：EMA 权重 vs 普通权重 vs 中间 step，谁更适合"提交"；
2. DVAC 标定：baseline(use_length=50) vs use_length=25 vs DVAC(α=1.0)。

判定口径（与官方评分一致）
--------------------------
每个 setting 各 5000 episode，最终分数 = (clean2clean 成功率 + clean2random
成功率) / 2。因此比较两个 run 时同样取两个 setting 的等权平均；显著性用
**逐任务配对差** 的 t 统计量（配对比独立样本量级更省 episode），并给符号检验。

用法
----
    python scripts/calibrate.py \
        --run A=/path/to/eval_run_A \
        --run B=/path/to/eval_run_B \
        [--run C=...] [--json report.json] [--threshold 2.0]

目录要求：每条 episode 的 `jobs/**/outcome.json`（本队 eval_parallel 的产物）。
也支持直接给 `reports/counts_*.json`（此时没有 episode 级配对，只做任务级配对）。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

SETTINGS = ("clean2clean", "clean2random")
# 单侧 95% 分位：标定只有 15 任务 × 2 setting = 30 个配对单元（df=29）→ t=1.699。
# 30 单元、每单元 5 eps 时 SE≈1.8 pt，只看均值会把 +2 pt 的噪声当成胜出，
# 因此判定必须用下限 LB = mean − 1.70×SE（等价要求 mean≥3.1 pt）。
T95 = 1.70


def load_run(path: str) -> Dict[Tuple[str, str, int], int]:
    """返回 {(setting, task, seed): success}。"""
    out: Dict[Tuple[str, str, int], int] = {}
    for fp in glob.glob(os.path.join(path, "jobs", "**", "outcome.json"), recursive=True):
        try:
            rec = json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        job = rec.get("job") or {}
        res = rec.get("result") or {}
        setting = job.get("setting")
        task = job.get("task")
        eps = res.get("episodes") or []
        if not setting or not task or not eps:
            continue
        for e in eps:
            seed = e.get("seed")
            if seed is None:
                continue
            out[(setting, str(task), int(seed))] = int(bool(e.get("success")))
    return out


def load_counts(path: str) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """从 counts_*.json 读 (successes, tests)，键为 (setting, task)。"""
    for fp in [path] + glob.glob(os.path.join(path, "**", "counts_*.json"), recursive=True):
        if not os.path.isfile(fp):
            continue
        d = json.load(open(fp, encoding="utf-8"))
        recs = d.get("records") or []
        out = {}
        for r in recs:
            setting, task = r.get("setting"), r.get("task")
            if not setting or not task:
                continue
            out[(str(setting), str(task))] = (int(r.get("successes", 0)), int(r.get("tests", 0)))
        if out:
            return out
    raise FileNotFoundError(f"没在 {path} 找到可用的评测结果")


def summarize(ep: Dict[Tuple[str, str, int], int]) -> Dict[str, dict]:
    per = {}
    for setting in SETTINGS:
        tasks = defaultdict(lambda: [0, 0])
        for (s, t, _seed), ok in ep.items():
            if s != setting:
                continue
            tasks[t][0] += ok
            tasks[t][1] += 1
        total_ok = sum(v[0] for v in tasks.values())
        total_n = sum(v[1] for v in tasks.values())
        per[setting] = {
            "successes": total_ok,
            "tests": total_n,
            "rate": (total_ok / total_n * 100.0) if total_n else float("nan"),
            "tasks": {k: {"successes": v[0], "tests": v[1]} for k, v in tasks.items()},
        }
    rates = [per[s]["rate"] for s in SETTINGS if not math.isnan(per[s]["rate"])]
    per["mean"] = {"rate": sum(rates) / len(rates) if rates else float("nan")}
    return per


def paired_compare(a: Dict[Tuple[str, str, int], int], b: Dict[Tuple[str, str, int], int]):
    """逐任务配对比较（同 setting 同任务内先各自算成功率，再求差）。"""
    keys = set(a) & set(b)
    if not keys:
        return None
    buckets = defaultdict(lambda: {"a": [0, 0], "b": [0, 0]})
    for (setting, task, _seed) in keys:
        buckets[(setting, task)]["a"][0] += a[(setting, task, _seed)]
        buckets[(setting, task)]["a"][1] += 1
        buckets[(setting, task)]["b"][0] += b[(setting, task, _seed)]
        buckets[(setting, task)]["b"][1] += 1

    diffs = []
    rows = []
    for (setting, task), v in sorted(buckets.items()):
        ra = v["a"][0] / max(v["a"][1], 1) * 100.0
        rb = v["b"][0] / max(v["b"][1], 1) * 100.0
        diffs.append(rb - ra)
        rows.append({"setting": setting, "task": task, "a": ra, "b": rb, "delta": rb - ra})

    n = len(diffs)
    mean_d = sum(diffs) / n
    sd = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1))
    se = sd / math.sqrt(n) if n else float("inf")
    pos = sum(1 for d in diffs if d > 0.5)
    neg = sum(1 for d in diffs if d < -0.5)
    return {
        "paired_units": n,
        "mean_delta_pt": mean_d,
        "se_pt": se,
        "t": (mean_d / se) if se > 0 else float("inf"),
        "lb_pt": (mean_d - T95 * se) if se != float("inf") else float("-inf"),
        "win_tasks": pos,
        "lose_tasks": neg,
        "tie_tasks": n - pos - neg,
        "rows": rows,
    }


def load_any(path: str):
    ep = load_run(path)
    if ep:
        return ep, True
    return {}, False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="配对比较两个评测 run")
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR",
                    help="可重复；第一个视为基线")
    ap.add_argument("--json", default=None, help="把完整报告写成 JSON")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="实际意义门槛（pt）。主判据 = 单侧 95%% 下限 LB>0 且 mean_delta ≥ 该门槛")
    args = ap.parse_args(argv)

    runs: List[Tuple[str, str]] = []
    for spec in args.run:
        if "=" not in spec:
            ap.error(f"--run 需要用 NAME=DIR 形式，收到 {spec!r}")
        name, path = spec.split("=", 1)
        runs.append((name, path))

    episodes: Dict[str, Dict[Tuple[str, str, int], int]] = {}
    counts_only: Dict[str, Dict[Tuple[str, str], Tuple[int, int]]] = {}
    for name, path in runs:
        ep, ok = load_any(path)
        if ok:
            episodes[name] = ep
        else:
            counts_only[name] = load_counts(path)
            print(f"[warn] {name} 没有 episode 级 outcome.json，退化为任务级计数比较")

    report = {"runs": {}, "comparisons": []}

    print("=" * 78)
    print(f"{'run':<28}{'clean2clean':>14}{'clean2random':>15}{'mean':>10}")
    print("-" * 78)
    for name, _ in runs:
        if name in episodes:
            s = summarize(episodes[name])
            print(f"{name:<28}{s['clean2clean']['rate']:>13.2f}%{s['clean2random']['rate']:>14.2f}%{s['mean']['rate']:>9.2f}")
            report["runs"][name] = {k: {kk: vv for kk, vv in v.items() if kk != "tasks"}
                                    for k, v in s.items()}
        else:
            c = counts_only[name]
            per = {}
            for setting in SETTINGS:
                ok = sum(v[0] for k, v in c.items() if k[0] == setting)
                n = sum(v[1] for k, v in c.items() if k[0] == setting)
                per[setting] = (ok, n, ok / max(n, 1) * 100.0)
            mean = sum(per[s][2] for s in SETTINGS) / len(SETTINGS)
            print(f"{name:<28}{per['clean2clean'][2]:>13.2f}%{per['clean2random'][2]:>14.2f}%{mean:>9.2f}")
            report["runs"][name] = {"mean": {"rate": mean}}
    print("=" * 78)

    base_name = runs[0][0]
    if base_name not in episodes:
        print("基线没有 episode 级数据，跳过配对检验。")
        return 0

    for name, _ in runs[1:]:
        if name not in episodes:
            print(f"\n[{name}] 无 episode 级数据，跳过配对检验。")
            continue
        cmp_ = paired_compare(episodes[base_name], episodes[name])
        if cmp_ is None:
            print(f"\n[{name}] 与基线没有共同 episode（seed/任务不一致），无法配对。")
            continue
        lb = cmp_["lb_pt"]
        if lb > 0 and cmp_["mean_delta_pt"] >= args.threshold:
            verdict = "胜出"
        elif cmp_["mean_delta_pt"] <= -args.threshold:
            verdict = "失败"
        else:
            verdict = "无显著差异"
        print(f"\n[{base_name} → {name}]  配对单元 {cmp_['paired_units']}（setting×task）")
        print(f"  平均提升 {cmp_['mean_delta_pt']:+.2f} pt  (SE {cmp_['se_pt']:.2f}, t={cmp_['t']:.2f})")
        print(f"  单侧 95% 下限 LB {lb:+.2f} pt  (= mean − {T95:.2f}×SE；单元越少越保守)")
        print(f"  任务胜负 {cmp_['win_tasks']} 胜 / {cmp_['lose_tasks']} 负 / {cmp_['tie_tasks']} 平"
              f"（仅展示：单任务分辨率 = 100/每任务 eps 数 pt，5 eps 时为 20 pt，不构成判据）")
        print(f"  判定：{verdict}（主判据 LB>0 且 mean ≥ {args.threshold} pt）")
        worst = sorted(cmp_["rows"], key=lambda r: r["delta"])[:5]
        best = sorted(cmp_["rows"], key=lambda r: -r["delta"])[:5]
        print("  掉点最多：" + ", ".join(f"{r['setting']}/{r['task']} {r['delta']:+.0f}" for r in worst))
        print("  涨点最多：" + ", ".join(f"{r['setting']}/{r['task']} {r['delta']:+.0f}" for r in best))
        report["comparisons"].append({"base": base_name, "cand": name, "verdict": verdict, **cmp_})

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已写出：{args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
