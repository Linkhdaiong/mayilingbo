#!/usr/bin/env bash
# ============================================================================
# 灵波具身大模型挑战赛 · 一次性最终微调（终局冲刺）
#
# 起点 : 已提交权重 global_step_45000/hf_ckpt（F32，53.90%）
# 目标 : 只启动这一轮微调，在同一份 release 数据上提高 clean2random 并整体提分
#
# 与 stage-1 的差别（每一项都有明确依据，见 灵波挑战赛_终局冲刺方案.md）：
#   1. 任务均衡采样        data.task_balanced=true
#   2. 增广扩到五轴近似    data.image_augment=true + LINGBOT_AUG_CFG（代码内默认已开）
#   3. 低运动窗口降权      data.adv_weight_path=<motion_weights 目录>
#   4. 训练期 EMA          train.ema_decay=0.999（官方 DCP/HF 通路自动导出 ema_hf_ckpt）
#   5. 温和重启 lr 2e-5 → 2e-6 cosine，8 卡全局 batch 64
#   6. 同一次运行完成 25000 个 optimizer step；旧单卡吞吐不能推算 8 卡耗时。
#
# 路径配置（换机必读）：
#   本脚本从 configs/local_paths.sh 读取 model_path / train_path / output_dir /
#   tokenizer_path / robot_config_root 和教师权重路径，
#   并在启动前把 4 个教师权重路径写进 yaml。官方参数解析器只支持整块替换 align_params，
#   不支持点路径覆盖，所以这里直接改文件而不是传命令行：
#     MOGE_PATH        -> align_params.depth.moge_path
#     MORGBD_PATH      -> align_params.depth.morgbd_path
#     DINO_VIDEO_CKPT  -> align_params.video.ckpt_path
#     DINO_VIDEO_CFG   -> align_params.video.config_path
#   四个缺一不可：align_params 非空时训练会 from_pretrained 它们，占位符 /path/to 会在加载时报错。
#   robot_config_root 由 ROBOT_CONFIG_ROOT 覆盖，默认指向仓库内 configs/robot_configs。
#
# 磁盘策略：默认只存 DCP（save_hf_weights=false）。训练结束或中途需要评测时，
# 用 tools/export_hf_from_dcp.py 按需导出 plain / EMA 权重为 hf_ckpt。
#
# 用法：
#   BASE_HF_CKPT=/path/global_step_45000/hf_ckpt \
#   DATA_MANIFEST=/path/clean_training_data.txt \
#   TOKENIZER_PATH=/path/Qwen3-VL-4B-Instruct \
#   MOGE_PATH=/path/moge2-vitb-normal.pt \
#   MORGBD_PATH=/path/depth/model.pt \
#   DINO_VIDEO_CKPT=/path/dino_video/teacher_step_10000.pth \
#   DINO_VIDEO_CFG=/path/dino_video/config.yaml \
#   ROBOT_CONFIG_ROOT=/path/robot_configs \
#   MOTION_WEIGHTS=/path/motion_weights_clean50 \
#   OUT_DIR=/out/stage2  bash scripts/train_stage2.sh
#   （MOTION_WEIGHTS 是**目录**，目录内含 <data_name>.pt）
# ============================================================================
set -euo pipefail

V2_ROOT=${V2_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PATH_CONFIG=${PATH_CONFIG:-$V2_ROOT/configs/local_paths.sh}
if [ ! -f "$PATH_CONFIG" ]; then
  echo "[final] 路径配置文件不存在：$PATH_CONFIG" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$PATH_CONFIG"
TRAIN_ENV=${TRAIN_ENV:?需要设置 TRAIN_ENV（训练环境的绝对路径）}
if [ ! -x "$TRAIN_ENV/bin/python" ] || [ ! -x "$TRAIN_ENV/bin/torchrun" ]; then
  echo "[final] TRAIN_ENV 必须包含 bin/python 和 bin/torchrun" >&2
  exit 2
fi
export PATH="$TRAIN_ENV/bin:$PATH"
OUT_DIR=${OUT_DIR:?需要设置 OUT_DIR（训练输出目录，不得已存在）}
LOG_DIR=${LOG_DIR:-${OUT_DIR}.logs}
BASE_HF_CKPT=${BASE_HF_CKPT:?需要设置 BASE_HF_CKPT（stage-1 的 hf_ckpt 目录）}
DATA_MANIFEST=${DATA_MANIFEST:?需要设置 DATA_MANIFEST（release 训练清单）}
TOKENIZER_PATH=${TOKENIZER_PATH:?需要设置 TOKENIZER_PATH（本地 Qwen3-VL 目录）}
ROBOT_CONFIG_ROOT=${ROBOT_CONFIG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/robot_configs}
MOGE_PATH=${MOGE_PATH:?需要设置 MOGE_PATH（moge2-vitb-normal.pt）}
MORGBD_PATH=${MORGBD_PATH:?需要设置 MORGBD_PATH（LingBot-Depth 权重）}
DINO_VIDEO_CKPT=${DINO_VIDEO_CKPT:?需要设置 DINO_VIDEO_CKPT（teacher_step_10000.pth）}
DINO_VIDEO_CFG=${DINO_VIDEO_CFG:?需要设置 DINO_VIDEO_CFG（dino_video 的 config.yaml）}
MOTION_WEIGHTS=${MOTION_WEIGHTS:?需要设置 MOTION_WEIGHTS（运动权重目录）}

MAX_STEPS=${MAX_STEPS:-25000}
# 5000 而非 2500：单份 DCP ≈63.8 GB（fp32 模型 25.5 + EMA bf16 12.75 + Muon 动量 fp32 25.5），
# 2500 步要存 10 份 ≈638 GB，超出现有磁盘预算；5000 步存 5 份 ≈319 GB，且 20k 仍在保存点上。
SAVE_STEPS=${SAVE_STEPS:-5000}
# 显存不足时的降级阶梯（**保持全局 batch = 64**，只换组合）：
#   每卡 2 × 8 卡 × GA 4 = 64；显存不足时改为每卡 1、GA 8，保持全局 batch 64。
MICRO_BS=${MICRO_BS:-2}
GA=${GA:-4}
LR=${LR:-2e-5}
LR_MIN=${LR_MIN:-2e-6}
EMA_DECAY=${EMA_DECAY:-0.999}
SAVE_HF=${SAVE_HF:-false}

cd "$V2_ROOT"

if [ -n "${GPU_IDS:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$GPU_IDS" != "$CUDA_VISIBLE_DEVICES" ]; then
  echo "[final] GPU_IDS 与 CUDA_VISIBLE_DEVICES 不一致" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU_IDS:-0,1,2,3,4,5,6,7}}
IFS=',' read -r -a _visible_gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [ -n "${NPROC_PER_NODE:-}" ] && [ "$NPROC_PER_NODE" != "${#_visible_gpu_ids[@]}" ]; then
  echo "[final] NPROC_PER_NODE 与可见 GPU 数不一致" >&2
  exit 2
fi
NPROC_PER_NODE=${#_visible_gpu_ids[@]}
if [ "$NPROC_PER_NODE" -ne 8 ]; then
  echo "[final] 目标机器要求 8 卡，当前可见进程数为 $NPROC_PER_NODE；请检查 CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
GLOBAL_BS=${GLOBAL_BS:-$((MICRO_BS * GA * NPROC_PER_NODE))}
EXPECTED_GLOBAL_BS=$((MICRO_BS * GA * NPROC_PER_NODE))
if [ "$GLOBAL_BS" -ne "$EXPECTED_GLOBAL_BS" ]; then
  echo "[final] global batch 错误：$GLOBAL_BS != $MICRO_BS x $NPROC_PER_NODE x $GA = $EXPECTED_GLOBAL_BS" >&2
  exit 2
fi

if [ -e "$OUT_DIR" ]; then
  echo "[stage2] 输出目录已存在，拒绝启动以免覆盖：$OUT_DIR" >&2
  exit 2
fi
if [ ! -d "$BASE_HF_CKPT" ]; then
  echo "[stage2] BASE_HF_CKPT 不存在：$BASE_HF_CKPT" >&2
  exit 2
fi
missing=0
for f in "$MOGE_PATH" "$MORGBD_PATH" "$DINO_VIDEO_CKPT" "$DINO_VIDEO_CFG"; do
  if [ ! -f "$f" ]; then echo "[stage2] 教师权重不存在：$f" >&2; missing=1; fi
done
if [ "$missing" = 1 ]; then
  echo "[stage2] 四个教师权重缺一不可（来源见 REQUIREMENTS.md），拒绝启动" >&2
  exit 2
fi
BASE_YAML=${BASE_YAML:-configs/vla/robotwin/robotwin.yaml}
if [[ "$BASE_YAML" != /* ]]; then BASE_YAML="$V2_ROOT/$BASE_YAML"; fi
for f in "$DATA_MANIFEST" "$BASE_YAML"; do
  [ -f "$f" ] || { echo "[final] 文件不存在：$f" >&2; exit 2; }
done
for d in "$TOKENIZER_PATH" "$ROBOT_CONFIG_ROOT" "$MOTION_WEIGHTS"; do
  [ -d "$d" ] || { echo "[final] 目录不存在：$d" >&2; exit 2; }
done
# 官方解析器无法按点路径覆盖 align_params，启动前把 4 个路径写进 yaml 副本再训练
RUN_YAML="$LOG_DIR/robotwin.run.yaml"
mkdir -p "$LOG_DIR"
"$TRAIN_ENV/bin/python" - "$BASE_YAML" "$RUN_YAML" "$MOGE_PATH" "$MORGBD_PATH" "$DINO_VIDEO_CKPT" "$DINO_VIDEO_CFG" <<'PY'
import sys, yaml
src, dst, moge, morgbd, ckpt, cfg = sys.argv[1:]
doc = yaml.safe_load(open(src, encoding="utf-8"))
depth, video = doc["train"]["align_params"]["depth"], doc["train"]["align_params"]["video"]
depth["moge_path"], depth["morgbd_path"] = moge, morgbd
video["ckpt_path"], video["config_path"] = ckpt, cfg
yaml.safe_dump(doc, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
PY
echo "[stage2] 已写入教师权重路径 -> $RUN_YAML"
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  echo "[final] 配置检查通过：$NPROC_PER_NODE ranks，batch=$GLOBAL_BS；未启动训练。"
  exit 0
fi
mkdir -p "$OUT_DIR"

export PATH="$TRAIN_ENV/bin:$PATH"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE - 1)))
  export CUDA_VISIBLE_DEVICES
fi
export MASTER_ADDR=127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1 DISABLE_TELEMETRY=1

EXTRA=()
if [ -n "$DATA_MANIFEST" ]; then EXTRA+=("--data.train_path=$DATA_MANIFEST"); fi
if [ -n "$TOKENIZER_PATH" ]; then EXTRA+=("--model.tokenizer_path=$TOKENIZER_PATH"); fi
if [ -n "$ROBOT_CONFIG_ROOT" ]; then EXTRA+=("--data.robot_config_root=$ROBOT_CONFIG_ROOT"); fi
if [ -n "$MOTION_WEIGHTS" ]; then
  EXTRA+=("--data.adv_weight_path=$MOTION_WEIGHTS")
else
  echo "[stage2] 警告：未提供 MOTION_WEIGHTS（应为目录），低运动降权不会生效（其余项照常）"
fi

echo "================ 一次性最终微调配置 ================"
echo "BASE_HF_CKPT  = $BASE_HF_CKPT"
echo "OUT_DIR       = $OUT_DIR"
echo "MAX_STEPS     = $MAX_STEPS   SAVE_STEPS = $SAVE_STEPS"
echo "batch         = $MICRO_BS x $NPROC_PER_NODE x $GA = $GLOBAL_BS"
echo "lr            = $LR -> $LR_MIN (cosine)"
echo "ema_decay     = $EMA_DECAY   save_hf_weights = $SAVE_HF"
echo "tokenizer     = ${TOKENIZER_PATH:-<yaml 里的占位符>}"
echo "motion weights= ${MOTION_WEIGHTS:-<none>}  (目录形式)"
echo "=============================================="

set +e
torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC_PER_NODE" \
  tasks/vla/train_lingbotvla.py "$RUN_YAML" \
  --model.model_path="$BASE_HF_CKPT" \
  --train.output_dir="$OUT_DIR" \
  --train.max_steps="$MAX_STEPS" \
  --train.save_steps="$SAVE_STEPS" --train.save_epochs=0 \
  --train.micro_batch_size="$MICRO_BS" \
  --train.gradient_accumulation_steps="$GA" \
  --train.global_batch_size="$GLOBAL_BS" \
  --train.lr="$LR" --train.lr_min="$LR_MIN" \
  --train.lr_decay_style=cosine --train.lr_warmup_ratio=0.01 \
  --train.enable_gradient_checkpointing=true \
  --train.enable_fp32=true \
  --train.enable_full_shard=true \
  --train.use_compile=true \
  --train.ema_decay="$EMA_DECAY" --train.ema_dtype=bfloat16 \
  --train.save_hf_weights="$SAVE_HF" \
  --train.async_save_hf_weights=false \
  --train.enable_resume=false \
  --train.enable_profiling=false \
  --data.image_augment=true \
  --data.task_balanced=true \
  --eval.enabled=false \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG_DIR/train.log"
rc=${PIPESTATUS[0]}
set -e

printf 'exit_code=%s\n' "$rc" > "$LOG_DIR/status"
echo "[stage2] torchrun exit_code=$rc；日志：$LOG_DIR/train.log；产物：$OUT_DIR"
echo "[stage2] 导出可评测权重："
echo "  python tools/export_hf_from_dcp.py --dcp $OUT_DIR/checkpoints/global_step_$MAX_STEPS \\"
echo "      --assets-from $OUT_DIR/model_assets --out <hf_dir>          # 普通权重"
echo "  python tools/export_hf_from_dcp.py --ema ... （同上）    # EMA 权重"
exit "$rc"
