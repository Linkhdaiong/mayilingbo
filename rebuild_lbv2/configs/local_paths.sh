#!/usr/bin/env bash
# Local machine paths for the one-shot final fine-tuning/evaluation workflow.
# Fill these absolute paths on the 8xA100 server before launching.
# 训练和评测入口共同 source 此文件。把 ${NAME:-} 改成
# ${NAME:-/服务器/绝对路径}；环境变量优先。TRAIN_ENV 填环境目录，不填名字。
# BASE_YAML 可用仓库相对路径，其余资源路径使用绝对路径。
# 引号中的路径可以包含空格；数据清单为逐行「任务名 数据集绝对路径」，
# 清单内路径不支持空格。GPU_IDS 为逗号分隔的物理卡编号，训练要求 8 个。
#
# Required training paths:
#   BASE_HF_CKPT   Existing submitted checkpoint, ending in .../global_step_45000/hf_ckpt
#   DATA_MANIFEST  Release training manifest used by the original fine-tune
#   TOKENIZER_PATH Local Qwen3-VL tokenizer/model directory
#   OUT_DIR        New output directory; it must not already exist
#   MOGE_PATH, MORGBD_PATH, DINO_VIDEO_CKPT, DINO_VIDEO_CFG: teacher assets
#   MOTION_WEIGHTS: directory containing one <data_name>.pt per manifest task
#
# Required evaluation paths:
#   EVAL_WORKDIR  RoboTwin checkout root
#   OUTPUT_BASE   Evaluation output root
#   CONDA_SH      conda.sh for the two environments
#
# Parsing rule: absolute paths are consumed literally after shell expansion;
# no path is inferred from the checkpoint name. Quote path assignments.
# A value left empty is rejected when required.

BASE_HF_CKPT="${BASE_HF_CKPT:-}"
DATA_MANIFEST="${DATA_MANIFEST:-}"
DATA_ROOT="${DATA_ROOT:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
OUT_DIR="${OUT_DIR:-}"
LOG_DIR="${LOG_DIR:-}"
TRAIN_ENV="${TRAIN_ENV:-}"
BASE_YAML="${BASE_YAML:-configs/vla/robotwin/robotwin.yaml}"
ROBOT_CONFIG_ROOT="${ROBOT_CONFIG_ROOT:-}"
MOGE_PATH="${MOGE_PATH:-}"
MORGBD_PATH="${MORGBD_PATH:-}"
DINO_VIDEO_CKPT="${DINO_VIDEO_CKPT:-}"
DINO_VIDEO_CFG="${DINO_VIDEO_CFG:-}"
MOTION_WEIGHTS="${MOTION_WEIGHTS:-}"

EVAL_WORKDIR="${EVAL_WORKDIR:-}"
OUTPUT_BASE="${OUTPUT_BASE:-}"
CONDA_SH="${CONDA_SH:-}"
MODEL_PATH="${MODEL_PATH:-}"
INFERENCE_WORKDIR="${INFERENCE_WORKDIR:-}"
QWEN3VL_PATH="${QWEN3VL_PATH:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DVAC_ARGS="${DVAC_ARGS:-}"
INFERENCE_ENV="${INFERENCE_ENV:-lingbotvla}"
SIM_ENV="${SIM_ENV:-RoboTwin}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-}"
