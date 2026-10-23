# PRO 6000 训练前要备齐的环境与权重

训练代码在 `rebuild_lbv2/`。`scripts/train_stage2.sh` 覆盖 5 个路径，并在启动前把 4 个教师权重路径写进 yaml 副本再训练。四个缺一不可：没设环境变量或文件不存在，脚本直接拒绝启动，不会跑到 `from_pretrained` 才报错。

规则约束（[初赛规则](https://tianchi.aliyun.com/competition/entrance/532514/customize898)）：训练起点必须是 `lingbot-vla-v2-6b`，训练数据只能是 RoboTwin 2.0 Aloha-AgileX 的 50 个任务 clean 数据、每任务 50 条。已提交的 `global_step_45000` 就是从这个起点在 clean 数据上训出来的，可以作为本轮起点。`lingbot-vla-v2-6b-robotwin` 是 clean+randomized 的后训练权重，**禁止**作为起点或教师。

## 1. 环境

官方脚本一次装齐，不要手装 `requirements.txt`：

```bash
cd rebuild_lbv2
bash tools/create_train_env.sh --env-name lingbotvla
```

它做的事：conda 建 Python 3.12 环境，装 `requirements.txt`（`torch==2.8.0`、`transformers==4.57.3` 等），再装 `flash-attn==2.8.3`。PRO 6000 是 Blackwell（sm_120），pip 上的 flash-attn 轮子大概率没有对应编译，优先用本地轮子：

```bash
bash tools/create_train_env.sh --env-name lingbotvla \
  --flash-attn-wheel /path/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

装完验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

要看到 CUDA 可用、显卡名含 `RTX PRO 6000`。`depth` 额外依赖在 `requirements-depth.txt`，官方训练脚本不装它；depth 教师走 `from_pretrained`，缺模块时再补装。

启动训练时把环境指过去（脚本默认找仓库上两级的 `miniconda3/envs/lingbotvla`）：

```bash
TRAIN_ENV=/path/miniconda3/envs/lingbotvla bash scripts/train_stage2.sh
```

## 2. 必加载的权重与数据

| # | 内容 | 放到 | 来源 | 脚本覆盖 |
|---|---|---|---|---|
| 1 | 起点权重 `global_step_45000/hf_ckpt`（F32，25.5 GB，6 分片） | 任意目录 | 已提交的那份 | `BASE_HF_CKPT` |
| 2 | clean 数据集 + 清单 | 清单里写新机绝对路径 | [RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)，只要 Aloha-AgileX 50 任务 × 50 条 clean | `DATA_MANIFEST` |
| 3 | Qwen3-VL-4B-Instruct | 任意目录 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | `TOKENIZER_PATH` |
| 4 | MoGe-2 | `MOGE_PATH` | [Ruicheng/moge-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal) 的 `moge2-vitb-normal.pt` | 是 |
| 5 | LingBot-Depth | `MORGBD_PATH` | [lingbot-vla-v2-6b/depth](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main/depth) | 是 |
| 6 | DINO-Video 教师 | `DINO_VIDEO_CKPT` 与 `DINO_VIDEO_CFG` | [lingbot-vla-v2-6b/dino_video](https://huggingface.co/robbyant/lingbot-vla-v2-6b/tree/main/dino_video) 的 `teacher_step_10000.pth` 和同目录 `config.yaml` | 是 |

4、5、6 是规则 3.4 明确允许的官方配方教师，不是可选项。`align_params` 非空时 `train_lingbotvla.py` 会打开 depth/video 对齐。

不在仓库里、需要下载的用官方脚本：

```bash
python3 scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir /w/qwen3vl
python3 scripts/download_hf_model.py --repo_id Ruicheng/moge-2-vitb-normal --local_dir /w/moge
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir /w/lingbot
```

`lingbot-vla-v2-6b` 这个仓库同时包含第 5、6 项（`depth/`、`dino_video/`）。不用手改 yaml：`train_stage2.sh` 启动时读这 4 个环境变量，写入 `$LOG_DIR/robotwin.run.yaml` 再训练，仓库里的 yaml 保持占位符不动。

```bash
TOKENIZER_PATH=/w/qwen3vl \
MOGE_PATH=/w/moge/moge2-vitb-normal.pt \
MORGBD_PATH=/w/lingbot/depth/model.pt \
DINO_VIDEO_CKPT=/w/lingbot/dino_video/teacher_step_10000.pth \
DINO_VIDEO_CFG=/w/lingbot/dino_video/config.yaml \
bash scripts/train_stage2.sh
```

`MORGBD_PATH` 的文件名以 `depth/` 目录里的实际文件为准，上面写的 `model.pt` 只是占位。四个里任何一个不存在，脚本打印缺哪个并退出。

## 3. 不用另备的

- `norm_stats`：用仓库内 `assets/norm_stats/robotwin_clean_50_selfcomputed.json`（count=549,787）。
- `robot_config_root`：`train_stage2.sh` 默认指向仓库内 `configs/robot_configs`，换机要改时设 `ROBOT_CONFIG_ROOT`。
- 运动权重：不是下载的，是在数据集上算的。不跑也能训练，只是低运动降权不生效：

```bash
python tools/build_motion_weights.py \
  --data-root /path/lerobot --manifest /path/clean_training_data.txt \
  --output-dir /path/motion_weights_clean50
# 产出 50 个 <data_name>.pt，训练时 MOTION_WEIGHTS=/path/motion_weights_clean50
```

## 4. 磁盘

单次 DCP ≈ 63.8 GB（模型 fp32 25.5 + EMA bf16 12.75 + Muon 动量 25.5），`save_steps=5000` 存 5 份 = 319 GB。连同起点权重、数据集、三份教师权重和导出峰值，预留 **≥ 600 GB**。

## 5. 起跑前核对

```bash
bash -n scripts/train_stage2.sh
test -d "$BASE_HF_CKPT" && test -f "$DATA_MANIFEST" && test -d "$TOKENIZER_PATH"
test -f /w/moge/moge2-vitb-normal.pt
test -f /w/lingbot/dino_video/teacher_step_10000.pth
df -h .            # 余量 ≥ 600 GB
nvidia-smi         # RTX PRO 6000，96 GB
```

然后先冒烟 50 步，确认日志里 `‖ema-model‖/‖model‖` 不是 0，再按方案 §3 的探针反算 `MAX_STEPS` 起长跑。
