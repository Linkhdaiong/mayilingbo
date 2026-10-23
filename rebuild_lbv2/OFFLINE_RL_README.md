# offline RL（advantage-weighted flow matching）实现说明

> 状态：**已实现并验证**（2026-10-10）。默认关闭，行为与改动前完全一致。
> 定位：**消融对照，不是主线** —— 原因见文末「权重方向问题」。

## 一、它是什么

在官方 `LingBotVlaV2Policy` 的 flow-matching 损失上，按样本的 advantage 权重加权：

```
L = mean( w_i * L1_fm(u_t, v_t) )
```

权重由 release 数据**内部**推导，训练全程**零环境调用**：

```
每条 episode 最后一步 r = 1，其余 r = 0
G_t = sum_{k>=t} gamma^(k-t) * r_k          # 折扣回报（离终点越近越大）
A_t = (G_t - mean(G)) / std(G)              # 标准化优势
w_t = clip(exp(beta * A_t), w_min, w_max)   # AWAC 式权重
w_t = w_t / mean(w)                         # 归一化，保持损失量级
```

`gamma = 1.0` 时 `w ≡ 1`，**退化为纯 BC** —— 天然的零效应基线，用于 A/B 对照。

## 二、文件

| 文件 | 作用 |
|---|---|
| `lingbotvla/offline_rl/advantage.py` | 核心：`AdvantageConfig`、episode 长度 → 回报/权重、sidecar 存取 |
| `lingbotvla/offline_rl/__init__.py` | 导出 |
| `tools/build_adv_weights.py` | 生成权重 sidecar，含集中度诊断 |
| `tests/test_advantage_weights.py` | 数学测试（38 项） |
| `tests/test_offline_rl_integration.py` | 集成测试（20 项） |
| `tests/test_weight_alignment.py` | 索引对齐测试（12 项） |
| `tests/test_config_chain.py` | 配置链路测试（7 项） |

改动（6 处锚点，均在原文件内）：

| 文件 | 改动 |
|---|---|
| `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py` | `forward(adv_weight=None)`；加权在损失归约**之前**；`None` 时短路 |
| `lingbotvla/data/vla_data/base_dataset.py` | 加载 sidecar；按 `(episode_index, frame_index)` **顺序无关**定位权重 |
| `tasks/vla/train_lingbotvla.py` | `MyDataArguments.adv_weight_path` |
| `configs/vla/robotwin/robotwin.yaml` | 默认关闭（注释说明，**不要写 null**） |

## 三、怎么用

```bash
# 1. 生成权重（需先把 release 数据转成 LeRobot 格式）
python tools/build_adv_weights.py \
    --train-path assets/training_data/robotwin.txt \
    --out /path/to/adv_weights.pt \
    --gamma 0.99 --beta 0.25 --w-min 0.5 --w-max 2.0 \
    --verify

# 2. 训练时启用
python tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml \
    --data.adv_weight_path=/path/to/adv_weights.pt
```

**不加 `--data.adv_weight_path` 就是纯 BC**，与改动前逐位一致。

## 四、⚠️ 两个已知陷阱

### 1. 不要在 yaml 里写 `adv_weight_path: null`

官方参数解析器会把 yaml 的 `null` 变成**字符串 `'null'`**，而非空字符串是**真值**，
训练时会去找一个名为 `"null"` 的文件并崩溃。默认关闭请**整个省略该键**（或注释掉）。

### 2. 权重集中度（这条是设计层面的）

`terminal_gamma` 模式下回报随「离终点的距离」单调变化。若裁剪区间放得太宽，
加权会把训练集中到轨迹的最后一小段。实测（`gamma=0.99`、300 帧、按优势 std 标准化后乘 β）：

| 配置 | 剖面 [t=0, 25%, 50%, 75%, 末] | 极差 | `w < 0.5×均值` 占比 |
|---|---|---|---|
| `β=2.0`，裁剪 `[0.1, 10]` | `[0.049, 0.075, 0.187, 1.317, 3.861]` | 79.4× | **65.0%** |
| `β=1.0`，裁剪 `[0.1, 10]` | `[0.198, 0.246, 0.389, 1.031, 5.583]` | 28.2× | **58.7%** |
| **默认 `β=0.25`，裁剪 `[0.5, 2.0]`** | `[0.746, 0.788, 0.883, 1.127, 1.873]` | **2.5×** | **0.0%** |

前两行意味着**超过一半的 episode（接近、抓取、运输）被压掉 2 倍以上**，
对需要**全程**视觉鲁棒性的任务（如 clean2random）大概率是负收益。

**默认裁剪区间 `[0.5, 2.0]` 是结构性保护**：它把最坏情况硬性限制在 ±2 倍以内。
实测即使把 `β` 误设成 50，极差也只到 4.0×，不会退化成「只训收尾」。
`build_adv_weights.py` 会打印集中度诊断，超过 40% 会告警。

## 五、索引对齐（为什么不怕错位）

权重 sidecar 是一维张量。若按「扁平 idx」取权重，就隐含假设
**LeRobot 底层样本的物理排列顺序 == 生成权重时的 episode 拼接顺序**。
转换脚本版本差异或分片方式变化都会打破这个假设，造成**静默错位**（不报错，只是掉分）。

因此实现改为用 LeRobot 自带的 `(episode_index, frame_index)` 定位：

```python
pos = episode_offsets[item["episode_index"]] + item["frame_index"]
```

sidecar 里存了 `episode_lengths`，据此算出每条 episode 的全局起始偏移。
`tests/test_weight_alignment.py` 验证：物理顺序打乱后权重仍正确；
对照组显示旧的扁平索引方式在打乱后会错位 **19/25** 个样本。

## 六、定位建议

**当消融对照，不当主线。** clean 数据里每条轨迹都成功，advantage 主要区分
「离完成还有多远」而非「哪条轨迹更好」，信号比有失败样本的常规 offline RL 弱得多。

按收益排序，优先级更高的方向：
1. **推理期策略** —— 规则明确「评测不限制单次推理时间与显存」。temporal ensemble、
   action chunk 长度调优、rollout 内重规划、FP32 推理，白送的分；
2. **视觉增强** —— 直接对着 clean2random 的域偏移做；
3. **clean-only 训练 + 50 任务 cotrain** —— 合规底线。

A/B 方式：`--gamma 1.0`（退化为纯 BC）vs 默认配置，同批验证集 loss + 校准评测成功率。
