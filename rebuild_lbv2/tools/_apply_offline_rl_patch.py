#!/usr/bin/env python3
"""把 offline RL (advantage-weighted flow matching) 接进官方训练代码。

用法:
    python tools/_apply_offline_rl_patch.py --check    # 只校验锚点
    python tools/_apply_offline_rl_patch.py --apply    # 写入
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

MODEL = "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
BASE = "lingbotvla/data/vla_data/base_dataset.py"
TRAIN = "tasks/vla/train_lingbotvla.py"
CFG = "configs/vla/robotwin/robotwin.yaml"

PATCHES = []


def patch(rel, old, new, note):
    PATCHES.append((rel, old, new, note))


# ---------------------------------------------------------------- 1. 模型 forward
patch(
    MODEL,
    """        future_video_current_patch=None,
        **kwargs
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_dict = {}""",
    """        future_video_current_patch=None,
        adv_weight=None,
        **kwargs
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_dict = {}""",
    "P1a LingbotVlaV2Policy.forward 增加 adv_weight 形参",
)

patch(
    MODEL,
    """        if joint_mask is not None:
            if "repeat" in self.config.loss_type:
                joint_mask = joint_mask.repeat(2, 1, 1)""",
    """        # ---- offline RL: advantage-weighted flow matching (AWAC-style) ----
        # 逐样本权重 w（形状 (B,)），由 lingbotvla.offline_rl.advantage 构造并归一化到
        # mean(w)=1，因此下面的归约公式无需改动即可保持损失量级与纯 BC 可比。
        #
        # 合规：训练全程零环境调用；权重只由 release 数据自身的 episode 边界推导
        # （成功轨迹末步 r=1 的折扣回报），见 offline_rl/advantage.py 顶部说明与规则 3.4。
        # adv_weight 为 None 时本分支完全不生效，行为与改动前逐位一致（纯 BC）。
        if adv_weight is not None:
            w = adv_weight.to(device=losses.device, dtype=losses.dtype).reshape(-1)
            if w.numel() != losses.shape[0]:
                raise ValueError(
                    f"adv_weight 长度 ({w.numel()}) 与 batch size ({losses.shape[0]}) 不一致"
                )
            losses = losses * w.view(-1, 1, 1)

        if joint_mask is not None:
            if "repeat" in self.config.loss_type:
                joint_mask = joint_mask.repeat(2, 1, 1)""",
    "P1b 归约前按样本加权 flow-matching 损失",
)

# ---------------------------------------------------------------- 2. 数据集
patch(
    BASE,
    """        self.return_item = return_item
        self.transform = transform

    def __len__(self):
        return len(self.dataset)""",
    """        self.return_item = return_item
        self.transform = transform

        # ---- offline RL: advantage 权重 sidecar（可选）----
        # 由 tools/build_adv_weights.py 生成；缺失或未配置时 self.adv_weights 为 None，
        # 训练行为与改动前完全一致（纯 BC）。
        self.adv_weights = None
        adv_weight_path = getattr(dataset_config, "adv_weight_path", None)
        if adv_weight_path:
            from ...offline_rl.advantage import load_weights

            _w, _cfg = load_weights(adv_weight_path)
            _n_expected = len(self.dataset)
            if int(_w.numel()) != _n_expected:
                raise ValueError(
                    f"advantage 权重数量 ({int(_w.numel())}) 与本数据集样本数 "
                    f"({_n_expected}) 不一致：{adv_weight_path}\\n"
                    "请确认 build_adv_weights.py 用的 train_path 与 data.train_path 完全一致。"
                )
            self.adv_weights = _w
            logger.info(
                f"offline RL: 已加载 advantage 权重 {adv_weight_path} "
                f"(n={int(_w.numel())}, config={_cfg})"
            )

    def __len__(self):
        return len(self.dataset)""",
    "P2a VLADataset.__init__ 加载 advantage 权重 sidecar",
)

patch(
    BASE,
    """        item = self.feature_transform.apply(raw_item)
        if self.transform is not None:
            item = self.transform(item, raw_item, self.feature_transform.feature_config.images, self.feature_transform.key_mapping)
        return item""",
    """        item = self.feature_transform.apply(raw_item)
        if self.transform is not None:
            item = self.transform(item, raw_item, self.feature_transform.feature_config.images, self.feature_transform.key_mapping)
        # offline RL: 把该样本的 advantage 权重挂进样本，经 collator 变成 (B,) 张量后
        # 由训练循环通过 model(**micro_batch) 传给 LingbotVlaV2Policy.forward。
        if self.adv_weights is not None:
            item["adv_weight"] = self.adv_weights[idx].clone()
        return item""",
    "P2b VLADataset.getitem 返回逐样本 adv_weight",
)

# ---------------------------------------------------------------- 3. 训练脚本配置
patch(
    TRAIN,
    """    use_future_image: bool = field(
        default=False,
        metadata={"help": "Whether to use future image."},
    )


@dataclass
class Arguments:""",
    """    use_future_image: bool = field(
        default=False,
        metadata={"help": "Whether to use future image."},
    )
    # ---- offline RL (advantage-weighted flow matching) ----
    # 指向 tools/build_adv_weights.py 生成的 .pt；留空则退化为纯 BC。
    # 该权重由 release 数据的 episode 边界推导，训练全程零环境调用（规则 3.4 允许）。
    adv_weight_path: str = field(
        default=None,
        metadata={"help": "Path to advantage-weight sidecar (.pt) for offline RL; None disables it."},
    )


@dataclass
class Arguments:""",
    "P3 MyDataArguments 增加 adv_weight_path",
)

# ---------------------------------------------------------------- 4. 配置 yaml
patch(
    CFG,
    """  num_workers: 8
  use_future_image: true""",
    """  num_workers: 8
  use_future_image: true
  # ---- offline RL (advantage-weighted flow matching) ----
  # 指向 tools/build_adv_weights.py 的产物；null = 关闭（纯 BC）。
  # 权重来自 release 数据的 episode 边界，训练全程零环境调用，规则 3.4 明确允许。
  adv_weight_path: null""",
    "P4 robotwin.yaml 增加 data.adv_weight_path",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not (a.check or a.apply):
        ap.error("需要 --check 或 --apply")

    print("=" * 74)
    print("offline RL 补丁校验" if a.check else "offline RL 补丁写入")
    print("=" * 74)

    contents = {}
    ok = True
    for rel, old, new, note in PATCHES:
        p = ROOT / rel
        if rel not in contents:
            contents[rel] = p.read_text(encoding="utf-8")
        s = contents[rel]
        n = s.count(old)
        already = s.count(new) if new else 0
        if n == 0 and already > 0:
            print(f"  ⏭️  {note}  （已应用过，跳过）")
            continue
        if n != 1:
            print(f"  ❌ {note}\n     锚点在 {rel} 中出现 {n} 次（期望 1）")
            ok = False
            continue
        print(f"  ✅ {note}  （锚点唯一）")
        contents[rel] = s.replace(old, new, 1)

    if not ok:
        print("\n❌ 锚点校验失败，未写入任何文件。")
        return 1
    if a.check:
        print("\n✅ 全部锚点校验通过（未写入）。加 --apply 执行写入。")
        return 0

    for rel, s in contents.items():
        (ROOT / rel).write_text(s, encoding="utf-8")
        print(f"  💾 已写入 {rel}")

    # 语法校验
    import ast

    for rel in contents:
        ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    print("\n✅ 补丁已应用，且全部文件语法通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
