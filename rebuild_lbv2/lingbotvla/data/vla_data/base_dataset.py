# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import inspect
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import hf_transform_to_torch
    LEROBOT_DATASET_API = "v3"
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import hf_transform_to_torch
    LEROBOT_DATASET_API = "v2"

from datasets import load_dataset as _hf_load_dataset

from ...utils import logging
from .utils import FeatureTransform
from .video_utils import decode_video_frames


logger = logging.get_logger(__name__)


def _to_relative_indices(dataset, query_indices):
    index_map = getattr(dataset, "_absolute_to_relative_idx", None)
    if index_map is None:
        return query_indices
    return [index_map[idx] for idx in query_indices]


def _get_task_name(tasks, task_idx):
    if hasattr(tasks, "iloc"):
        return tasks.iloc[task_idx].name
    return tasks[task_idx]


def _resolve_lerobot_location(repo_id):
    repo_path = Path(repo_id).expanduser()
    if repo_path.exists():
        return repo_path.name, repo_path
    return repo_id, None


def _filter_supported_kwargs(callable_obj, kwargs):
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}



class LeRobotDataset(BaseLeRobotDataset):    
    def __init__(
        self,
        repo_id: str,
        load_image: bool = True,
        **kwargs,
    ):
        super().__init__(repo_id, **kwargs)
        self.load_image = load_image

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """
        Query dataset for indices across keys, skipping video keys.

        Tries column-first [key][indices] for speed, falls back to row-first.

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)
        """
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            # Map absolute indices to relative indices if needed
            relative_indices = _to_relative_indices(self, q_idx)
            result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            if LEROBOT_DATASET_API == "v3":
                # LeRobot v3 stores episodes sequentially in a shared mp4, so
                # query timestamps are relative to the episode start.
                ep = self.meta.episodes[ep_idx]
                from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
                query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def __getitem__(self, idx) -> dict:
        # Ensure dataset is loaded when we actually need to read from it
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
            
        if len(self.meta.video_keys) > 0 and self.load_image:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None and self.load_image:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = _get_task_name(self.meta.tasks, task_idx)

        return item

class VLADataset(Dataset):
    def __init__(
        self,
        repo_id,
        data_name,
        dataset_config,
        robot_config_root,
        config=None,
        processor=None,
        video_backend = 'torchcodec',
        chunk_size = 50,
        image_size = (224, 224),
        do_nomalize = True,
        return_item = False,
        disabled_image_features = False,
        feature_transform = None,
        transform=None,
        image_augment = False,
        use_depth_align = False,
        use_future_image = False,
    ):
        if do_nomalize and config is None:
            raise ValueError("VLADataset requires a model config; pass model.config via build_vla_dataset.")

        self.processor = processor
        self.config = config
        self.chunk_size = chunk_size
        self.data_name = data_name
        self.disabled_image_features = disabled_image_features
        self.use_depth_align = use_depth_align
        self.use_future_image = use_future_image

        load_image = True if do_nomalize else False

        if feature_transform is None:
            robot_config = os.path.join(robot_config_root, f'{data_name}.yaml')
            self.feature_transform = FeatureTransform(robot_config, dataset_config, self.config, \
                        processor, disabled_image_features, do_nomalize, \
                        chunk_size=chunk_size, return_item_befor_padding=return_item,\
                        image_augment=image_augment, use_depth_align=use_depth_align,
                        use_future_image=use_future_image)
        else:
            self.feature_transform = feature_transform

        self.action_features = self.feature_transform.actions
        self.state_features = self.feature_transform.states
        self.image_features = self.feature_transform.images
        
        lerobot_repo_id, lerobot_root = _resolve_lerobot_location(repo_id)
        metadata_kwargs = _filter_supported_kwargs(
            LeRobotDatasetMetadata.__init__,
            {"repo_id": lerobot_repo_id, "root": lerobot_root},
        )
        self.dataset_meta = LeRobotDatasetMetadata(**metadata_kwargs)
        merged_delta = {**self.get_delta_timestamps(), **self.get_video_delta_timestamps()}

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            image_transforms=Resize(image_size),
            delta_timestamps=merged_delta,
            load_image=load_image
        )

        self.return_item = return_item
        self.transform = transform

        # ---- offline RL: advantage 权重 sidecar（可选）----
        # 由 tools/build_adv_weights.py 生成；缺失或未配置时 self.adv_weights 为 None，
        # 训练行为与改动前完全一致（纯 BC）。
        self.adv_weights = None
        adv_weight_path = getattr(dataset_config, "adv_weight_path", None)
        if adv_weight_path and os.path.isdir(adv_weight_path):
            # 目录形式：每个任务一份 sidecar（多任务训练必须用这种形式）。
            # 原因：sidecar 的校验口径是"与本数据集的样本数一致"（见下），而
            # MultiVLADataset 会把同一份 dataset_config 传给每个任务的 VLADataset，
            # 因此单个全局 sidecar 无法通过校验。目录形式对单任务训练同样适用。
            _p = os.path.join(str(adv_weight_path), f"{self.data_name}.pt")
            if not os.path.isfile(_p):
                _peek = sorted(os.listdir(str(adv_weight_path)))[:3]
                raise FileNotFoundError(
                    f"adv_weight_path 指向目录，但缺少本任务对应的 sidecar：{_p}\n"
                    f"（目录内前几个文件：{_peek}…）\n"
                    "请用 tools/build_motion_weights.py --output-dir <dir> 重新生成。"
                )
            adv_weight_path = _p
        if adv_weight_path:
            from ...offline_rl.advantage import load_weights

            _w, _cfg = load_weights(adv_weight_path)
            _n_expected = len(self.dataset)
            if int(_w.numel()) != _n_expected:
                raise ValueError(
                    f"advantage 权重数量 ({int(_w.numel())}) 与本数据集样本数 "
                    f"({_n_expected}) 不一致：{adv_weight_path}\n"
                    "请确认 build_adv_weights.py 用的 train_path 与 data.train_path 完全一致。"
                )
            self.adv_weights = _w
            # 用 (episode_index, frame_index) 定位权重，而不是扁平 idx —— 与 LeRobot
            # 底层样本的物理排列顺序解耦。sidecar 里存了每条 episode 的长度，据此
            # 算出 episode -> 全局起始偏移。
            self._adv_episode_offsets = None
            _lens = (_cfg or {}).get("episode_lengths")
            if _lens:
                _lt = torch.tensor(list(_lens), dtype=torch.long)
                if int(_lt.sum()) != _n_expected:
                    raise ValueError(
                        f"sidecar 里 episode 长度之和 ({int(_lt.sum())}) 与数据集样本数 "
                        f"({_n_expected}) 不一致：{adv_weight_path}"
                    )
                self._adv_episode_offsets = torch.cat(
                    [torch.zeros(1, dtype=torch.long), _lt.cumsum(0)[:-1]]
                )
                logger.info(
                    f"offline RL: 权重按 (episode_index, frame_index) 定位 "
                    f"(n_episodes={int(_lt.numel())})"
                )
            else:
                logger.warning(
                    "offline RL: sidecar 缺少 episode_lengths，回退到扁平索引定位；"
                    "若权重与样本错位，请用新版 tools/build_adv_weights.py 重新生成。"
                )
            logger.info(
                f"offline RL: 已加载 advantage 权重 {adv_weight_path} "
                f"(n={int(_w.numel())}, config={_cfg})"
            )

    def __len__(self):
        return len(self.dataset)

    def get_features(self):
        features = set()
        for feature_category, _features in self.feature_transform.org_features.items():
            if len(self.feature_transform.actions_convert_from_state)>0 and feature_category == 'actions':
                continue
            features.update(_features)
        features.update(self.feature_transform.feature_to_keep)
        features = [x for x in list(features) if x not in ['action_is_pad', 'task', 'subtask']]
        return features

    def get_delta_timestamps(self, return_indices = False):
        delta_timestamps = {}
        fps = None if return_indices else self.dataset_meta.fps
        if not len(self.feature_transform.actions_convert_from_state)>0:
            for action_feature in self.feature_transform.org_features['actions']:
                delta_timestamps[action_feature] = [t / fps if fps else t for t in range(self.chunk_size)]
        else:
            for state_feature in self.feature_transform.org_features['states']:
                delta_timestamps[state_feature] = [t / fps if fps else t for t in range(self.chunk_size+1)]
        return delta_timestamps

    def get_video_delta_timestamps(self):
        """Multi-frame time offsets for video keys; returns an empty dict when disabled."""

        fps = self.dataset_meta.fps
        if self.use_future_image:
            offsets = [0, (self.chunk_size - 1) / fps]
            return {cam: offsets for cam in self.feature_transform.org_features['images']}
        else:
            return {}

    def check_lerobot_item(self, item):
        # if state or action is a 0-d tensor, convert it to 1-d tensor
        if not len(self.feature_transform.actions_convert_from_state)>0:
            for action_feature in self.feature_transform.org_features['actions']:
                if len(item[action_feature].shape) == 1:
                    item[action_feature] = item[action_feature].unsqueeze(-1)
            
            for state_feature in self.feature_transform.org_features['states']:
                if len(item[state_feature].shape) == 0:
                    item[state_feature] = item[state_feature].unsqueeze(-1)
        else:
            for state_feature in self.feature_transform.org_features['states']:
                if len(item[state_feature].shape) == 1:
                    item[state_feature] = item[state_feature].unsqueeze(-1)
        return item

    def getitem(self, idx):
        raw_item = self.check_lerobot_item(self.dataset[idx])
        if (
            self.use_future_image
            and "future_video_effective_fps" not in raw_item
            and hasattr(self, "dataset_meta")
        ):
            raw_item["future_video_effective_fps"] = torch.tensor(
                float(self.dataset_meta.fps) / float(max(1, self.chunk_size - 1)),
                dtype=torch.float32,
            )
        item = self.feature_transform.apply(raw_item)
        if self.transform is not None:
            item = self.transform(item, raw_item, self.feature_transform.feature_config.images, self.feature_transform.key_mapping)
        # offline RL: 把该样本的 advantage 权重挂进样本，经 collator 变成 (B,) 张量后
        # 由训练循环通过 model(**micro_batch) 传给 LingbotVlaV2Policy.forward。
        if self.adv_weights is not None:
            if self._adv_episode_offsets is not None:
                # 顺序无关：用 LeRobot 自带的 (episode_index, frame_index) 定位，
                # 不依赖 self.dataset 的物理排列顺序，因此转换脚本/版本差异
                # 不会造成权重与样本错位。
                _ep = int(raw_item["episode_index"])
                _fr = int(raw_item["frame_index"])
                _pos = int(self._adv_episode_offsets[_ep]) + _fr
                item["adv_weight"] = self.adv_weights[_pos].clone()
            else:
                item["adv_weight"] = self.adv_weights[idx].clone()
        return item

    def __getitem__(self, idx):
        item = self.getitem(idx)
        return item