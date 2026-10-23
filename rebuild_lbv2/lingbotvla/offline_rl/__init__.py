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
"""offline RL（advantage-weighted flow matching）。

见 :mod:`lingbotvla.offline_rl.advantage` 顶部的合规说明。
"""

from .advantage import (
    AdvantageConfig,
    advantage_weights,
    build_weights_from_lengths,
    build_weights_from_lerobot,
    episode_lengths_to_returns,
    load_weights,
    read_lerobot_episode_lengths,
    save_weights,
)

__all__ = [
    "AdvantageConfig",
    "advantage_weights",
    "build_weights_from_lengths",
    "build_weights_from_lerobot",
    "episode_lengths_to_returns",
    "load_weights",
    "read_lerobot_episode_lengths",
    "save_weights",
]
