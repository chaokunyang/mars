# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List

from .....core.operand.shuffle import ShuffleType
from .....resource import Resource
from ..api import ExecutionConfig, register_config_cls
from ..utils import get_band_resources_from_config


# the default times to retry subtask.
DEFAULT_SUBTASK_MAX_RETRIES = 3


@register_config_cls
class RayExecutionConfig(ExecutionConfig):
    name = "ray"

    def __init__(self, execution_config: Dict):
        super().__init__(execution_config)
        self._subtask_max_retries = execution_config.get("ray", {}).get(
            "subtask_max_retries", DEFAULT_SUBTASK_MAX_RETRIES
        )

    def get_band_resources(self):
        """
        Get the band resources from config for generating ray virtual
        resources.
        """
        return get_band_resources_from_config(self._config)

    def get_deploy_band_resources(self) -> List[Dict[str, Resource]]:
        return []

    def get_shuffle_type(self) -> ShuffleType:
        return ShuffleType.PUSH

    @property
    def subtask_max_retries(self):
        return self._subtask_max_retries
