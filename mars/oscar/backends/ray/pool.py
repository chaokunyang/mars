# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import asyncio
import inspect
import logging
import os
import types
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from .communication import ChannelID, RayServer
from .utils import process_address_to_placement, process_placement_to_address, get_placement_group
from ..mars.config import ActorPoolConfig
from ..mars.pool import AbstractActorPool, MainActorPool, SubActorPool, create_actor_pool
from ....utils import lazy_import

ray = lazy_import('ray')
logger = logging.getLogger(__name__)


class RayActorPoolMixin(AbstractActorPool, ABC):

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        """Method for communication based on ray actors"""
        if not hasattr(self, '_external_servers'):
            ray_servers = [server for server in self._servers if isinstance(server, RayServer)]
            assert len(ray_servers) == 1, f"Ray only support single server but got {ray_servers}."
            self._external_servers = ray_servers
        reply = await self._external_servers[0].__on_ray_recv__(channel_id, message)
        return await _serialize(reply)


async def _serialize(message):
    # TODO(chaokunyang) register mars serializer with ray.util.register_serializer
    from ..mars.message import ResultMessage
    if isinstance(message, ResultMessage) and isinstance(message.result, asyncio.Future):
        message.result = await message.result
    return message


class RayMainActorPool(RayActorPoolMixin, MainActorPool):

    @classmethod
    def get_external_addresses(
            cls, address: str, n_process: int = None, ports: List[int] = None):
        assert not ports, f"ports should be none when actor pool running on ray, but got {ports}"
        pg_name, bundle_index, _process_index = process_address_to_placement(address)
        return [process_placement_to_address(pg_name, bundle_index, i) for i in range(n_process + 1)]

    @classmethod
    def get_sub_pool_manager_cls(cls):
        return cls.RaySubActorPoolManager

    class RaySubActorPoolManager(MainActorPool.SubActorPoolManager):

        @classmethod
        async def start_sub_pool(
                cls,
                actor_pool_config: ActorPoolConfig,
                process_index: int,
                start_method: str = None):
            external_addresses = \
                actor_pool_config.get_pool_config(process_index)['external_address']
            assert len(external_addresses) == 1, \
                f"Ray pool allows only one external address but got {external_addresses}"
            external_address = external_addresses[0]
            pg_name, bundle_index, _process_index = process_address_to_placement(external_address)
            assert process_index == _process_index, \
                f"process_index {process_index} is not consistent with index {_process_index} " \
                f"in external_address {external_address}"
            pg = get_placement_group(pg_name) if pg_name else None
            if not pg:
                bundle_index = -1
            # Hold actor_handle to avoid actor being freed.
            actor_handle = ray.remote(RaySubPool).options(
                name=external_address, placement_group=pg,
                placement_group_bundle_index=bundle_index).remote()
            await actor_handle.start.remote(actor_pool_config, process_index)
            return actor_handle

        def kill_sub_pool(self, process: 'ray.actor.ActorHandle'):
            ray.kill(process)

        async def is_sub_pool_alive(self, process: 'ray.actor.ActorHandle'):
            try:
                await process.health_check.remote()
                return True
            except Exception:
                logger.info("Detected RaySubPool %s died", process)
                return False


class RaySubActorPool(RayActorPoolMixin, SubActorPool):
    pass


class PoolStatus(Enum):
    HEALTHY = 0
    UNHEALTHY = 1


class RayPoolBase(ABC):
    actor_pool: Optional['RayActorPoolMixin']

    def __init__(self):
        self.actor_pool = None

    @abstractmethod
    async def start(self, *args, **kwargs):
        raise NotImplementedError

    async def __on_ray_recv__(self, channel_id: ChannelID, message):
        return await self.actor_pool.__on_ray_recv__(channel_id, message)

    def health_check(self):
        return PoolStatus.HEALTHY

    async def __proxy_call__(self, attribute, *args, **kwargs):
        attr = getattr(self.actor_pool, attribute)
        if isinstance(attr, types.MethodType):
            if inspect.iscoroutinefunction(attr):
                return await attr(*args, **kwargs)
            return attr(*args, **kwargs)
        else:
            return attr


class RayMainPool(RayPoolBase):
    actor_pool: RayMainActorPool

    async def start(self, address, n_process, **kwargs):
        self.actor_pool = await create_actor_pool(
            address, n_process=n_process, pool_cls=RayMainActorPool,
            subprocess_start_method="ray", **kwargs)


class RaySubPool(RayPoolBase):
    actor_pool: RaySubActorPool

    async def start(self, *args, **kwargs):
        actor_config, process_index = args
        env = actor_config.get_pool_config(process_index)['env']
        if env:
            os.environ.update(env)
        pool = await RaySubActorPool.create({
            'actor_pool_config': actor_config,
            'process_index': process_index
        })
        await pool.start()
        self.actor_pool = pool
        asyncio.create_task(pool.join())
