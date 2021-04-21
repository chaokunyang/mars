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
import functools
import logging
import os
import uuid
import yaml
from typing import Union, Dict, List
import cloudpickle

from mars.oscar.backends.ray.driver import RayActorDriver
from mars.serialization import serialize, deserialize
from mars.serialization.ray import register_ray_serializers
from .pool import create_supervisor_actor_pool, create_worker_actor_pool
from .service import start_supervisor, start_worker, stop_supervisor, stop_worker
from ... import oscar as mo
from ...core.session import _new_session, AbstractSession, SessionType, ExecutionInfo, register_session_cls
from ...utils import lazy_import, implements

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    aiohttp = None
    web = None

ray = lazy_import('ray')
logger = logging.getLogger(__name__)


def _load_config(filename=None):
    # use default config
    if not filename:  # pragma: no cover
        d = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(d, 'rayconfig.yml')
    with open(filename) as f:
        return yaml.safe_load(f)


async def new_cluster(cluster_name: str,
                      supervisor_mem: int = 4 * 1024 ** 3,
                      worker_num: int = 1,
                      worker_cpu: int = 16,
                      worker_mem: int = 32 * 1024 ** 3,
                      config: Union[str, Dict] = None):
    config = config or _load_config()
    cluster = RayCluster(cluster_name, supervisor_mem, worker_num,
                         worker_cpu, worker_mem, config)
    await cluster.start()
    return await RayClient.create(cluster)


class RayCluster:
    _supervisor_pool: 'ray.actor.ActorHandle'
    _worker_pools: List['ray.actor.ActorHandle']
    session_proxy: Union['SessionProxy', None]

    def __init__(self,
                 cluster_name: str,
                 supervisor_mem: int = 4 * 1024 ** 3,
                 worker_num: int = 1,
                 worker_cpu: int = 16,
                 worker_mem: int = 32 * 1024 ** 3,
                 config: Union[str, Dict] = None):
        self._cluster_name = cluster_name
        self._supervisor_mem = supervisor_mem
        self._worker_num = worker_num
        self._worker_cpu = worker_cpu
        self._worker_mem = worker_mem
        self._config = config
        self._band_to_slot = band_to_slot = dict()
        # TODO(chaokunyang) Support gpu
        band_to_slot['numa-0'] = self._worker_cpu
        self.supervisor_address = None
        # Hold actor handles to avoid being freed
        self._supervisor_pool = None
        self._worker_addresses = []
        self._worker_pools = []
        self.session_proxy = None

    async def start(self):
        address_to_resources = dict()
        supervisor_node_address = f'ray://{self._cluster_name}/0'
        address_to_resources[supervisor_node_address] = {
            'CPU': 1,
            # 'memory': self._supervisor_mem
        }
        worker_node_addresses = []
        for worker_index in range(1, self._worker_num + 1):
            worker_node_address = f'ray://{self._cluster_name}/{worker_index}'
            worker_node_addresses.append(worker_node_address)
            address_to_resources[worker_node_address] = {
                'CPU': self._worker_cpu,
                # 'memory': self._worker_mem
            }
        mo.setup_cluster(address_to_resources)

        # create supervisor actor pool
        self._supervisor_pool = await create_supervisor_actor_pool(
            supervisor_node_address, n_process=0)
        # start service
        self.supervisor_address = f'{supervisor_node_address}/0'
        await start_supervisor(self.supervisor_address, config=self._config)

        for worker_node_address in worker_node_addresses:
            worker_pool = await create_worker_actor_pool(worker_node_address, self._band_to_slot)
            self._worker_pools.append(worker_pool)
            worker_address = f'{worker_node_address}/0'
            self._worker_addresses.append(worker_address)
            await start_worker(worker_address,
                               self.supervisor_address,
                               self._band_to_slot,
                               config=self._config)

        self.session_proxy = await SessionProxy.create(self)

    async def stop(self):
        await self.session_proxy.stop()
        for worker_address in self._worker_addresses:
            await stop_worker(worker_address, self._config)
        await stop_supervisor(self.supervisor_address, self._config)
        for pool in self._worker_pools:
            await pool.actor_pool.remote('stop')
        await self._supervisor_pool.actor_pool.remote('stop')


class RayClient:
    def __init__(self,
                 cluster: RayCluster,
                 session: 'SessionType'):
        self.cluster = cluster
        self._address = cluster.supervisor_address
        self._session = session

    @classmethod
    async def create(cls, cluster: RayCluster) -> 'RayClient':
        session = await _new_session(
            cluster.supervisor_address, backend='oscar', default=True)
        cluster.session_proxy.add_session(session._session_id, session)
        return RayClient(cluster, session)

    @property
    def address(self):
        return self._session.address

    @property
    def session(self):
        return self._session

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def stop(self):
        await self.cluster.stop()
        RayActorDriver.stop_cluster()


class SessionProxy:

    def __init__(self, cluster: RayCluster, session_dict: Dict[str, SessionType]):
        self.cluster = cluster
        self.proxy_address = None
        self._session_dict = session_dict
        self._tileables_dict = {}
        self._server = None
        self.closed = False

    @classmethod
    async def create(cls, cluster: RayCluster, session_dict: Dict[str, SessionType] = None):
        proxy = SessionProxy(cluster, session_dict or dict())
        if aiohttp:
            await proxy.start()
        return proxy

    async def start(self):
        from mars.utils import get_next_port
        import socket
        host = socket.gethostbyname(socket.gethostname())
        port = get_next_port()
        self.proxy_address = f'http://{host}:{port}'
        logger.info('Starting session proxy on %s', self.proxy_address)

        def error_processor(func):

            @functools.wraps(func)
            async def _handler(request: 'web.Request'):
                try:
                    param = deserialize(*(cloudpickle.loads(await request.read())))
                    result = await func(param)
                    return web.Response(body=cloudpickle.dumps(serialize(result)), status=200)
                except Exception as e:
                    logger.warning(f'Execution {func} error: {e}')
                    import traceback
                    traceback.print_exc()
                    return web.Response(body=cloudpickle.dumps(serialize(e)), status=500)

            return _handler

        app = web.Application(client_max_size=1024 ** 3 * 32)
        app.add_routes([web.post('/destroy', error_processor(self.handle_destroy)),
                        web.post('/execute', error_processor(self.handle_execute)),
                        web.post('/fetch', error_processor(self.handle_fetch))])
        self._server: asyncio.Task = asyncio.create_task(web._run_app(app, host=host, port=port))

    def add_session(self, session_id, session):
        self._session_dict[session_id] = session

    async def _get_session(self, session_id):
        session = self._session_dict.get(session_id)
        if not session:
            session = await _new_session(self.cluster.supervisor_address, session_id=session_id,
                                         backend='oscar', default=True)
            self._session_dict[session_id] = session
        return session

    async def handle_destroy(self, param):
        session_id = param
        session = await self._get_session(session_id)
        await session.destroy()
        self._session_dict.pop(session_id)
        return web.Response()

    async def handle_execute(self, param):
        session_id, tileables, tileables_ids, kwargs = param
        self._tileables_dict.update(dict(zip(tileables_ids, tileables)))
        session = await self._get_session(session_id)
        execution_info = await session.execute(*tileables, **kwargs)
        await execution_info

    async def handle_fetch(self, param):
        session_id, tileables, tileables_ids = param
        tileables = [self._tileables_dict.get(_id, t) for t, _id in zip(tileables, tileables_ids)]
        session = await self._get_session(session_id)
        result = await session.fetch(*tileables)
        return result

    async def stop(self):
        if not self.closed:
            self.closed = True
            for session in self._session_dict.values():
                await session.destroy()
            self._session_dict = dict()
            self._server.cancel()


class _ExecutionInfo(ExecutionInfo):

    def progress(self) -> float:
        return 1.0


class RayClientSession(AbstractSession):

    def __init__(self, address: str, session_id: str = None):
        super().__init__(address, session_id or str(uuid.uuid4()))
        self.session = aiohttp.ClientSession(loop=asyncio.get_event_loop())
        self.proxy_address = address
        register_ray_serializers()

    @classmethod
    @implements(AbstractSession.init)
    async def init(cls, address: str, session_id: str = None, **kwargs) -> 'SessionType':
        return RayClientSession(address, session_id)

    async def destroy(self):
        async with self._post(f'destroy', self._session_id) as response:
            await RayClientSession._deserialize(response)
        await self.session.close()

    async def execute(self, *tileables, **kwargs) -> ExecutionInfo:
        tileables_ids = [id(t) for t in tileables]
        async with self._post(f'execute', self._session_id, tileables, tileables_ids, kwargs) as response:
            await RayClientSession._deserialize(response)
            future = asyncio.get_event_loop().create_future()
            future.set_result(None)
            return _ExecutionInfo(future)

    async def fetch(self, *tileables) -> list:
        async with self._post(f'fetch', self._session_id, tileables, [id(t) for t in tileables]) as response:
            return await RayClientSession._deserialize(response)

    def _post(self, endpoint: str, *args):
        data = cloudpickle.dumps(serialize(args))
        return self.session.post(f'{self.proxy_address}/{endpoint}', data=data)

    @staticmethod
    async def _deserialize(response):
        result = deserialize(*cloudpickle.loads(await response.read()))
        if response.status != 200:
            assert isinstance(result, Exception)
            raise result
        return result
