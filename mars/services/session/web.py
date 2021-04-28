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

from mars.services.web.core import ServiceWebHandlerBase, ServiceWebAPIBase
from tornado.httpclient import AsyncHTTPClient
from .api import SessionAPI


class SessionWebHandler(ServiceWebHandlerBase):
    _api_cls = SessionAPI

    async def create(self, address: str, **kwargs):
        return self._api_registry.add_instance(await SessionAPI.create(address, **kwargs))


_service_name = 'session'
web_handlers = {
    f'/api/service/{_service_name}/rpc': SessionWebHandler,
}


class SessionWebAPI(ServiceWebAPIBase):
    _service_name = _service_name

    @classmethod
    async def create(cls, address: str, **kwargs):
        http_client = AsyncHTTPClient()
        api_id = await cls._post(http_client, {}, f'create',  None, address, **kwargs)
        return SessionWebAPI(http_client, api_id)