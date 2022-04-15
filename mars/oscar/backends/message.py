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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import TracebackType
from typing import Any, Type, Tuple, Dict, List

from ...lib.tblib import pickling_support
from ...serialization.core import Serializer, pickle, buffered
from ...utils import classproperty, dataslots, implements, wrap_exception
from ..core import ActorRef

try:
    from random import randbytes
except ImportError:
    from random import getrandbits

    def randbytes(n: int) -> bytes:
        return getrandbits(n * 8).to_bytes(n, "little")


# make sure traceback can be pickled
pickling_support.install()


DEFAULT_PROTOCOL = 0


class MessageType(Enum):
    control = 0
    result = 1
    error = 2
    create_actor = 3
    destroy_actor = 4
    has_actor = 5
    actor_ref = 6
    send = 7
    tell = 8
    cancel = 9


class ControlMessageType(Enum):
    stop = 0
    restart = 1
    sync_config = 2
    get_config = 3
    wait_pool_recovered = 4
    add_sub_pool_actor = 5


@dataslots
@dataclass
class MessageTraceItem:
    uid: str
    address: str
    method: str


@dataslots
@dataclass
class ProfilingContext:
    task_id: str


class _MessageBase(ABC):
    __slots__ = "protocol", "message_id", "message_trace", "profiling_context"
    __non_pickle_slots__ = ()

    def __init__(
        self,
        message_id: bytes,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
        profiling_context: ProfilingContext = None,
    ):
        self.message_id = message_id
        if protocol is None:
            protocol = DEFAULT_PROTOCOL
        self.protocol = protocol
        # A message can be in the scope of other messages,
        # this is mainly used for detecting deadlocks,
        # e.g. Actor `A` sent a message(id: 1) to actor `B`,
        # in the processing of `B`, it sent back a message(id: 2) to `A`,
        # deadlock happens, because `A` is still waiting for reply from `B`.
        # In this case, the `scoped_message_ids` will be [1, 2],
        # `A` will find that id:1 already exists in inbox,
        # thus deadlock detected.
        self.message_trace = message_trace
        self.profiling_context = profiling_context

    @classproperty
    @abstractmethod
    def message_type(self) -> MessageType:
        """
        Message type.

        Returns
        -------
        message_type: MessageType
            message type.
        """

    def __repr__(self):
        slots = _get_slots(self.__class__, "__slot__")
        values = ", ".join(
            ["{}={!r}".format(slot, getattr(self, slot)) for slot in slots]
        )
        return "{}({})".format(self.__class__.__name__, values)


class ControlMessage(_MessageBase):
    __slots__ = "address", "control_message_type", "content"
    __non_pickle_slots__ = ("content",)

    def __init__(
        self,
        message_id: bytes,
        address: str,
        control_message_type: ControlMessageType,
        content: Any,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
    ):
        super().__init__(message_id, protocol=protocol, message_trace=message_trace)
        self.address = address
        self.control_message_type = control_message_type
        self.content = content

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.control


class ResultMessage(_MessageBase):
    __slots__ = ("result",)
    __non_pickle_slots__ = ("result",)

    def __init__(
        self,
        message_id: bytes,
        result: Any,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
        profiling_context: ProfilingContext = None,
    ):
        super().__init__(
            message_id,
            protocol=protocol,
            message_trace=message_trace,
            profiling_context=profiling_context,
        )
        self.result = result

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.result


class ErrorMessage(_MessageBase):
    __slots__ = "address", "pid", "error_type", "error", "traceback"

    # Check the as_instanceof_cause is not recursive.
    #
    # e.g. SubtaskRunnerActor.run_subtask will reraise the exception raised
    # from SubtaskProcessorActor.run. But these two actors are in the same
    # process, so we don't want to append duplicated address and pid in the
    # error message.
    class AsCauseBase:
        def __str__(self):
            return f"[address={self.address}, pid={self.pid}] {str(self.__wrapped__)}"

    def __init__(
        self,
        message_id: bytes,
        address: str,
        pid: int,
        error_type: Type[BaseException],
        error: BaseException,
        traceback: TracebackType,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
    ):
        super().__init__(message_id, protocol=protocol, message_trace=message_trace)
        self.address = address
        self.pid = pid
        self.error_type = error_type
        self.error = error
        self.traceback = traceback

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.error

    def as_instanceof_cause(self):
        if issubclass(self.error_type, ErrorMessage.AsCauseBase):
            return self.error.with_traceback(self.traceback)

        return wrap_exception(
            self.error,
            (ErrorMessage.AsCauseBase,),
            traceback=self.traceback,
            attr_dict=dict(address=self.address, pid=self.pid),
        )


class CreateActorMessage(_MessageBase):
    __slots__ = (
        "actor_cls",
        "actor_id",
        "args",
        "kwargs",
        "allocate_strategy",
        "from_main",
    )
    __non_pickle_slots__ = ("args", "kwargs")

    def __init__(
        self,
        message_id: bytes,
        actor_cls: Type,
        actor_id: bytes,
        args: Tuple,
        kwargs: Dict,
        allocate_strategy,
        from_main: bool = False,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
    ):
        super().__init__(message_id, protocol=protocol, message_trace=message_trace)
        self.actor_cls = actor_cls
        self.actor_id = actor_id
        self.args = args
        self.kwargs = kwargs
        self.allocate_strategy = allocate_strategy
        self.from_main = from_main

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.create_actor


class DestroyActorMessage(_MessageBase):
    __slots__ = "actor_ref", "from_main"

    def __init__(
        self,
        message_id: bytes,
        actor_ref: ActorRef,
        from_main: bool = False,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
    ):
        super().__init__(message_id, protocol=protocol, message_trace=message_trace)
        self.actor_ref = actor_ref
        self.from_main = from_main

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.destroy_actor


class HasActorMessage(_MessageBase):
    __slots__ = ("actor_ref",)

    def __init__(
        self,
        message_id: bytes,
        actor_ref: ActorRef,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
    ):
        super().__init__(message_id, protocol=protocol, message_trace=message_trace)
        self.actor_ref = actor_ref

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.has_actor


class ActorRefMessage(_MessageBase):
    __slots__ = ("actor_ref",)

    def __init__(
        self,
        message_id: bytes,
        actor_ref: ActorRef,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
    ):
        super().__init__(message_id, protocol=protocol, message_trace=message_trace)
        self.actor_ref = actor_ref

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.actor_ref


class SendMessage(_MessageBase):
    __slots__ = (
        "actor_ref",
        "content",
    )
    __non_pickle_slots__ = ("content",)

    def __init__(
        self,
        message_id: bytes,
        actor_ref: ActorRef,
        content: Any,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
        profiling_context: ProfilingContext = None,
    ):
        super().__init__(
            message_id,
            protocol=protocol,
            message_trace=message_trace,
            profiling_context=profiling_context,
        )
        self.actor_ref = actor_ref
        self.content = content

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.send


class TellMessage(SendMessage):
    __slots__ = ()

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.tell


class CancelMessage(_MessageBase):
    __slots__ = (
        "address",
        "cancel_message_id",
    )

    def __init__(
        self,
        message_id: bytes,
        address: str,
        cancel_message_id: bytes,
        protocol: int = None,
        message_trace: List[MessageTraceItem] = None,
    ):
        super().__init__(message_id, protocol=protocol, message_trace=message_trace)
        self.address = address
        self.cancel_message_id = cancel_message_id

    @classproperty
    @implements(_MessageBase.message_type)
    def message_type(self) -> MessageType:
        return MessageType.cancel


class DeserializeMessageFailed(Exception):
    def __init__(self, message_id):
        self.message_id = message_id

    def __str__(self):
        return f"Deserialize {self.message_id} failed"


class MessageSerializer(Serializer):
    serializer_name = "actor_message"

    @buffered
    def serialize(self, obj: _MessageBase, context: Dict):
        assert obj.protocol == 0, "only support protocol 0 for now"

        message_class = type(obj)
        pickle_slots = [getattr(obj, slot) for slot in _get_pickle_slots(message_class)]
        new_header = {
            b"msg_cls": message_class,
            b"msg_id": obj.message_id,
            b"pickles": pickle_slots,
        }
        non_pickle_slots = [
            getattr(obj, slot) for slot in _get_non_pickle_slots(message_class)
        ]
        if non_pickle_slots:
            header, buffers = yield non_pickle_slots
            new_header[b"non_pickles"] = header
        else:
            buffers = []
        return new_header, buffers

    def deserialize(self, header: Dict, buffers: List, context: Dict):
        message_id = header[b"msg_id"]
        message_class = header[b"msg_cls"]
        try:
            message = object.__new__(message_class)
            for slot, val in zip(_get_pickle_slots(message_class), header[b"pickles"]):
                setattr(message, slot, val)
            non_pickles_header = header.get(b"non_pickles")
            if non_pickles_header:
                non_pickles = yield non_pickles_header, buffers
                for slot, val in zip(_get_non_pickle_slots(message_class), non_pickles):
                    setattr(message, slot, val)
                print(f"message {message}")
            return message
        except pickle.UnpicklingError as e:  # pragma: no cover
            raise DeserializeMessageFailed(message_id) from e


# register message serializer
MessageSerializer.register(_MessageBase)


@lru_cache(20)
def _get_slots(message_cls: Type[_MessageBase], slot_name: str):
    slots = set()
    for tp in message_cls.__mro__:
        if issubclass(tp, _MessageBase):
            tp_slots = (
                getattr(tp, slot_name, []) if slot_name != "__slot__" else tp.__slots__
            )
            slots.update(tp_slots)
    return slots


@lru_cache(20)
def _get_pickle_slots(message_cls: Type[_MessageBase]):
    return sorted(
        _get_slots(message_cls, "__slot__")
        - _get_slots(message_cls, "__non_pickle_slots__")
    )


@lru_cache(20)
def _get_non_pickle_slots(message_cls: Type[_MessageBase]):
    return sorted(_get_slots(message_cls, "__non_pickle_slots__"))


def new_message_id():
    return randbytes(32)
