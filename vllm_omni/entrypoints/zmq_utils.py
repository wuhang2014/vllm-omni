# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ZMQ-based queue utilities for Omni IPC."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any

import zmq


@dataclass(frozen=True)
class ZmqQueueSpec:
    """Serializable socket spec for constructing a ZMQ queue in a worker."""

    endpoint: str
    socket_type: int
    bind: bool = False


class ZmqQueue:
    """Queue-like wrapper on a ZMQ socket."""

    def __init__(
        self,
        ctx: zmq.Context,
        socket_type: int,
        *,
        bind: str | None = None,
        connect: str | None = None,
        recv_timeout_ms: int | None = None,
        send_timeout_ms: int | None = None,
    ) -> None:
        self._socket = ctx.socket(socket_type)
        self._socket.linger = 0
        if recv_timeout_ms is not None:
            self._socket.rcvtimeo = recv_timeout_ms
        if send_timeout_ms is not None:
            self._socket.sndtimeo = send_timeout_ms
        if bind is not None:
            self._socket.bind(bind)
        if connect is not None:
            self._socket.connect(connect)

    def put(self, obj: Any) -> None:
        self._socket.send_pyobj(obj)

    def put_nowait(self, obj: Any) -> None:
        try:
            self._socket.send_pyobj(obj, flags=zmq.NOBLOCK)
        except zmq.Again as e:
            raise queue.Full() from e

    def get(self, timeout: float | None = None) -> Any:
        if timeout is None:
            return self._socket.recv_pyobj()
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        events = dict(poller.poll(int(timeout * 1000)))
        if events.get(self._socket) == zmq.POLLIN:
            return self._socket.recv_pyobj()
        raise queue.Empty()

    def get_nowait(self) -> Any:
        try:
            return self._socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again as e:
            raise queue.Empty() from e

    def empty(self) -> bool:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        events = dict(poller.poll(0))
        return events.get(self._socket) != zmq.POLLIN

    def close(self) -> None:
        try:
            self._socket.close(0)
        except Exception:
            pass


def create_zmq_queue(ctx: zmq.Context, spec: ZmqQueueSpec) -> ZmqQueue:
    """Create a ZmqQueue from a serialized spec."""

    if spec.bind:
        return ZmqQueue(ctx, spec.socket_type, bind=spec.endpoint)
    return ZmqQueue(ctx, spec.socket_type, connect=spec.endpoint)


def request_zmq_out_spec(
    master_endpoint: str,
    stage_id: int,
    *,
    timeout_ms: int = 30000,
) -> ZmqQueueSpec:
    """Request the output queue spec for a stage via the master handshake.

    Note: This function only returns out_spec for backward compatibility.
    Use request_zmq_specs() to get both in_spec and out_spec.
    """

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.linger = 0
    sock.rcvtimeo = int(timeout_ms)
    sock.sndtimeo = int(timeout_ms)
    sock.connect(master_endpoint)
    try:
        sock.send_pyobj({"type": "handshake", "stage_id": int(stage_id)})
        resp = sock.recv_pyobj()
    finally:
        try:
            sock.close(0)
        except Exception:
            pass

    if not isinstance(resp, dict):
        raise RuntimeError(f"Invalid handshake response: {type(resp)}")
    out_spec = resp.get("out_spec")
    if out_spec is None:
        raise RuntimeError(f"Handshake response missing out_spec: {resp}")
    if isinstance(out_spec, ZmqQueueSpec):
        return out_spec
    if isinstance(out_spec, dict):
        return ZmqQueueSpec(**out_spec)
    raise RuntimeError(f"Invalid out_spec type: {type(out_spec)}")


def request_zmq_specs(
    master_endpoint: str,
    stage_id: int,
    *,
    timeout_ms: int = 30000,
) -> tuple[ZmqQueueSpec, ZmqQueueSpec]:
    """Request both input and output queue specs for a stage via the master handshake.

    Returns:
        tuple[ZmqQueueSpec, ZmqQueueSpec]: A tuple of (in_spec, out_spec)
    """

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.linger = 0
    sock.rcvtimeo = int(timeout_ms)
    sock.sndtimeo = int(timeout_ms)
    sock.connect(master_endpoint)
    try:
        sock.send_pyobj({"type": "handshake", "stage_id": int(stage_id)})
        resp = sock.recv_pyobj()
    finally:
        try:
            sock.close(0)
        except Exception:
            pass

    if not isinstance(resp, dict):
        raise RuntimeError(f"Invalid handshake response: {type(resp)}")

    in_spec_data = resp.get("in_spec")
    out_spec_data = resp.get("out_spec")

    if in_spec_data is None or out_spec_data is None:
        raise RuntimeError(f"Handshake response missing specs: {resp}")

    # Convert to ZmqQueueSpec if needed
    if isinstance(in_spec_data, ZmqQueueSpec):
        in_spec = in_spec_data
    elif isinstance(in_spec_data, dict):
        in_spec = ZmqQueueSpec(**in_spec_data)
    else:
        raise RuntimeError(f"Invalid in_spec type: {type(in_spec_data)}")

    if isinstance(out_spec_data, ZmqQueueSpec):
        out_spec = out_spec_data
    elif isinstance(out_spec_data, dict):
        out_spec = ZmqQueueSpec(**out_spec_data)
    else:
        raise RuntimeError(f"Invalid out_spec type: {type(out_spec_data)}")

    return in_spec, out_spec
