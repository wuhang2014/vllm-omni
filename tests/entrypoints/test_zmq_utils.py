# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ZMQ-based queue utilities."""

import queue
import time

import pytest
import zmq

from vllm_omni.entrypoints.zmq_utils import ZmqQueue, create_zmq_queue


@pytest.fixture
def zmq_context():
    """Provide a ZMQ context and clean it up after the test."""
    ctx = zmq.Context()
    yield ctx
    ctx.term()


@pytest.fixture
def unique_endpoint(tmp_path):
    """Generate a unique IPC endpoint path for each test."""
    counter = 0
    def _make_endpoint():
        nonlocal counter
        counter += 1
        return f"ipc://{tmp_path}/test_{counter}.ipc"
    return _make_endpoint


class TestZmqQueue:
    """Test suite for ZmqQueue class."""

    def test_init_with_bind(self, zmq_context, unique_endpoint):
        """Test ZmqQueue initialization with bind mode."""
        endpoint = unique_endpoint()
        queue_obj = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        
        assert queue_obj.endpoint == endpoint
        assert queue_obj._socket is not None
        assert queue_obj._poller is not None
        
        queue_obj.close()

    def test_init_with_connect(self, zmq_context, unique_endpoint):
        """Test ZmqQueue initialization with connect mode."""
        endpoint = unique_endpoint()
        # First create a server to bind
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        
        # Then create a client to connect
        client = ZmqQueue(zmq_context, zmq.PUSH, connect=endpoint)
        
        assert client.endpoint == endpoint
        assert client._socket is not None
        
        client.close()
        server.close()

    def test_init_without_bind_or_connect_raises_error(self, zmq_context):
        """Test that initialization without bind or connect raises ValueError."""
        with pytest.raises(ValueError, match="Either bind or connect must be specified"):
            ZmqQueue(zmq_context, zmq.PULL)

    def test_init_with_timeouts(self, zmq_context, unique_endpoint):
        """Test ZmqQueue initialization with timeout settings."""
        endpoint = unique_endpoint()
        recv_timeout = 1000  # 1 second
        send_timeout = 500   # 0.5 seconds
        
        queue_obj = ZmqQueue(
            zmq_context,
            zmq.PULL,
            bind=endpoint,
            recv_timeout_ms=recv_timeout,
            send_timeout_ms=send_timeout
        )
        
        assert queue_obj._default_recv_timeout == recv_timeout
        assert queue_obj._default_send_timeout == send_timeout
        assert queue_obj._socket.rcvtimeo == recv_timeout
        assert queue_obj._socket.sndtimeo == send_timeout
        
        queue_obj.close()

    def test_put_and_get(self, zmq_context, unique_endpoint):
        """Test basic put and get operations."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        client = ZmqQueue(zmq_context, zmq.PUSH, connect=endpoint)
        
        # Give sockets time to connect
        time.sleep(0.1)
        
        test_data = {"key": "value", "number": 42}
        client.put(test_data)
        
        # Give message time to arrive
        time.sleep(0.1)
        
        received = server.get()
        assert received == test_data
        
        client.close()
        server.close()

    def test_put_nowait(self, zmq_context, unique_endpoint):
        """Test put_nowait operation."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        client = ZmqQueue(zmq_context, zmq.PUSH, connect=endpoint)
        
        # Give sockets time to connect
        time.sleep(0.1)
        
        test_data = ["item1", "item2", "item3"]
        client.put_nowait(test_data)
        
        # Give message time to arrive
        time.sleep(0.1)
        
        received = server.get()
        assert received == test_data
        
        client.close()
        server.close()

    def test_get_with_timeout(self, zmq_context, unique_endpoint):
        """Test get operation with timeout."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        
        # Try to get from empty queue with timeout
        with pytest.raises(queue.Empty):
            server.get(timeout=0.1)
        
        server.close()

    def test_get_without_timeout(self, zmq_context, unique_endpoint):
        """Test get operation without timeout (blocking until data arrives)."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        client = ZmqQueue(zmq_context, zmq.PUSH, connect=endpoint)
        
        # Give sockets time to connect
        time.sleep(0.1)
        
        test_data = "test message"
        client.put(test_data)
        
        # Give message time to arrive
        time.sleep(0.1)
        
        # Get without timeout should return immediately when data is available
        received = server.get(timeout=None)
        assert received == test_data
        
        client.close()
        server.close()

    def test_get_nowait_empty_queue(self, zmq_context, unique_endpoint):
        """Test get_nowait on empty queue raises queue.Empty."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        
        with pytest.raises(queue.Empty):
            server.get_nowait()
        
        server.close()

    def test_get_nowait_with_data(self, zmq_context, unique_endpoint):
        """Test get_nowait operation when data is available."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        client = ZmqQueue(zmq_context, zmq.PUSH, connect=endpoint)
        
        # Give sockets time to connect
        time.sleep(0.1)
        
        test_data = 12345
        client.put(test_data)
        
        # Give message time to arrive
        time.sleep(0.1)
        
        received = server.get_nowait()
        assert received == test_data
        
        client.close()
        server.close()

    def test_empty_on_empty_queue(self, zmq_context, unique_endpoint):
        """Test empty() returns True on empty queue."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        
        assert server.empty() is True
        
        server.close()

    def test_empty_on_non_empty_queue(self, zmq_context, unique_endpoint):
        """Test empty() returns False when queue has data."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        client = ZmqQueue(zmq_context, zmq.PUSH, connect=endpoint)
        
        # Give sockets time to connect
        time.sleep(0.1)
        
        client.put("data")
        
        # Give message time to arrive
        time.sleep(0.1)
        
        assert server.empty() is False
        
        # Consume the message
        server.get_nowait()
        
        # Now it should be empty
        assert server.empty() is True
        
        client.close()
        server.close()

    def test_close(self, zmq_context, unique_endpoint):
        """Test close operation."""
        endpoint = unique_endpoint()
        queue_obj = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        
        # Close should not raise any exceptions
        queue_obj.close()
        
        # Calling close multiple times should be safe
        queue_obj.close()

    def test_multiple_messages(self, zmq_context, unique_endpoint):
        """Test sending and receiving multiple messages."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        client = ZmqQueue(zmq_context, zmq.PUSH, connect=endpoint)
        
        # Give sockets time to connect
        time.sleep(0.1)
        
        messages = [
            {"id": 1, "data": "first"},
            {"id": 2, "data": "second"},
            {"id": 3, "data": "third"},
        ]
        
        for msg in messages:
            client.put(msg)
        
        # Give messages time to arrive
        time.sleep(0.2)
        
        for expected_msg in messages:
            received = server.get_nowait()
            assert received == expected_msg
        
        client.close()
        server.close()

    def test_different_data_types(self, zmq_context, unique_endpoint):
        """Test that ZmqQueue can handle various Python data types."""
        endpoint = unique_endpoint()
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        client = ZmqQueue(zmq_context, zmq.PUSH, connect=endpoint)
        
        # Give sockets time to connect
        time.sleep(0.1)
        
        test_cases = [
            42,                           # int
            3.14,                         # float
            "string",                     # str
            [1, 2, 3],                   # list
            {"key": "value"},            # dict
            (1, 2, 3),                   # tuple
            None,                        # None
            True,                        # bool
        ]
        
        for data in test_cases:
            client.put(data)
        
        # Give messages time to arrive
        time.sleep(0.2)
        
        for expected_data in test_cases:
            received = server.get_nowait()
            assert received == expected_data
        
        client.close()
        server.close()


class TestCreateZmqQueue:
    """Test suite for create_zmq_queue helper function."""

    def test_create_zmq_queue(self, zmq_context, unique_endpoint):
        """Test create_zmq_queue helper function."""
        endpoint = unique_endpoint()
        # First create a server to bind
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        
        # Create client using helper function
        client = create_zmq_queue(zmq_context, endpoint, zmq.PUSH)
        
        assert client.endpoint == endpoint
        assert client._socket is not None
        
        # Test that it works
        time.sleep(0.1)
        client.put("test")
        time.sleep(0.1)
        
        received = server.get_nowait()
        assert received == "test"
        
        client.close()
        server.close()

    def test_create_zmq_queue_uses_connect_mode(self, zmq_context, unique_endpoint):
        """Test that create_zmq_queue uses connect mode by default."""
        endpoint = unique_endpoint()
        # First create a server to bind
        server = ZmqQueue(zmq_context, zmq.PULL, bind=endpoint)
        
        # Create client using helper function (should connect, not bind)
        client = create_zmq_queue(zmq_context, endpoint, zmq.PUSH)
        
        # Verify it can communicate (proves it connected successfully)
        time.sleep(0.1)
        test_data = "connection test"
        client.put(test_data)
        time.sleep(0.1)
        
        received = server.get_nowait()
        assert received == test_data
        
        client.close()
        server.close()
