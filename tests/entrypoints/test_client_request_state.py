"""Unit tests for vllm_omni/entrypoints/client_request_state.py"""
import asyncio

import pytest

from vllm_omni.entrypoints.client_request_state import ClientRequestState


class TestClientRequestState:
    """Tests for ClientRequestState class"""

    def test_init_with_default_queue(self):
        """Test initialization with default queue"""
        state = ClientRequestState("test-request-id")

        assert state.request_id == "test-request-id"
        assert state.stage_id is None
        assert isinstance(state.queue, asyncio.Queue)

    def test_init_with_custom_queue(self):
        """Test initialization with custom queue"""
        custom_queue = asyncio.Queue(maxsize=10)
        state = ClientRequestState("test-request-id", queue=custom_queue)

        assert state.request_id == "test-request-id"
        assert state.stage_id is None
        assert state.queue is custom_queue

    def test_stage_id_can_be_set(self):
        """Test that stage_id can be set after initialization"""
        state = ClientRequestState("test-request-id")
        assert state.stage_id is None

        state.stage_id = 5
        assert state.stage_id == 5

    @pytest.mark.asyncio
    async def test_queue_can_be_used_for_communication(self):
        """Test that the queue can be used for async communication"""
        state = ClientRequestState("test-request-id")

        # Put an item in the queue
        await state.queue.put("test_data")

        # Get the item from the queue
        result = await state.queue.get()
        assert result == "test_data"

    @pytest.mark.asyncio
    async def test_multiple_items_in_queue(self):
        """Test handling multiple items in the queue"""
        state = ClientRequestState("test-request-id")

        # Put multiple items
        await state.queue.put("item1")
        await state.queue.put("item2")
        await state.queue.put("item3")

        # Get items in order
        assert await state.queue.get() == "item1"
        assert await state.queue.get() == "item2"
        assert await state.queue.get() == "item3"

    def test_request_id_is_immutable_via_attribute(self):
        """Test that request_id attribute is set correctly"""
        state = ClientRequestState("original-id")
        assert state.request_id == "original-id"

        # While Python doesn't prevent reassignment, we verify the initial value
        # In production code, you might want to use @property to make it read-only

    @pytest.mark.asyncio
    async def test_queue_empty_behavior(self):
        """Test queue empty behavior"""
        state = ClientRequestState("test-request-id")

        # Queue should be empty initially
        assert state.queue.empty()

        # Put and get an item
        await state.queue.put("data")
        assert not state.queue.empty()

        await state.queue.get()
        assert state.queue.empty()

    def test_different_instances_have_separate_queues(self):
        """Test that different instances have separate queues"""
        state1 = ClientRequestState("request-1")
        state2 = ClientRequestState("request-2")

        assert state1.queue is not state2.queue
        assert state1.request_id != state2.request_id

    @pytest.mark.asyncio
    async def test_queue_size_unlimited_by_default(self):
        """Test that default queue has unlimited size"""
        state = ClientRequestState("test-request-id")

        # Should be able to put many items without blocking
        for i in range(1000):
            await state.queue.put(i)

        assert state.queue.qsize() == 1000

    def test_init_with_none_queue_creates_new_queue(self):
        """Test that passing None as queue creates a new queue"""
        state = ClientRequestState("test-request-id", queue=None)

        assert isinstance(state.queue, asyncio.Queue)
        assert state.request_id == "test-request-id"
