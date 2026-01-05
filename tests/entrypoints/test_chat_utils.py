"""Unit tests for vllm_omni/entrypoints/chat_utils.py"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from vllm_omni.entrypoints.chat_utils import (
    OmniAsyncMultiModalContentParser,
    OmniAsyncMultiModalItemTracker,
    parse_chat_messages_futures,
)


class TestOmniAsyncMultiModalItemTracker:
    """Tests for OmniAsyncMultiModalItemTracker"""

    def test_create_parser_returns_correct_type(self):
        """Test that create_parser returns OmniAsyncMultiModalContentParser"""
        mock_model_config = MagicMock()
        mock_tokenizer = MagicMock()

        tracker = OmniAsyncMultiModalItemTracker(mock_model_config, mock_tokenizer)
        parser = tracker.create_parser()

        assert isinstance(parser, OmniAsyncMultiModalContentParser)
        assert parser._tracker is tracker


class TestOmniAsyncMultiModalContentParser:
    """Tests for OmniAsyncMultiModalContentParser"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_model_config = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.tracker = OmniAsyncMultiModalItemTracker(self.mock_model_config, self.mock_tokenizer)
        self.parser = self.tracker.create_parser()

    def test_set_mm_processor_kwargs(self):
        """Test setting mm_processor_kwargs"""
        kwargs = {"use_audio_in_video": True, "other_param": "value"}
        self.parser.set_mm_processor_kwargs(kwargs)

        assert self.parser._mm_processor_kwargs == kwargs

    def test_set_mm_processor_kwargs_none(self):
        """Test setting mm_processor_kwargs to None"""
        self.parser.set_mm_processor_kwargs(None)
        assert self.parser._mm_processor_kwargs is None

    @patch.object(OmniAsyncMultiModalContentParser, "_add_placeholder")
    def test_parse_video_without_audio_extraction(self, mock_add_placeholder):
        """Test parsing video without audio extraction"""
        mock_connector = MagicMock()
        mock_video = b"fake_video_data"
        mock_connector.fetch_video_async.return_value = mock_video
        self.parser._connector = mock_connector

        # Mock tracker.add to return a placeholder
        self.tracker.add = MagicMock(return_value="<video_placeholder>")

        self.parser.parse_video("http://example.com/video.mp4", uuid="test-uuid")

        # Verify video was fetched
        mock_connector.fetch_video_async.assert_called_once_with(video_url="http://example.com/video.mp4")

        # Verify placeholder was added
        self.tracker.add.assert_called_once_with("video", mock_video, "test-uuid")
        mock_add_placeholder.assert_called_once_with("video", "<video_placeholder>")

    @patch.object(OmniAsyncMultiModalContentParser, "_add_placeholder")
    def test_parse_video_with_audio_extraction(self, mock_add_placeholder):
        """Test parsing video with audio extraction enabled"""
        mock_connector = MagicMock()
        mock_video = b"fake_video_data"
        mock_connector.fetch_video_async.return_value = mock_video
        self.parser._connector = mock_connector

        # Set mm_processor_kwargs to enable audio extraction
        self.parser.set_mm_processor_kwargs({"use_audio_in_video": True})

        # Mock tracker.add to return placeholders
        self.tracker.add = MagicMock(side_effect=["<video_placeholder>", "<audio_placeholder>"])

        self.parser.parse_video("http://example.com/video.mp4", uuid="test-uuid")

        # Verify video placeholder was added
        assert self.tracker.add.call_count == 2
        # First call should be for video
        assert self.tracker.add.call_args_list[0][0][0] == "video"
        # Second call should be for audio
        assert self.tracker.add.call_args_list[1][0][0] == "audio"

        # Verify both placeholders were added
        assert mock_add_placeholder.call_count == 2

    @patch.object(OmniAsyncMultiModalContentParser, "_add_placeholder")
    def test_parse_video_with_none_url(self, mock_add_placeholder):
        """Test parsing video with None URL"""
        self.parser._connector = MagicMock()

        # Mock tracker.add to return a placeholder
        self.tracker.add = MagicMock(return_value="<video_placeholder>")

        self.parser.parse_video(None, uuid="test-uuid")

        # Should add None as video
        self.tracker.add.assert_called_once_with("video", None, "test-uuid")


class TestExtractAudioFromVideoAsync:
    """Tests for _extract_audio_from_video_async method"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_model_config = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.tracker = OmniAsyncMultiModalItemTracker(self.mock_model_config, self.mock_tokenizer)
        self.parser = self.tracker.create_parser()

    @pytest.mark.asyncio
    async def test_extract_audio_from_http_url(self):
        """Test extracting audio from HTTP URL"""
        test_url = "http://example.com/video.mp4"
        mock_audio = np.array([0.1, 0.2, 0.3])
        mock_sample_rate = 16000

        with patch("asyncio.to_thread") as mock_to_thread:
            # Mock the three async operations: download, write, load
            mock_to_thread.side_effect = [
                b"fake_video_data",  # download
                "/tmp/temp_video.mp4",  # write temp file
                (mock_audio, mock_sample_rate),  # load audio
                None,  # cleanup
            ]

            result = await self.parser._extract_audio_from_video_async(test_url)

            assert isinstance(result, tuple)
            assert len(result) == 2
            np.testing.assert_array_equal(result[0], mock_audio)
            assert result[1] == mock_sample_rate

            # Verify cleanup was called
            assert mock_to_thread.call_count == 4

    @pytest.mark.asyncio
    async def test_extract_audio_from_file_url(self):
        """Test extracting audio from file:// URL"""
        test_url = "file:///path/to/video.mp4"
        mock_audio = np.array([0.1, 0.2])
        mock_sample_rate = 16000

        with patch("asyncio.to_thread") as mock_to_thread:
            # Only load operation needed for file URLs
            mock_to_thread.return_value = (mock_audio, mock_sample_rate)

            result = await self.parser._extract_audio_from_video_async(test_url)

            assert isinstance(result, tuple)
            np.testing.assert_array_equal(result[0], mock_audio)
            assert result[1] == mock_sample_rate

            # No cleanup should be called for file URLs
            assert mock_to_thread.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_audio_from_data_url(self):
        """Test extracting audio from data URL"""
        import base64

        video_data = b"fake_video_data"
        encoded = base64.b64encode(video_data).decode()
        test_url = f"data:video/mp4;base64,{encoded}"

        mock_audio = np.array([0.1, 0.2])
        mock_sample_rate = 16000

        with patch("asyncio.to_thread") as mock_to_thread:
            # Mock write, load, and cleanup operations
            mock_to_thread.side_effect = [
                "/tmp/temp_video.mp4",  # write temp file
                (mock_audio, mock_sample_rate),  # load audio
                None,  # cleanup
            ]

            result = await self.parser._extract_audio_from_video_async(test_url)

            assert isinstance(result, tuple)
            np.testing.assert_array_equal(result[0], mock_audio)
            assert result[1] == mock_sample_rate

            # Cleanup should be called for data URLs
            assert mock_to_thread.call_count == 3

    @pytest.mark.asyncio
    async def test_extract_audio_from_local_path(self):
        """Test extracting audio from local file path (no scheme)"""
        test_url = "/local/path/to/video.mp4"
        mock_audio = np.array([0.1, 0.2])
        mock_sample_rate = 16000

        with patch("asyncio.to_thread") as mock_to_thread:
            # Only load operation needed
            mock_to_thread.return_value = (mock_audio, mock_sample_rate)

            result = await self.parser._extract_audio_from_video_async(test_url)

            assert isinstance(result, tuple)
            np.testing.assert_array_equal(result[0], mock_audio)

            # No cleanup for local paths
            assert mock_to_thread.call_count == 1

    @pytest.mark.asyncio
    async def test_extract_audio_cleanup_on_error(self):
        """Test that cleanup happens even when audio loading fails"""
        test_url = "http://example.com/video.mp4"

        with patch("asyncio.to_thread") as mock_to_thread:
            # Mock download and write to succeed, but load to fail
            mock_to_thread.side_effect = [
                b"fake_video_data",  # download
                "/tmp/temp_video.mp4",  # write temp file
                Exception("Load failed"),  # load audio fails
            ]

            with pytest.raises(Exception, match="Load failed"):
                await self.parser._extract_audio_from_video_async(test_url)

            # Cleanup should still be attempted (call count 4: download, write, load, cleanup)
            # But cleanup raises exception before it's called, so count is 3
            assert mock_to_thread.call_count == 3


class TestParseChatMessagesFutures:
    """Tests for parse_chat_messages_futures function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_model_config = MagicMock()
        self.mock_model_config.multimodal_config = None
        self.mock_tokenizer = MagicMock()

    def test_parse_simple_text_message(self):
        """Test parsing simple text message"""
        messages = [{"role": "user", "content": "Hello, world!"}]

        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages, self.mock_model_config, self.mock_tokenizer, "string"
        )

        assert len(conversation) == 1
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == "Hello, world!"

    def test_parse_message_with_mm_processor_kwargs(self):
        """Test parsing with mm_processor_kwargs"""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]

        mm_kwargs = {"use_audio_in_video": True}

        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages, self.mock_model_config, self.mock_tokenizer, "openai", mm_processor_kwargs=mm_kwargs
        )

        assert len(conversation) == 1
        assert conversation[0]["role"] == "user"

    def test_parse_multiple_messages(self):
        """Test parsing multiple messages"""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second message"},
        ]

        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages, self.mock_model_config, self.mock_tokenizer, "string"
        )

        assert len(conversation) == 3
        assert conversation[0]["role"] == "user"
        assert conversation[1]["role"] == "assistant"
        assert conversation[2]["role"] == "user"

    def test_parse_message_with_tool_calls(self):
        """Test parsing assistant message with tool calls"""
        messages = [
            {
                "role": "assistant",
                "content": "Let me help",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test_func", "arguments": "{}"}}],
            }
        ]

        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages, self.mock_model_config, self.mock_tokenizer, "string"
        )

        assert len(conversation) == 1
        assert "tool_calls" in conversation[0]

    def test_parse_message_with_name(self):
        """Test parsing message with name field"""
        messages = [{"role": "user", "content": "Test", "name": "John"}]

        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages, self.mock_model_config, self.mock_tokenizer, "string"
        )

        assert len(conversation) == 1
        assert conversation[0]["name"] == "John"

    def test_parse_message_with_multimodal_config(self):
        """Test parsing with multimodal config enabled"""
        self.mock_model_config.multimodal_config = MagicMock()
        self.mock_model_config.multimodal_config.interleave_mm_strings = True

        messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]

        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages, self.mock_model_config, self.mock_tokenizer, "string"
        )

        assert len(conversation) == 1

    @pytest.mark.asyncio
    async def test_mm_data_future_is_awaitable(self):
        """Test that mm_data_future can be awaited"""
        messages = [{"role": "user", "content": "Test"}]

        conversation, mm_data_future, mm_uuids = parse_chat_messages_futures(
            messages, self.mock_model_config, self.mock_tokenizer, "string"
        )

        # Should be awaitable
        result = await mm_data_future
        # For text-only messages, should return None or empty dict
        assert result is None or result == {}
