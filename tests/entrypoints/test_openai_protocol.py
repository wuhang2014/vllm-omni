"""Unit tests for vllm_omni/entrypoints/openai/protocol modules"""
import numpy as np
import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol import (
    ImageData,
    ImageGenerationRequest,
    ImageGenerationResponse,
    OmniChatCompletionStreamResponse,
    ResponseFormat,
)
from vllm_omni.entrypoints.openai.protocol.audio import AudioResponse, CreateAudio, OpenAICreateSpeechRequest


class TestResponseFormat:
    """Tests for ResponseFormat enum"""

    def test_response_format_values(self):
        """Test ResponseFormat enum values"""
        assert ResponseFormat.B64_JSON == "b64_json"
        assert ResponseFormat.URL == "url"

    def test_response_format_is_enum(self):
        """Test ResponseFormat is a proper enum"""
        assert isinstance(ResponseFormat.B64_JSON, str)
        assert isinstance(ResponseFormat.URL, str)


class TestImageGenerationRequest:
    """Tests for ImageGenerationRequest model"""

    def test_minimal_valid_request(self):
        """Test creating request with minimal required fields"""
        request = ImageGenerationRequest(prompt="A beautiful sunset")

        assert request.prompt == "A beautiful sunset"
        assert request.n == 1
        assert request.response_format == ResponseFormat.B64_JSON
        assert request.model is None
        assert request.size is None

    def test_full_request_with_all_fields(self):
        """Test creating request with all fields"""
        request = ImageGenerationRequest(
            prompt="A cat in a hat",
            model="stable-diffusion",
            n=2,
            size="512x512",
            response_format=ResponseFormat.B64_JSON,
            user="user-123",
            negative_prompt="blurry, low quality",
            num_inference_steps=50,
            guidance_scale=7.5,
            true_cfg_scale=5.0,
            seed=42,
            vae_use_slicing=True,
            vae_use_tiling=False,
        )

        assert request.prompt == "A cat in a hat"
        assert request.model == "stable-diffusion"
        assert request.n == 2
        assert request.size == "512x512"
        assert request.negative_prompt == "blurry, low quality"
        assert request.num_inference_steps == 50
        assert request.guidance_scale == 7.5
        assert request.seed == 42

    def test_size_validation_accepts_valid_formats(self):
        """Test size validation accepts valid formats"""
        valid_sizes = ["512x512", "1024x1024", "768x512", "1920x1080"]

        for size in valid_sizes:
            request = ImageGenerationRequest(prompt="test", size=size)
            assert request.size == size

    def test_size_validation_rejects_invalid_formats(self):
        """Test size validation rejects invalid formats"""
        invalid_sizes = ["512", "512-512", "512 x 512", "abc", ""]

        for size in invalid_sizes:
            with pytest.raises(ValidationError):
                ImageGenerationRequest(prompt="test", size=size)

    def test_size_none_is_valid(self):
        """Test that size can be None"""
        request = ImageGenerationRequest(prompt="test", size=None)
        assert request.size is None

    def test_n_validation_range(self):
        """Test n parameter validation range"""
        # Valid range: 1-10
        request = ImageGenerationRequest(prompt="test", n=1)
        assert request.n == 1

        request = ImageGenerationRequest(prompt="test", n=10)
        assert request.n == 10

        # Invalid: less than 1
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", n=0)

        # Invalid: greater than 10
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", n=11)

    def test_response_format_validation_only_b64_json(self):
        """Test response_format only accepts b64_json"""
        # Valid: b64_json
        request = ImageGenerationRequest(prompt="test", response_format=ResponseFormat.B64_JSON)
        assert request.response_format == ResponseFormat.B64_JSON

        # Invalid: url
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", response_format=ResponseFormat.URL)

    def test_num_inference_steps_validation(self):
        """Test num_inference_steps validation"""
        # Valid range: 1-200
        request = ImageGenerationRequest(prompt="test", num_inference_steps=50)
        assert request.num_inference_steps == 50

        # Invalid: 0
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", num_inference_steps=0)

        # Invalid: > 200
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", num_inference_steps=201)

    def test_guidance_scale_validation(self):
        """Test guidance_scale validation"""
        # Valid range: 0.0-20.0
        request = ImageGenerationRequest(prompt="test", guidance_scale=7.5)
        assert request.guidance_scale == 7.5

        # Invalid: negative
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", guidance_scale=-1.0)

        # Invalid: > 20.0
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", guidance_scale=21.0)

    def test_true_cfg_scale_validation(self):
        """Test true_cfg_scale validation"""
        # Valid range: 0.0-20.0
        request = ImageGenerationRequest(prompt="test", true_cfg_scale=5.0)
        assert request.true_cfg_scale == 5.0

        # Invalid: negative
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", true_cfg_scale=-1.0)

        # Invalid: > 20.0
        with pytest.raises(ValidationError):
            ImageGenerationRequest(prompt="test", true_cfg_scale=21.0)

    def test_vae_flags_default_to_false(self):
        """Test VAE optimization flags default to False"""
        request = ImageGenerationRequest(prompt="test")

        assert request.vae_use_slicing is False
        assert request.vae_use_tiling is False


class TestImageData:
    """Tests for ImageData model"""

    def test_image_data_with_b64_json(self):
        """Test ImageData with b64_json"""
        data = ImageData(b64_json="base64encodedstring")

        assert data.b64_json == "base64encodedstring"
        assert data.url is None
        assert data.revised_prompt is None

    def test_image_data_with_url(self):
        """Test ImageData with URL"""
        data = ImageData(url="https://example.com/image.png")

        assert data.url == "https://example.com/image.png"
        assert data.b64_json is None

    def test_image_data_all_fields(self):
        """Test ImageData with all fields"""
        data = ImageData(
            b64_json="base64string",
            url="https://example.com/image.png",
            revised_prompt="A revised prompt",
        )

        assert data.b64_json == "base64string"
        assert data.url == "https://example.com/image.png"
        assert data.revised_prompt == "A revised prompt"

    def test_image_data_can_be_empty(self):
        """Test ImageData can be created with defaults"""
        data = ImageData()

        assert data.b64_json is None
        assert data.url is None
        assert data.revised_prompt is None


class TestImageGenerationResponse:
    """Tests for ImageGenerationResponse model"""

    def test_basic_response(self):
        """Test creating basic response"""
        data = [ImageData(b64_json="image1"), ImageData(b64_json="image2")]
        response = ImageGenerationResponse(created=1234567890, data=data)

        assert response.created == 1234567890
        assert len(response.data) == 2
        assert response.data[0].b64_json == "image1"
        assert response.data[1].b64_json == "image2"

    def test_empty_data_list(self):
        """Test response with empty data list"""
        response = ImageGenerationResponse(created=1234567890, data=[])

        assert response.created == 1234567890
        assert len(response.data) == 0

    def test_response_serialization(self):
        """Test response can be serialized to dict"""
        data = [ImageData(b64_json="image1")]
        response = ImageGenerationResponse(created=1234567890, data=data)

        response_dict = response.model_dump()
        assert response_dict["created"] == 1234567890
        assert len(response_dict["data"]) == 1


class TestOpenAICreateSpeechRequest:
    """Tests for OpenAICreateSpeechRequest model"""

    def test_minimal_request(self):
        """Test creating minimal speech request"""
        request = OpenAICreateSpeechRequest(input="Hello world", voice="alloy", response_format="wav")

        assert request.input == "Hello world"
        assert request.voice == "alloy"
        assert request.response_format == "wav"
        assert request.speed == 1.0
        assert request.stream_format == "audio"

    def test_all_voice_options(self):
        """Test all valid voice options"""
        # These values are defined as Literal type hints in OpenAICreateSpeechRequest
        # and must match the protocol definition
        valid_voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]

        for voice in valid_voices:
            request = OpenAICreateSpeechRequest(input="test", voice=voice, response_format="wav")
            assert request.voice == voice

    def test_invalid_voice_rejected(self):
        """Test invalid voice is rejected"""
        with pytest.raises(ValidationError):
            OpenAICreateSpeechRequest(input="test", voice="invalid_voice", response_format="wav")

    def test_all_response_formats(self):
        """Test all valid response formats"""
        # These values are defined as Literal type hints in OpenAICreateSpeechRequest
        # and must match the protocol definition
        valid_formats = ["wav", "pcm", "flac", "mp3", "aac", "opus"]

        for fmt in valid_formats:
            request = OpenAICreateSpeechRequest(input="test", voice="alloy", response_format=fmt)
            assert request.response_format == fmt

    def test_invalid_response_format_rejected(self):
        """Test invalid response format is rejected"""
        with pytest.raises(ValidationError):
            OpenAICreateSpeechRequest(input="test", voice="alloy", response_format="invalid")

    def test_speed_validation(self):
        """Test speed parameter validation"""
        # Valid range: 0.25-4.0
        request = OpenAICreateSpeechRequest(input="test", voice="alloy", response_format="wav", speed=1.5)
        assert request.speed == 1.5

        # Invalid: too slow
        with pytest.raises(ValidationError):
            OpenAICreateSpeechRequest(input="test", voice="alloy", response_format="wav", speed=0.1)

        # Invalid: too fast
        with pytest.raises(ValidationError):
            OpenAICreateSpeechRequest(input="test", voice="alloy", response_format="wav", speed=5.0)

    def test_sse_stream_format_rejected(self):
        """Test SSE stream format is rejected"""
        with pytest.raises(ValidationError, match="'sse' is not a supported stream_format"):
            OpenAICreateSpeechRequest(input="test", voice="alloy", response_format="wav", stream_format="sse")

    def test_audio_stream_format_accepted(self):
        """Test audio stream format is accepted"""
        request = OpenAICreateSpeechRequest(input="test", voice="alloy", response_format="wav", stream_format="audio")
        assert request.stream_format == "audio"

    def test_optional_fields(self):
        """Test optional fields"""
        request = OpenAICreateSpeechRequest(
            input="test", voice="alloy", response_format="wav", model="tts-1", instructions="Speak slowly"
        )

        assert request.model == "tts-1"
        assert request.instructions == "Speak slowly"


class TestCreateAudio:
    """Tests for CreateAudio model"""

    def test_basic_create_audio(self):
        """Test creating basic CreateAudio"""
        audio_array = np.array([0.1, 0.2, 0.3])
        audio = CreateAudio(audio_tensor=audio_array)

        assert isinstance(audio.audio_tensor, np.ndarray)
        np.testing.assert_array_equal(audio.audio_tensor, audio_array)
        assert audio.sample_rate == 24000
        assert audio.response_format == "wav"
        assert audio.speed == 1.0
        assert audio.stream_format == "audio"
        assert audio.base64_encode is True

    def test_create_audio_with_custom_values(self):
        """Test CreateAudio with custom values"""
        audio_array = np.array([0.1, 0.2, 0.3])
        audio = CreateAudio(
            audio_tensor=audio_array,
            sample_rate=16000,
            response_format="mp3",
            speed=1.5,
            stream_format="audio",
            base64_encode=False,
        )

        assert audio.sample_rate == 16000
        assert audio.response_format == "mp3"
        assert audio.speed == 1.5
        assert audio.base64_encode is False

    def test_numpy_array_allowed(self):
        """Test that numpy arrays are allowed (arbitrary_types_allowed)"""
        audio_array = np.random.rand(1000)
        audio = CreateAudio(audio_tensor=audio_array)

        assert isinstance(audio.audio_tensor, np.ndarray)
        assert len(audio.audio_tensor) == 1000


class TestAudioResponse:
    """Tests for AudioResponse model"""

    def test_audio_response_with_bytes(self):
        """Test AudioResponse with bytes"""
        audio_data = b"fake audio data"
        response = AudioResponse(audio_data=audio_data, media_type="audio/wav")

        assert response.audio_data == audio_data
        assert response.media_type == "audio/wav"

    def test_audio_response_with_string(self):
        """Test AudioResponse with base64 string"""
        audio_data = "base64encodedaudiostring"
        response = AudioResponse(audio_data=audio_data, media_type="audio/wav")

        assert response.audio_data == audio_data
        assert response.media_type == "audio/wav"

    def test_audio_response_different_media_types(self):
        """Test AudioResponse with different media types"""
        media_types = ["audio/wav", "audio/mp3", "audio/flac", "audio/ogg"]

        for media_type in media_types:
            response = AudioResponse(audio_data=b"data", media_type=media_type)
            assert response.media_type == media_type


class TestOmniChatCompletionStreamResponse:
    """Tests for OmniChatCompletionStreamResponse model"""

    def test_omni_chat_completion_has_modality_field(self):
        """Test that OmniChatCompletionStreamResponse has modality field"""
        # Note: This test verifies the class definition, not instantiation
        # since it inherits from ChatCompletionStreamResponse which has required fields
        assert hasattr(OmniChatCompletionStreamResponse, "modality")
        # Create instance with required fields from parent class
        # This is simplified - in practice you'd need all required fields
        response = OmniChatCompletionStreamResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[],
        )
        assert response.modality == "text"

    def test_omni_chat_completion_modality_default(self):
        """Test that modality defaults to 'text'"""
        response = OmniChatCompletionStreamResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[],
        )
        assert response.modality == "text"

    def test_omni_chat_completion_modality_can_be_set(self):
        """Test that modality can be set to custom value"""
        response = OmniChatCompletionStreamResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[],
            modality="audio",
        )
        assert response.modality == "audio"

    def test_omni_chat_completion_modality_can_be_none(self):
        """Test that modality can be None"""
        response = OmniChatCompletionStreamResponse(
            id="test-id",
            created=1234567890,
            model="test-model",
            choices=[],
            modality=None,
        )
        assert response.modality is None
