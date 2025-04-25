import requests
import urllib.parse
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:7851/v1/audio/speech",
        stream_api_url: str = "http://127.0.0.1:7851/api/tts-generate-streaming",
        model: str = "ignored",   # AllTalk requires it, but doesn't enforce naming
        voice: str = "nova", # If streaming use eg "female_01.wav" if generating use eg "nova"
        language: str = "en",
        response_format: str = "wav",
        speed: float = 1.0,
        use_streaming: bool = False,
        **kwargs
    ):
        self.api_url = api_url
        self.stream_api_url = stream_api_url
        self.model = model
        self.voice = voice
        self.language = language
        self.response_format = response_format
        self.speed = speed
        self.use_streaming = use_streaming
        self.new_audio_dir = "cache"
        self.file_extension = "wav"

    async def async_generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Overrides default async behavior to return a stream URL directly
        or delegate to generate_audio synchronously for file-based mode.
        """
        logger.debug(f"[AllTalk] async_generate_audio() called")
        return self.generate_audio(text, file_name_no_ext)

    def generate_audio(self, text, file_name_no_ext=None):
        logger.debug(f"[AllTalk] use_streaming: {self.use_streaming}")
        logger.debug(f"[AllTalk] stream_api_url: {self.stream_api_url}")

        if self.use_streaming:
            return self.stream_audio(text)

        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.response_format,
            "speed": self.speed,
        }

        try:
            # Send POST request to the TTS API
            response = requests.post(self.api_url, json=payload, timeout=120)

            # Check if the request was successful
            if response.status_code == 200:
                # Save the audio content to a file
                with open(file_name, "wb") as f:
                    f.write(response.content)
                return file_name
            else:
                # Handle errors or unsuccessful requests
                logger.critical(
                    f"AllTalk-TTS: Failed to generate audio. Status: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.exception(f"AllTalk-TTS: Exception while generating audio: {e}")
            return None

    def stream_audio(self, text: str) -> str:
        encoded_text = urllib.parse.quote_plus(text)
        return (
            f"{self.stream_api_url}"
            f"?text={encoded_text}"
            f"&voice={self.voice}"
            f"&language={self.language}"
            f"&output_format={self.response_format}"
            f"&speed={self.speed}"
            f"&output_file=nul"
        )
        logger.debug(f"Streaming URL: {streaming_url}")
        return streaming_url
