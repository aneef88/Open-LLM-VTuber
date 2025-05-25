import requests
import aiohttp
import asyncio
import json
import time
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:7851/v1/audio/speech",
        streaming_api_url: str = "http://127.0.0.1:7851/api/tts-generate-streaming",
        model: str = "ignored",   # AllTalk requires it, but doesn't enforce naming
        voice: str = "nova",
        response_format: str = "wav",
        speed: float = 1.0,
        stream: bool = False,
    ):
        self.api_url = api_url
        self.streaming_api_url = streaming_api_url
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.speed = speed
        self.stream = stream
        self.new_audio_dir = "cache"
        self.file_extension = "wav"

    def generate_audio(self, text, file_name_no_ext=None):
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": self.response_format,
            "speed": self.speed,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)

            if response.status_code == 200:
                with open(file_name, "wb") as f:
                    f.write(response.content)
                return file_name
            else:
                logger.critical(
                    f"AllTalk-TTS: Failed to generate audio. Status: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.exception(f"AllTalk-TTS: Exception while generating audio: {e}")
            return None

    async def async_generate_audio_stream(self, text, file_name_no_ext=None):
        # Build a unique base name (no “.wav” here—the server will append it)
        base = file_name_no_ext or "stream"
        unique_name = f"{base}_{int(time.time())}"
        payload = {
            "text": text,
            "voice": self.voice,
            "language": "en",
            "output_file": unique_name,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.streaming_api_url, data=payload) as response:
                    if response.status == 200:
                        async for chunk in response.content.iter_chunked(1024):
                            logger.debug(f"[DEBUG] First 20 bytes: {chunk[:20]!r}")
                            yield chunk
                    else:
                        text_response = await response.text()
                        logger.error(f"AllTalk-TTS: Failed to generate audio. Status: {response.status} - {text_response}")
        except Exception as e:
            logger.exception(f"AllTalk-TTS: Exception while streaming audio: {e}")
