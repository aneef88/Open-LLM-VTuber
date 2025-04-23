import requests
from loguru import logger
from .tts_interface import TTSInterface


class TTSEngine(TTSInterface):
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:7851/v1/audio/speech",
        model: str = "ignored",   # AllTalk requires it, but doesn't enforce naming
        voice: str = "nova",
        response_format: str = "wav",
        speed: float = 1.0,
    ):
        self.api_url = api_url
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.speed = speed
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
