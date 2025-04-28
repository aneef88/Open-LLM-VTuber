# tts_engines/dia_tts.py

import os
import time
import requests # Use the requests library for HTTP calls
import json     # For logging the payload
from typing import Optional # For type hinting

from loguru import logger

# --- Attempt to import the interface ---
try:
    # Assumes this file is in a 'tts_engines' directory relative to 'tts_interface'
    from .tts_interface import TTSInterface
except ImportError:
    logger.warning("TTSInterface not found in standard location. Check import path. Using placeholder.")
    # Basic Placeholder if the import fails (allows testing but lacks real functionality)
    class TTSInterface:
        def generate_audio(self, text: str, file_name_no_ext=None) -> str:
            raise NotImplementedError("Placeholder: generate_audio not implemented")
        def generate_cache_file_name(self, base_name: str, extension: str) -> str:
            # Simple cache implementation for placeholder
            cache_dir = "cache_dia_placeholder"
            if not os.path.exists(cache_dir): os.makedirs(cache_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            safe_base = "".join(c for c in base_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')[:50] or "audio"
            return os.path.join(cache_dir, f"{safe_base}_{timestamp}.{extension}")
        def remove_file(self, filepath: str, verbose: bool = True):
             if os.path.exists(filepath): os.remove(filepath)


class TTSEngine(TTSInterface):
    """
    TTS Engine implementation that connects to the Nari/Dia FastAPI server.
    """
    def __init__(self,
                 # --- Parameters received from factory based on new DiaTTSConfig ---
                 server_url: str,
                 prompt_id: Optional[str] = None,
                 max_new_tokens: Optional[int] = None,
                 cfg_scale: Optional[float] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 cfg_filter_top_k: Optional[int] = None,
                 speed_factor: Optional[float] = None,
                 seed: Optional[int] = None
                 # Removed Gradio-specific params: audio_prompt_path, output_dir, output_file_ext
                ):
        """
        Initializes the client for the Nari/Dia FastAPI TTS server.

        Args:
            server_url: The URL of the FastAPI /tts endpoint.
            prompt_id: The default prompt_id configured for this engine instance.
            **kwargs: Optional generation parameters to override server defaults.
        """
        logger.info("Initializing Nari/Dia FastAPI TTS Client...")

        if not server_url:
            logger.error("FATAL: 'server_url' must be provided for Dia FastAPI TTS engine.")
            raise ValueError("'server_url' is required.")

        self.server_url = server_url # Ensure endpoint path is correct

        # Store configured defaults/overrides
        self.config_prompt_id = prompt_id
        self.config_max_new_tokens = max_new_tokens
        self.config_cfg_scale = cfg_scale
        self.config_temperature = temperature
        self.config_top_p = top_p
        self.config_cfg_filter_top_k = cfg_filter_top_k
        self.config_speed_factor = speed_factor
        self.config_seed = seed

        logger.info(f"Nari/Dia FastAPI TTS Client configured.")
        logger.info(f"Server URL: {self.server_url}")
        logger.info(f"Configured Default Prompt ID: {self.config_prompt_id or 'None (use server default)'}")

    # Inherits generate_cache_file_name and remove_file from TTSInterface
    # (or uses placeholder if import failed)

    def generate_audio(self, text: str, file_name_no_ext: Optional[str] = None) -> Optional[str]:
        """
        Generates audio by calling the Nari/Dia FastAPI TTS server.

        Args:
            text: The text to synthesize.
            file_name_no_ext: Base name for the output file (timestamp added).

        Returns:
            The absolute path to the generated WAV file on success, None on failure.
        """
        if not text:
            logger.warning("generate_audio called with empty text. Skipping.")
            return None

        # --- Prepare Output Path ---
        if file_name_no_ext is None:
            # Create a default base name from the text
            file_name_no_ext = text.split('.')[0][:30]
        # Use the inherited helper to create a unique cache file path
        output_path = self.generate_cache_file_name(file_name_no_ext, "wav")
        logger.info(f"Requesting audio generation for text: '{text[:50]}...'")
        logger.debug(f"Target output path: {output_path}")

        # --- Construct API Payload ---
        payload = {"text": text}

        # Add parameters from config only if they are not None (i.e., explicitly set)
        if self.config_prompt_id is not None:
            payload["prompt_id"] = self.config_prompt_id
        if self.config_max_new_tokens is not None:
            payload["max_new_tokens"] = self.config_max_new_tokens
        if self.config_cfg_scale is not None:
            payload["cfg_scale"] = self.config_cfg_scale
        if self.config_temperature is not None:
            payload["temperature"] = self.config_temperature
        if self.config_top_p is not None:
            payload["top_p"] = self.config_top_p
        if self.config_cfg_filter_top_k is not None:
            payload["cfg_filter_top_k"] = self.config_cfg_filter_top_k
        if self.config_speed_factor is not None:
            payload["speed_factor"] = self.config_speed_factor
        if self.config_seed is not None:
            payload["seed"] = self.config_seed

        logger.debug(f"Sending payload: {json.dumps(payload)}")

        # --- Make HTTP POST Request ---
        try:
            # Set a reasonable timeout (adjust as needed)
            timeout_seconds = 300
            response = requests.post(
                self.server_url,
                headers={"Content-Type": "application/json", "Accept": "audio/wav"},
                json=payload,
                stream=True, # Get streaming response
                timeout=timeout_seconds
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # --- Process Successful Response ---
            logger.info(f"Received successful response (Status: {response.status_code}). Saving audio...")
            bytes_written = 0
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        bytes_written += len(chunk)

            if bytes_written > 0:
                logger.info(f"Successfully saved {bytes_written} bytes of audio to: {output_path}")
                return os.path.abspath(output_path) # Return the full path
            else:
                logger.error("Server returned 200 OK but the response body was empty.")
                self.remove_file(output_path) # Clean up empty file
                return None

        # --- Handle Errors ---
        except requests.exceptions.Timeout:
            logger.error(f"Request to Nari/Dia TTS server timed out after {timeout_seconds} seconds: {self.server_url}")
            return None
        except requests.exceptions.ConnectionError as e:
             logger.error(f"Could not connect to Nari/Dia TTS server at {self.server_url}: {e}")
             return None
        except requests.exceptions.HTTPError as e:
             logger.error(f"Nari/Dia TTS server returned an error status: {e.response.status_code}")
             # Try to log server's detail message
             try:
                 error_detail = e.response.json()
                 logger.error(f"Server error detail: {error_detail.get('detail', e.response.text)}")
             except json.JSONDecodeError:
                 logger.error(f"Server error response (non-JSON): {e.response.text[:500]}...")
             return None
        except requests.exceptions.RequestException as e:
            logger.error(f"An unexpected error occurred during the HTTP request: {e}")
            return None
        except IOError as e:
             logger.error(f"Failed to write received audio to file '{output_path}': {e}")
             return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Dia TTS generation: {e}", exc_info=True)
            # Clean up potentially partially written file on unexpected error
            self.remove_file(output_path)
            return None