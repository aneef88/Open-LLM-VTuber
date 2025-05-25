from typing import Union, List, Dict, Any, Optional
import asyncio
import json
from loguru import logger
import numpy as np

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
) -> str:
    """Process a single-user conversation turn"""
    # Create TTSTaskManager for this conversation (backward-compatible toggle)
    tts_manager = TTSTaskManager(
        stream_tts=context.character_config.tts_config.stream
    )

    # If streaming mode is enabled, start the chunk pump and swap sender
    if tts_manager.stream_tts:
        handler = context.websocket_handler  # WebSocketHandler instance
        ws = handler.client_connections[client_uid]
        # Launch background task to stream chunks
        asyncio.create_task(
            handler.stream_audio_to_client(
                ws,
                tts_manager._payload_queue,
            )
        )
        # Bound send_fn: only needs payload argument
        send_fn = lambda payload: handler.websocket_send_payload(ws, payload)
    else:
        send_fn = websocket_send

    try:
        logger.debug("▶️ process_single_conversation starting…")
        # Send initial signals
        await send_conversation_start_signals(send_fn)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, send_fn
        )

        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
        )

        # Store user message (if history enabled)
        if context.history_uid:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
            )
        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        # Process agent response (this will queue TTS tasks)
        full_response = await process_agent_response(
            context=context,
            batch_input=batch_input,
            websocket_send=send_fn,
            tts_manager=tts_manager,
        )

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            await send_fn(json.dumps({"type": "backend-synth-complete"}))

        # Finalize turn (cleanup, etc.)
        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=send_fn,
            client_uid=client_uid,
        )

        # Store AI message (if history enabled)
        if context.history_uid and full_response:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            logger.info(f"AI response: {full_response}")

        return full_response

    except asyncio.CancelledError:
        logger.info(f"Conversation {session_emoji} cancelled due to interrupt.")
        raise

    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await send_fn(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise

    finally:
        cleanup_conversation(tts_manager, session_emoji)


async def process_agent_response(
    context: ServiceContext,
    batch_input: Any,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
) -> str:
    """Process agent response and generate TTS tasks"""
    full_response = ""
    try:
        # Stream LLM output, queueing TTS as we go
        agent_output = context.agent_engine.chat(batch_input)
        async for output in agent_output:
            part = await process_agent_output(
                output=output,
                character_config=context.character_config,
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=websocket_send,
                tts_manager=tts_manager,
                translate_engine=context.translate_engine,
            )
            full_response += part
    except Exception as e:
        logger.error(f"Error processing agent response: {e}")
        raise
    return full_response
