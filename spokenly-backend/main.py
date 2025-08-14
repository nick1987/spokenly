import asyncio
import json
import logging
import os
import time
import numpy as np
from typing import Dict, Optional, Set, List

# Deepgram SDK imports
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    DeepgramError,
)
# Corrected imports for response types for deepgram-sdk v4.x
from deepgram.clients.listen import (
    LiveResultResponse,
    UtteranceEndResponse,
)

from deepgram.clients.listen.v1.websocket import (
    MetadataResponse
)

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings
from pydantic import Field
import logging

# Firebase imports
from firebase_admin.auth import UserRecord
from firebase_config import get_current_user, verify_firebase_token

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages.

# Audio processing imports
try:
    import webrtcvad
    from scipy.signal import butter, filtfilt
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logging.warning("Audio processing libraries not available. Install numpy, scipy, webrtcvad for enhanced audio processing.")

# Import latency configuration
from latency_config import get_profile, list_profiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    deepgram_api_key: str = Field(..., env="DEEPGRAM_API_KEY")
    deepgram_language: str = Field("en-US", env="DEEPGRAM_LANGUAGE")
    deepgram_model: str = Field("nova-2", env="DEEPGRAM_MODEL")
    max_connections_per_user: int = Field(50, env="MAX_CONNECTIONS_PER_USER")
    connection_timeout: int = Field(30, env="CONNECTION_TIMEOUT")
    
    # Firebase settings
    firebase_service_account_path: str = Field("", env="FIREBASE_SERVICE_ACCOUNT_PATH")
    firebase_project_id: str = Field("", env="FIREBASE_PROJECT_ID")
    
    # Audio processing settings
    sample_rate: int = Field(16000, env="SAMPLE_RATE")
    enable_vad: bool = Field(True, env="ENABLE_VAD")
    vad_aggressiveness: int = Field(1, env="VAD_AGGRESSIVENESS")
    enable_noise_reduction: bool = Field(True, env="ENABLE_NOISE_REDUCTION")
    enable_audio_enhancement: bool = Field(True, env="ENABLE_AUDIO_ENHANCEMENT")
    
    # Ultra-low latency settings (always prioritize speed)
    enable_interim_results: bool = Field(True, env="ENABLE_INTERIM_RESULTS")
    utterance_end_ms: int = Field(500, env="UTTERANCE_END_MS")
    endpointing: int = Field(100, env="ENDPOINTING")
    vad_events: bool = Field(True, env="VAD_EVENTS")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Initialize Deepgram Client
# The SDK will automatically handle retries and connection management.
try:
    config = DeepgramClientOptions(
        verbose=logging.WARNING,  # Change to logging.DEBUG for more verbose output
        options={"keepalive": "true"}
    )
    deepgram: DeepgramClient = DeepgramClient(settings.deepgram_api_key, config)
except Exception as e:
    logger.error(f"Could not create Deepgram client: {e}")
    raise

class ConnectionManager:
    """Manages WebSocket connections and provides connection tracking."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self.session_transcripts: Dict[str, list] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of session_ids

    async def connect(self, websocket: WebSocket, session_id: str = "anonymous", role: str = "recorder", user_id: str = None):
        """Accept a new WebSocket connection and track it."""
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        
        if len(self.active_connections[session_id]) >= settings.max_connections_per_user * 2:
            await websocket.close(code=1008, reason="Too many connections")
            return False
        
        self.active_connections[session_id].add(websocket)
        self.connection_metadata[websocket] = {
            "session_id": session_id,
            "role": role,
            "user_id": user_id,
            "connected_at": time.time(),
        }
        
        # Track user sessions
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
        
        logger.info(f"New {role} connection for session {session_id} (user: {user_id or 'anonymous'}). Total connections: {len(self.active_connections[session_id])}")
        return True

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection from tracking."""
        metadata = self.connection_metadata.get(websocket)
        if metadata:
            session_id = metadata["session_id"]
            role = metadata["role"]
            user_id = metadata.get("user_id")
            if session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            
            del self.connection_metadata[websocket]
            logger.info(f"Connection closed for session {session_id} (role: {role})")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)

    async def broadcast_to_session(self, message: dict, session_id: str, exclude_websocket: WebSocket = None):
        """Send a message to all WebSocket connections for a specific session."""
        if session_id not in self.active_connections:
            logger.warning(f"No connections found for session {session_id}")
            return
            
        disconnected_websockets = []
        for websocket in self.active_connections[session_id]:
            logger.info( "Found WebSocket:" )
            if exclude_websocket and websocket == exclude_websocket:
                continue
                
            try:
                logger.info( f"Sending message : {json.dumps(message)}" )
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast message to session {session_id}: {e}")
                disconnected_websockets.append(websocket)
        
        for websocket in disconnected_websockets:
            self.disconnect(websocket)

    async def send_session_history(self, websocket: WebSocket, session_id: str):
        """Send all historical transcripts for a session to a new viewer."""
        if session_id in self.session_transcripts:
            transcripts = self.session_transcripts[session_id]
            if transcripts:
                history_message = {
                    "type": "session_history",
                    "transcripts": transcripts,
                    "session_id": session_id
                }
                await self.send_personal_message(history_message, websocket)

    def add_transcript_to_session(self, session_id: str, transcript: dict):
        """Add a transcript to the session history."""
        if session_id not in self.session_transcripts:
            self.session_transcripts[session_id] = []
        
        self.session_transcripts[session_id].append(transcript)

    def get_connection_stats(self) -> Dict:
        """Get connection statistics for monitoring."""
        return {
            "total_connections": sum(len(conns) for conns in self.active_connections.values()),
            "sessions_active": len(self.active_connections),
        }

class AudioProcessor:
    """Handles audio processing and transcription logic using Deepgram SDK."""
    
    def __init__(self):
        # This class is kept for structure, but processing is currently bypassed.
        pass

    async def process_audio_stream(self, websocket: WebSocket, session_id: str):
        """Process incoming audio stream using Deepgram SDK."""
        
        dg_connection = None
        try:
            # STEP 1: Create a Deepgram LiveTranscription connection with advanced options
            options = self._get_deepgram_options(settings.deepgram_language)
            dg_connection = deepgram.listen.asyncwebsocket.v("1")

            # STEP 2: Define event handlers
            async def on_open(cls, open, **kwargs):
                logger.info(f"Deepgram connection opened for session {session_id}.")

            async def on_message(cls,result: LiveResultResponse, **kwargs):
                transcript_data = self._extract_transcript(result)

                if transcript_data:
                    if transcript_data["is_final"]:
                        manager.add_transcript_to_session(session_id, transcript_data)
                    await manager.broadcast_to_session(transcript_data, session_id)

            async def on_metadata(cls, metadata: MetadataResponse, **kwargs):
                logger.info(f"Deepgram metadata received for session {session_id}: {metadata}")

            async def on_utterance_end(cls, utterance_end: UtteranceEndResponse, **kwargs):
                await manager.broadcast_to_session({"type": "utterance_end"}, session_id)

            async def on_error(cls, error: DeepgramError, **kwargs):
                await manager.broadcast_to_session({"type": "error", "message": str(error)}, session_id)

            async def on_close(cls, close, **kwargs):
                logger.info(f"Deepgram connection closed for session {session_id}.")

            # STEP 3: Register event handlers
            dg_connection.on(LiveTranscriptionEvents.Open, on_open)
            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            dg_connection.on(LiveTranscriptionEvents.Metadata, on_metadata)
            dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
            dg_connection.on(LiveTranscriptionEvents.Error, on_error)
            dg_connection.on(LiveTranscriptionEvents.Close, on_close)

            # STEP 4: Start the connection
            if not await dg_connection.start(options):
                logger.error(f"Failed to start Deepgram connection for session {session_id}")
                return

            # STEP 5: Process audio from the client WebSocket
            while True:
                try:
                    audio_data = await websocket.receive_bytes()
                    await dg_connection.send(audio_data)
                except WebSocketDisconnect:
                    logger.info(f"Client disconnected for session {session_id}.")
                    break
        
        except Exception as e:
            logger.error(f"Error during audio processing for session {session_id}: {e}")
        
        finally:
            # STEP 6: Cleanly close the Deepgram connection
            if dg_connection:
                await dg_connection.finish()
                logger.info(f"Finished Deepgram connection for session {session_id}.")

    def _get_deepgram_options(self, language: str) -> LiveOptions:
        """Create LiveOptions object with ultra-low latency settings."""
        return LiveOptions(
            model=settings.deepgram_model,
            language=settings.deepgram_language,
            punctuate=True,
            interim_results=True,
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            # Ultra-low latency optimizations - using conservative values
            utterance_end_ms=1000,
            endpointing=300,
        )

    def _extract_transcript(self, response: LiveResultResponse) -> Optional[dict]:
        """Extract enhanced transcript information from Deepgram SDK response."""
        # *** FIX: Only process 'Results' type messages ***
        logger.info( "in _extract_transcript" )

        if response.type != 'Results':
            return None
            
        if not response.channel or not response.channel.alternatives:
            return None

        alternative = response.channel.alternatives[0]
        transcript_text = alternative.transcript

        if not transcript_text.strip():
            return None

        return {
            "type": "transcript",
            "text": transcript_text,
            "is_final": response.is_final,
            "confidence": alternative.confidence,
            "timestamp": time.time()
        }

# Initialize services
manager = ConnectionManager()
audio_processor = AudioProcessor()

# FastAPI app setup
app = FastAPI(
    title="Spokenly Backend (SDK Version)",
    description="Real-time speech-to-text service using Deepgram Python SDK",
    version="3.1.2"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Spokenly Backend (SDK Version) is running"}

@app.get("/auth/me")
async def get_current_user_info(current_user: UserRecord = Depends(get_current_user)):
    """Get current authenticated user information."""
    return {
        "uid": current_user.uid,
        "email": current_user.email,
        "display_name": current_user.display_name,
        "photo_url": current_user.photo_url,
        "email_verified": current_user.email_verified,
        "created_at": current_user.user_metadata.creation_timestamp if current_user.user_metadata else None
    }

@app.get("/auth/verify")
async def verify_token(current_user: UserRecord = Depends(get_current_user)):
    """Verify if the current token is valid."""
    return {
        "valid": True,
        "user_id": current_user.uid,
        "email": current_user.email
    }

@app.get("/health")
async def health_check():
    """Detailed health check with connection stats."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "connections": manager.get_connection_stats(),
        "deepgram_status": "available"
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, role: str = "recorder", token: str = None):
    """Main WebSocket endpoint for audio streaming and transcription."""
    user_id = None
    
    # Verify authentication if token is provided
    if token:
        try:
            from firebase_admin import auth
            decoded_token = auth.verify_id_token(token)
            user_id = decoded_token['uid']
            logger.info(f"Authenticated user {user_id} connecting to session {session_id}")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            await websocket.close(code=4001, reason="Authentication failed")
            return
    
    if not await manager.connect(websocket, session_id, role, user_id):
        return

    try:
        await manager.send_personal_message({
            "type": "connection_established",
            "session_id": session_id,
            "role": role,
        }, websocket)

        if role == "viewer":
            await manager.send_session_history(websocket, session_id)
            while True:
                await websocket.receive_text()
        
        elif role == "recorder":
            await audio_processor.process_audio_stream(websocket, session_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id} (role: {role})")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for session {session_id}: {e}")
    finally:
        manager.disconnect(websocket)

@app.get("/transcripts/{session_id}")
async def get_session_transcripts(session_id: str):
    """Get transcript history for a specific session."""
    transcripts = manager.session_transcripts.get(session_id, [])
    return {
        "session_id": session_id,
        "transcript_count": len(transcripts),
        "transcripts": transcripts
    }

@app.get("/user/sessions")
async def get_user_sessions(current_user: UserRecord = Depends(get_current_user)):
    """Get all sessions for the authenticated user."""
    user_id = current_user.uid
    user_session_ids = manager.user_sessions.get(user_id, set())
    
    sessions = []
    for session_id in user_session_ids:
        transcripts = manager.session_transcripts.get(session_id, [])
        sessions.append({
            "session_id": session_id,
            "transcript_count": len(transcripts),
            "last_activity": transcripts[-1]["timestamp"] if transcripts else None
        })
    
    return {
        "user_id": user_id,
        "sessions": sessions
    }

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages and dialects."""
    return {
        "supported_languages": ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR", "ja-JP", "ko-KR", "zh-CN"],
        "default_language": settings.deepgram_language,
        "audio_processing_available": AUDIO_PROCESSING_AVAILABLE
    }

@app.get("/audio-settings")
async def get_audio_settings():
    """Get current audio processing settings."""
    return {
        "sample_rate": settings.sample_rate,
        "enable_vad": settings.enable_vad,
        "vad_aggressiveness": settings.vad_aggressiveness,
        "enable_noise_reduction": settings.enable_noise_reduction,
        "enable_audio_enhancement": settings.enable_audio_enhancement,
        "deepgram_model": settings.deepgram_model,
        "audio_processing_available": AUDIO_PROCESSING_AVAILABLE
    }

@app.get("/latency-mode")
async def get_latency_mode():
    """Get current latency mode information."""
    return {
        "mode": "ultra-low",
        "description": "Always prioritizes speed over accuracy",
        "settings": {
            "utterance_end_ms": settings.utterance_end_ms,
            "endpointing": settings.endpointing,
            "interim_results": settings.enable_interim_results,
            "vad_aggressiveness": settings.vad_aggressiveness
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
