import asyncio
import json
import logging
import os
import time
import numpy as np
from typing import Dict, Optional, Set, List, Tuple
from contextlib import asynccontextmanager

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic_settings import BaseSettings
from pydantic import Field
from starlette.websockets import WebSocketDisconnect

# Audio processing imports
try:
    import webrtcvad
    import librosa
    from scipy import signal
    from scipy.signal import butter, filtfilt
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Audio processing libraries not available. Install numpy, scipy, librosa, webrtcvad for enhanced audio processing.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    deepgram_api_key: str = Field(..., env="DEEPGRAM_API_KEY")
    deepgram_url: str = Field("wss://api.deepgram.com/v1/listen", env="DEEPGRAM_URL")
    deepgram_language: str = Field("en-US", env="DEEPGRAM_LANGUAGE")
    deepgram_model: str = Field("nova-2", env="DEEPGRAM_MODEL")  # Use Nova-2 for compatibility
    deepgram_tier: str = Field("", env="DEEPGRAM_TIER")  # No tier parameter for compatibility
    max_connections_per_user: int = Field(50, env="MAX_CONNECTIONS_PER_USER")
    connection_timeout: int = Field(30, env="CONNECTION_TIMEOUT")
    retry_attempts: int = Field(3, env="RETRY_ATTEMPTS")
    retry_delay: float = Field(1.0, env="RETRY_DELAY")
    # Audio processing settings
    sample_rate: int = Field(16000, env="SAMPLE_RATE")
    enable_vad: bool = Field(False, env="ENABLE_VAD")
    vad_aggressiveness: int = Field(3, env="VAD_AGGRESSIVENESS")
    enable_noise_reduction: bool = Field(False, env="ENABLE_NOISE_REDUCTION")
    enable_audio_enhancement: bool = Field(False, env="ENABLE_AUDIO_ENHANCEMENT")
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env file

settings = Settings()

class ConnectionManager:
    """Manages WebSocket connections and provides connection tracking."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self.session_transcripts: Dict[str, list] = {}  # Store transcripts per session
    
    async def connect(self, websocket: WebSocket, session_id: str = "anonymous", role: str = "recorder"):
        """Accept a new WebSocket connection and track it."""
        await websocket.accept()
        
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        
        # Check connection limits (allow more connections per session)
        if len(self.active_connections[session_id]) >= settings.max_connections_per_user * 2:
            await websocket.close(code=1008, reason="Too many connections")
            return False
        
        self.active_connections[session_id].add(websocket)
        self.connection_metadata[websocket] = {
            "session_id": session_id,
            "role": role,
            "connected_at": time.time(),
            "messages_sent": 0,
            "messages_received": 0
        }
        
        logger.info(f"New {role} connection for session {session_id}. Total connections: {len(self.active_connections[session_id])}")
        return True
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection from tracking."""
        metadata = self.connection_metadata.get(websocket)
        if metadata:
            session_id = metadata["session_id"]
            role = metadata["role"]
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
            self.connection_metadata[websocket]["messages_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.disconnect(websocket)
    
    async def broadcast_to_session(self, message: dict, session_id: str, exclude_websocket: WebSocket = None):
        """Send a message to all WebSocket connections for a specific session."""
        logger.info(f"Broadcasting to session {session_id}: {message}")
        
        if session_id in self.active_connections:
            connection_count = len(self.active_connections[session_id])
            logger.info(f"Found {connection_count} connections for session {session_id}")
            
            disconnected_websockets = []
            for websocket in self.active_connections[session_id]:
                # Skip the excluded websocket (usually the sender)
                if exclude_websocket and websocket == exclude_websocket:
                    logger.info(f"Skipping sender websocket for session {session_id}")
                    continue
                    
                try:
                    await websocket.send_json(message)
                    self.connection_metadata[websocket]["messages_sent"] += 1
                    logger.info(f"Successfully sent message to viewer for session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to broadcast message to session {session_id}: {e}")
                    disconnected_websockets.append(websocket)
            
            # Clean up disconnected websockets
            for websocket in disconnected_websockets:
                self.disconnect(websocket)
        else:
            logger.warning(f"No connections found for session {session_id}")
    
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
                logger.info(f"Sent {len(transcripts)} historical transcripts to viewer for session {session_id}")
    
    def add_transcript_to_session(self, session_id: str, transcript: dict):
        """Add a transcript to the session history."""
        if session_id not in self.session_transcripts:
            self.session_transcripts[session_id] = []
        
        self.session_transcripts[session_id].append(transcript)
        logger.info(f"Added transcript to session {session_id} history. Total: {len(self.session_transcripts[session_id])}")
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics for monitoring."""
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        return {
            "total_connections": total_connections,
            "sessions_active": len(self.active_connections),
            "connections_by_session": {session: len(connections) for session, connections in self.active_connections.items()},
            "sessions_with_transcripts": len(self.session_transcripts)
        }

class DeepgramService:
    """Handles communication with Deepgram's WebSocket API."""
    
    def __init__(self):
        self.connection_attempts = 0
    
    @asynccontextmanager
    async def get_connection(self, language: str = None):
        """Context manager for Deepgram WebSocket connection with retry logic."""
        connection = None
        try:
            for attempt in range(settings.retry_attempts):
                try:
                    # Get optimal parameters for the detected language
                    params = self._get_deepgram_params(language or settings.deepgram_language)
                    
                    # Build query string from parameters
                    query_params = "&".join([f"{k}={v}" for k, v in params.items() if v])
                    uri = f"{settings.deepgram_url}?{query_params}"
                    
                    headers = {"Authorization": f"Token {settings.deepgram_api_key}"}
                    
                    logger.info(f"Connecting to Deepgram with URI: {uri}")
                    connection = await websockets.connect(
                        uri, 
                        extra_headers=headers,
                        ping_interval=20,
                        ping_timeout=10
                    )
                    self.connection_attempts = 0
                    logger.info(f"Successfully connected to Deepgram with language: {params.get('language', 'unknown')}")
                    yield connection
                    break
                    
                except Exception as e:
                    self.connection_attempts += 1
                    logger.error(f"Deepgram connection attempt {attempt + 1} failed: {e}")
                    
                    if attempt < settings.retry_attempts - 1:
                        await asyncio.sleep(settings.retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error("All Deepgram connection attempts failed")
                        raise
        finally:
            if connection:
                await connection.close()
    
    def _get_deepgram_params(self, detected_language: str) -> Dict[str, str]:
        """Get optimal Deepgram parameters for the detected language."""
        base_params = {
            "model": settings.deepgram_model,
            "language": detected_language,
            "punctuate": "true",
            "interim_results": "true",
            "smart_format": "true",
            "encoding": "linear16",
            "channels": "1",
            "sample_rate": str(settings.sample_rate)
        }
        
        # Only add tier if it's not empty
        if settings.deepgram_tier:
            base_params["tier"] = settings.deepgram_tier
        
        return base_params

class AudioProcessor:
    """Handles audio processing and transcription logic with enhanced quality."""
    
    def __init__(self, deepgram_service: DeepgramService):
        self.deepgram_service = deepgram_service
        self.transcript_buffer = {}
        
        # Initialize VAD if available
        if AUDIO_PROCESSING_AVAILABLE and settings.enable_vad:
            try:
                self.vad = webrtcvad.Vad(settings.vad_aggressiveness)
                self.vad_enabled = True
            except Exception as e:
                logger.warning(f"Failed to initialize VAD: {e}")
                self.vad_enabled = False
        else:
            self.vad_enabled = False
        
        # Audio processing parameters
        self.frame_duration = 30  # ms
        self.frame_size = int(settings.sample_rate * self.frame_duration / 1000)
        self.audio_buffer = []
        self.silence_threshold = 0.01
        self.noise_reduction_strength = 0.1
        
        # Dialect and language support
        self.supported_languages = {
            "en-US": ["en-US", "en-GB", "en-AU", "en-IN", "en-NZ"],
            "es-ES": ["es-ES", "es-MX", "es-AR", "es-CO", "es-PE"],
            "fr-FR": ["fr-FR", "fr-CA", "fr-BE", "fr-CH"],
            "de-DE": ["de-DE", "de-AT", "de-CH"],
            "hi-IN": ["hi-IN", "en-IN"],  # Hindi with English code-switching
            "ar-SA": ["ar-SA", "ar-EG", "ar-MA", "ar-DZ"],
            "zh-CN": ["zh-CN", "zh-TW", "zh-HK"],
            "ja-JP": ["ja-JP"],
            "ko-KR": ["ko-KR"],
            "pt-BR": ["pt-BR", "pt-PT"],
            "ru-RU": ["ru-RU"],
            "it-IT": ["it-IT", "it-CH"],
            "nl-NL": ["nl-NL", "nl-BE"],
            "pl-PL": ["pl-PL"],
            "tr-TR": ["tr-TR"],
            "sv-SE": ["sv-SE"],
            "da-DK": ["da-DK"],
            "no-NO": ["no-NO"],
            "fi-FI": ["fi-FI"],
            "he-IL": ["he-IL"],
            "th-TH": ["th-TH"],
            "vi-VN": ["vi-VN"],
            "id-ID": ["id-ID"],
            "ms-MY": ["ms-MY"],
            "tl-PH": ["tl-PH"],
            "bn-IN": ["bn-IN"],
            "ta-IN": ["ta-IN"],
            "te-IN": ["te-IN"],
            "kn-IN": ["kn-IN"],
            "ml-IN": ["ml-IN"],
            "gu-IN": ["gu-IN"],
            "pa-IN": ["pa-IN"],
            "or-IN": ["or-IN"],
            "as-IN": ["as-IN"],
            "ne-NP": ["ne-NP"],
            "si-LK": ["si-LK"],
            "my-MM": ["my-MM"],
            "km-KH": ["km-KH"],
            "lo-LA": ["lo-LA"],
            "mn-MN": ["mn-MN"],
            "ka-GE": ["ka-GE"],
            "hy-AM": ["hy-AM"],
            "az-AZ": ["az-AZ"],
            "kk-KZ": ["kk-KZ"],
            "ky-KG": ["ky-KG"],
            "uz-UZ": ["uz-UZ"],
            "tk-TM": ["tk-TM"],
            "tg-TJ": ["tg-TJ"],
            "fa-IR": ["fa-IR"],
            "ps-AF": ["ps-AF"],
            "ur-PK": ["ur-PK"],
            "sd-PK": ["sd-PK"],
            "bn-BD": ["bn-BD"],
            "si-LK": ["si-LK"],
            "my-MM": ["my-MM"],
            "km-KH": ["km-KH"],
            "lo-LA": ["lo-LA"],
            "mn-MN": ["mn-MN"],
            "ka-GE": ["ka-GE"],
            "hy-AM": ["hy-AM"],
            "az-AZ": ["az-AZ"],
            "kk-KZ": ["kk-KZ"],
            "ky-KG": ["ky-KG"],
            "uz-UZ": ["uz-UZ"],
            "tk-TM": ["tk-TM"],
            "tg-TJ": ["tg-TJ"],
            "fa-IR": ["fa-IR"],
            "ps-AF": ["ps-AF"],
            "ur-PK": ["ur-PK"],
            "sd-PK": ["sd-PK"],
            "bn-BD": ["bn-BD"]
        }
    
    async def process_audio_stream(self, websocket: WebSocket, session_id: str):
        """Process incoming audio stream and return transcriptions."""
        # Detect language from initial audio samples
        detected_language = settings.deepgram_language
        language_detection_samples = []
        
        # Collect some audio samples for language detection
        try:
            for _ in range(10):  # Collect 10 audio frames for detection
                try:
                    audio_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.1)
                    language_detection_samples.append(audio_data)
                    if len(language_detection_samples) >= 5:  # Use first 5 frames
                        break
                except asyncio.TimeoutError:
                    break
        except Exception as e:
            logger.warning(f"Could not collect language detection samples: {e}")
        
        # Detect language if we have samples
        if language_detection_samples:
            try:
                combined_audio = b''.join(language_detection_samples)
                detected_language = self.detect_language_and_dialect(combined_audio)
                logger.info(f"Detected language: {detected_language}")
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
        
        async with self.deepgram_service.get_connection(detected_language) as dg_ws:
            # Create tasks for bidirectional communication
            send_task = asyncio.create_task(self._send_audio_to_deepgram(websocket, dg_ws, language_detection_samples))
            receive_task = asyncio.create_task(self._receive_transcriptions(websocket, dg_ws, session_id))
            
            try:
                # Wait for either task to complete (or fail)
                done, pending = await asyncio.wait(
                    [send_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                        
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                raise
    
    async def _send_audio_to_deepgram(self, websocket: WebSocket, dg_ws, language_detection_samples: List[bytes] = None):
        """Send audio data from client to Deepgram with enhanced processing."""
        try:
            # First, send the language detection samples if we have them
            if language_detection_samples:
                for audio_data in language_detection_samples:
                    processed_audio = self.process_audio_frame(audio_data)
                    if processed_audio:  # Only send if we have audio after processing
                        await dg_ws.send(processed_audio)
            
            # Then process the ongoing audio stream
            while True:
                try:
                    # Check if websocket is still connected before receiving
                    if websocket.client_state.value == 3:  # WebSocketState.DISCONNECTED
                        logger.info("WebSocket disconnected, stopping audio send")
                        break
                    
                    audio_data = await websocket.receive_bytes()
                    logger.info(f"Received audio data: {len(audio_data)} bytes")
                    
                    # Process the audio frame for better quality
                    processed_audio = self.process_audio_frame(audio_data)
                    
                    # Only send if we have audio after processing (VAD might remove silence)
                    if processed_audio:
                        # Debug: log audio data occasionally
                        if len(processed_audio) > 0 and len(processed_audio) % 100 == 0:  # Log more frequently
                            # Check if audio is all zeros (silence)
                            audio_array = np.frombuffer(processed_audio, dtype=np.int16)
                            max_amplitude = np.max(np.abs(audio_array))
                            mean_amplitude = np.mean(np.abs(audio_array))
                            logger.info(f"Audio debug - Size: {len(processed_audio)}, Max: {max_amplitude}, Mean: {mean_amplitude:.1f}, First 4 bytes: {processed_audio[:4].hex()}")
                            
                            # Check if audio is all zeros
                            if max_amplitude == 0:
                                logger.warning("⚠️ Audio is completely silent (all zeros)")
                            elif max_amplitude < 100:
                                logger.warning(f"⚠️ Audio amplitude too low: {max_amplitude} (should be > 100 for speech)")
                            else:
                                logger.info(f"✅ Audio has good amplitude: {max_amplitude}")
                        
                        logger.info(f"Sending {len(processed_audio)} bytes to Deepgram")
                        await dg_ws.send(processed_audio)
                    else:
                        logger.info("No audio data to send after processing")
                        
                except WebSocketDisconnect:
                    logger.info("Client disconnected during audio send")
                    break
                except RuntimeError as e:
                    if "disconnect message has been received" in str(e):
                        logger.info("WebSocket disconnect detected, stopping audio send")
                        break
                    else:
                        raise
                
        except Exception as e:
            logger.error(f"Error sending audio to Deepgram: {e}")
            raise
    
    async def _receive_transcriptions(self, websocket: WebSocket, dg_ws, session_id: str):
        """Receive transcriptions from Deepgram and send to client."""
        try:
            logger.info(f"Starting to receive transcriptions from Deepgram for session {session_id}")
            async for msg in dg_ws:
                try:
                    logger.info(f"Received message from Deepgram: {len(msg)} bytes")
                    response = json.loads(msg)
                    logger.info(f"Parsed Deepgram response: {response}")
                    
                    # Extract enhanced transcript from Deepgram response
                    transcript_data = self._extract_transcript(response)
                    if transcript_data:
                        logger.info(f"Extracted transcript data: {transcript_data}")
                        
                        # Always send both interim and final transcripts to the frontend
                        # Only add finalized transcripts to session history
                        if transcript_data["is_final"]:
                            manager.add_transcript_to_session(session_id, transcript_data)
                            logger.info(f"Added final transcript to session history: {transcript_data['text']}")

                        # Prepare transcript message for frontend
                        transcript_message = {
                            "type": "transcript",
                            "text": transcript_data["text"],
                            "speaker": transcript_data["speaker"],
                            "is_final": transcript_data["is_final"],
                            "confidence": transcript_data["confidence"],
                            "detected_language": transcript_data["detected_language"],
                            "topics": transcript_data["topics"],
                            "utterances": transcript_data["utterances"],
                            "timestamp": transcript_data["timestamp"]
                        }

                        # Debug: log every transcript sent to frontend
                        logger.info(f"[LIVE TRANSCRIPT] Session {session_id} | Final: {transcript_data['is_final']} | Text: '{transcript_data['text']}' | Confidence: {transcript_data['confidence']}")

                        try:
                            # Send to the original websocket (recorder)
                            await websocket.send_json(transcript_message)
                            logger.info(f"✅ Successfully sent transcript to recorder for session {session_id}")
                            
                            # Broadcast to other connections for the same session (excluding the sender)
                            await manager.broadcast_to_session(transcript_message, session_id, exclude_websocket=websocket)
                        except Exception as e:
                            logger.error(f"❌ Failed to send transcript to frontend: {e}")
                    else:
                        logger.info(f"No transcript data extracted from response: {response}")
                        # Log the full response for debugging
                        logger.debug(f"Full Deepgram response: {response}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Deepgram response: {e}")
                    logger.error(f"Raw message: {msg}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram connection closed")
        except Exception as e:
            logger.error(f"Error receiving transcriptions: {e}")
            raise
    
    def _extract_transcript(self, response: dict) -> Optional[dict]:
        """Extract enhanced transcript information from Deepgram response."""
        try:
            # Log all response types for debugging
            response_type = response.get("type")
            logger.info(f"Processing Deepgram response type: {response_type}")
            
            # Only process Results responses, skip metadata
            if response_type == "Metadata":
                logger.info("Skipping metadata response")
                return None
            
            # Handle Results responses
            if response_type == "Results":
                channel = response.get("channel", {})
                alternatives = channel.get("alternatives", [])
                
                if not alternatives:
                    logger.info("No alternatives found in response")
                    return None
                
                alternative = alternatives[0]
                transcript_text = alternative.get("transcript", "")
                
                # Allow empty transcripts for interim results
                if not transcript_text and not response.get("is_final", False):
                    logger.info("Empty interim transcript - allowing for continuation")
                    return None
                
                if not transcript_text.strip():
                    logger.info("Empty transcript text - skipping")
                    return None
            
            # Extract speaker information if available
            speaker = None
            if "words" in alternative and alternative["words"]:
                # Get speaker from first word
                first_word = alternative["words"][0]
                speaker = first_word.get("speaker", None)
            
            # Extract confidence score
            confidence = alternative.get("confidence", 0.0)
            
            # Extract language if detected
            detected_language = response.get("metadata", {}).get("language", None)
            
            # Extract topics if available
            topics = response.get("metadata", {}).get("topics", [])
            
            # Extract utterances if available
            utterances = response.get("metadata", {}).get("utterances", [])
            
            return {
                "text": transcript_text,
                "speaker": speaker,
                "confidence": confidence,
                "detected_language": detected_language,
                "topics": topics,
                "utterances": utterances,
                "is_final": response.get("is_final", False),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error extracting transcript: {e}")
            return None
    
    def get_user_transcript_history(self, user_id: str) -> list:
        """Get transcript history for a specific user."""
        return self.transcript_buffer.get(user_id, [])
    
    def process_audio_frame(self, audio_data: bytes) -> bytes:
        """Process audio frame - now receiving raw linear16 data from frontend."""
        if len(audio_data) == 0:
            return None
        
        # Frontend now sends raw linear16 data, so we can pass it through directly
        return audio_data

    def _apply_noise_reduction(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply spectral noise reduction to audio."""
        try:
            # Convert to float for processing
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Apply spectral subtraction
            # Calculate noise profile from first 0.5 seconds
            noise_samples = int(0.5 * settings.sample_rate)
            if len(audio_float) > noise_samples:
                noise_profile = np.mean(np.abs(audio_float[:noise_samples]))
                
                # Apply spectral subtraction
                fft = np.fft.fft(audio_float)
                magnitude = np.abs(fft)
                phase = np.angle(fft)
                
                # Subtract noise from magnitude
                cleaned_magnitude = np.maximum(magnitude - noise_profile * self.noise_reduction_strength, 0)
                
                # Reconstruct signal
                cleaned_fft = cleaned_magnitude * np.exp(1j * phase)
                cleaned_audio = np.real(np.fft.ifft(cleaned_fft))
                
                # Convert back to int16
                return (cleaned_audio * 32768.0).astype(np.int16)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return audio_array

    def _apply_audio_enhancement(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply audio enhancement filters with robust error handling."""
        try:
            audio_float = audio_array.astype(np.float32) / 32768.0
            nyquist = settings.sample_rate / 2
            # High-pass filter
            cutoff_hp = 80  # Hz
            norm_hp = cutoff_hp / nyquist
            if not (0 < norm_hp < 1):
                raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1 (highpass)")
            b, a = butter(4, norm_hp, btype='high')
            audio_float = filtfilt(b, a, audio_float)
            # Low-pass filter
            cutoff_lp = 4000  # Hz (reduced from 8000 to stay within valid range)
            norm_lp = cutoff_lp / nyquist
            if not (0 < norm_lp < 1):
                raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1 (lowpass)")
            b, a = butter(4, norm_lp, btype='low')
            audio_float = filtfilt(b, a, audio_float)
            # Compression
            threshold = 0.3
            ratio = 2.0
            audio_float = np.where(
                np.abs(audio_float) > threshold,
                np.sign(audio_float) * (threshold + (np.abs(audio_float) - threshold) / ratio),
                audio_float
            )
            return (audio_float * 32768.0).astype(np.int16)
        except Exception as e:
            logger.error(f"Error in audio enhancement: {e}")
            return audio_array
    
    def _apply_vad(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply Voice Activity Detection to remove silence."""
        try:
            # Process audio in frames
            frame_size = self.frame_size
            processed_frames = []
            
            for i in range(0, len(audio_array), frame_size):
                frame = audio_array[i:i + frame_size]
                if len(frame) == frame_size:
                    # Check if frame contains speech
                    is_speech = self.vad.is_speech(frame.tobytes(), settings.sample_rate)
                    if is_speech:
                        processed_frames.append(frame)
                else:
                    # Last frame, keep it
                    processed_frames.append(frame)
            
            if processed_frames:
                return np.concatenate(processed_frames)
            else:
                return np.array([], dtype=np.int16)
                
        except Exception as e:
            logger.error(f"Error in VAD: {e}")
            return audio_array
    
    def detect_language_and_dialect(self, audio_data: bytes) -> str:
        """Detect the most likely language and dialect from audio."""
        # This is a simplified implementation
        # In a production system, you might use a language detection service
        # For now, we'll return the configured language
        return settings.deepgram_language
    


# Initialize services
manager = ConnectionManager()
deepgram_service = DeepgramService()
audio_processor = AudioProcessor(deepgram_service)

# FastAPI app setup
app = FastAPI(
    title="Spokenly Backend",
    description="Real-time speech-to-text service with WebSocket support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Spokenly Backend is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check with connection stats."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "connections": manager.get_connection_stats(),
        "deepgram_connection_attempts": deepgram_service.connection_attempts
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, role: str = "recorder"):
    """Main WebSocket endpoint for audio streaming and transcription."""
    try:
        # Validate role parameter
        if role not in ["recorder", "viewer"]:
            await websocket.close(code=1008, reason="Invalid role")
            return
        
        # Connect and track the WebSocket
        if not await manager.connect(websocket, session_id, role):
            return
        
        # Send connection confirmation
        await manager.send_personal_message({
            "type": "connection_established",
            "session_id": session_id,
            "role": role,
            "timestamp": time.time()
        }, websocket)
        
        # For viewers, send historical transcripts first
        if role == "viewer":
            await manager.send_session_history(websocket, session_id)
        
        # Only process audio stream for recorders
        if role == "recorder":
            await audio_processor.process_audio_stream(websocket, session_id)
        else:
            # For viewers, keep the connection alive with proper error handling
            try:
                # Send a keep-alive ping every 30 seconds
                while True:
                    await asyncio.sleep(30)
                    try:
                        await websocket.ping()
                    except Exception:
                        # Connection is closed or error occurred
                        break
            except WebSocketDisconnect:
                logger.info(f"Viewer disconnected normally for session {session_id}")
            except Exception as e:
                logger.error(f"Error in viewer connection for session {session_id}: {e}")
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id} (role: {role})")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for session {session_id} (role: {role}): {e}")
        try:
            await manager.send_personal_message({
                "type": "error",
                "message": "An error occurred during processing"
            }, websocket)
        except:
            pass
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

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages and dialects."""
    return {
        "supported_languages": audio_processor.supported_languages,
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
        "deepgram_tier": settings.deepgram_tier,
        "audio_processing_available": AUDIO_PROCESSING_AVAILABLE
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
