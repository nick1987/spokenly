import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

// Global variable to ensure sessionId is only generated once
let globalSessionId = null;

// Speaker color mapping
const speakerColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'];

function getSpeakerColor(speakerId) {
  if (!speakerId) return '#666666'; // Default color for unknown speakers
  return speakerColors[speakerId % speakerColors.length];
}

// NEW: Floating Bubble Component
const FloatingBubble = ({ text, position, onClose, timestamp, confidence }) => {
  const bubbleRef = useRef(null);
  
  useEffect(() => {
    // Auto-remove bubble after 10 seconds
    const timer = setTimeout(() => {
      onClose();
    }, 10000);
    
    return () => clearTimeout(timer);
  }, [onClose]);
  
  return (
    <div 
      ref={bubbleRef}
      className="floating-bubble"
      style={{
        left: position.x,
        top: position.y,
        animation: 'bubbleFloat 0.5s ease-out'
      }}
    >
      <div className="bubble-header">
        <span className="bubble-timestamp">{timestamp}</span>
        <span className="bubble-confidence">{Math.round(confidence * 100)}%</span>
        <button className="bubble-close" onClick={onClose}>×</button>
      </div>
      <div className="bubble-content">{text}</div>
    </div>
  );
};

// NEW: Word counting utility
const countWords = (text) => {
  return text.trim().split(/\s+/).filter(word => word.length > 0).length;
};

// NEW: Generate random position for floating bubbles
const generateBubblePosition = () => {
  const margin = 50;
  const maxX = window.innerWidth - 300 - margin;
  const maxY = window.innerHeight - 150 - margin;
  
  return {
    x: Math.max(margin, Math.random() * maxX),
    y: Math.max(margin, Math.random() * maxY)
  };
};

function getInitialSessionId() {
  // If already generated, return the existing one
  if (globalSessionId) {
    return globalSessionId;
  }
  
  // If share link present, use its sessionId; otherwise, generate a new one
  const urlParams = new URLSearchParams(window.location.search);
  const shareData = urlParams.get('share');
  if (shareData) {
    try {
      const decodedData = JSON.parse(atob(shareData));
      if (decodedData.sessionId) {
        globalSessionId = decodedData.sessionId;
        return globalSessionId;
      }
    } catch (error) {
      console.error('Error parsing share data:', error);
    }
  }
  
  // Generate a shorter, more shareable session ID
  const newSessionId = 'session-' + Math.random().toString(36).substr(2, 9);
  globalSessionId = newSessionId;
  return newSessionId;
}

const App = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  // Transcript state: array of { text, isFinal }
  const [transcriptChunks, setTranscriptChunks] = useState([]);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [audioLevel, setAudioLevel] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordingStartTime, setRecordingStartTime] = useState(null);
  
  // Customizable options for sharing
  const [chunkSize, setChunkSize] = useState(40);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [autoScroll, setAutoScroll] = useState(true);
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  
  // NEW: Voice transcription mode features
  const [transcriptionMode, setTranscriptionMode] = useState(false);
  const [floatingBubbles, setFloatingBubbles] = useState([]);
  const [wordChunkSize, setWordChunkSize] = useState(15); // 10-20 words
  const [bubblePositions, setBubblePositions] = useState([]);
  const [currentWordCount, setCurrentWordCount] = useState(0);
  const [currentChunk, setCurrentChunk] = useState('');
  const [isTranscribingWhileCoding, setIsTranscribingWhileCoding] = useState(false);
  
  // Replace sessionId state with a ref
  const sessionIdRef = useRef(getInitialSessionId());
  const sessionId = sessionIdRef.current;
  
  // Determine if this is a viewer BEFORE any connections
  const isViewerRef = useRef(false);
  const [isViewer, setIsViewer] = useState(() => {
    // Check if this is a share link on initial load
    const urlParams = new URLSearchParams(window.location.search);
    const shareData = urlParams.get('share');
    if (shareData) {
      try {
        const decodedData = JSON.parse(atob(shareData));
        if (decodedData.sessionId) {
          // This is a viewer joining an existing session
          isViewerRef.current = true;
          return true;
        }
      } catch (error) {
        console.error('Error parsing share data:', error);
      }
    }
    return false;
  });
  
  const mediaRecorderRef = useRef(null);
  const websocketRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttemptsRef = useRef(0);
  const recordingTimerRef = useRef(null);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 1000;

  // Handle incoming WebSocket messages (chunked transcript logic)
  const handleWebSocketMessage = useCallback((data) => {
    switch (data.type) {
      case 'connection_established':
        break;
      case 'session_history':
        if (data.transcripts && data.transcripts.length > 0) {
          // Only add finalized transcripts from history
          setTranscriptChunks(
            data.transcripts
              .filter(t => t.is_final)
              .map(t => ({
                id: Date.now() + Math.random(),
                text: t.text,
                isFinal: t.is_final, // Use backend's is_final
                timestamp: new Date(t.timestamp * 1000).toLocaleTimeString(),
                speaker: t.speaker || null,
                confidence: t.confidence || 0,
                detectedLanguage: t.detected_language || null,
                topics: t.topics || [],
                speakerColor: getSpeakerColor(t.speaker)
              }))
          );
        }
        break;
      case 'transcript': {
        console.log('[WebSocket] Received transcript:', data);
        
        // NEW: Handle word-based chunking for transcription mode
        if (transcriptionMode && data.is_final) {
          const newText = data.text;
          const currentText = currentChunk + ' ' + newText;
          const wordCount = countWords(currentText);
          
          setCurrentChunk(currentText);
          setCurrentWordCount(wordCount);
          
          // Create floating bubble when word count reaches chunk size
          if (wordCount >= wordChunkSize) {
            const bubbleId = Date.now() + Math.random();
            const position = generateBubblePosition();
            
            setFloatingBubbles(prev => [...prev, {
              id: bubbleId,
              text: currentText.trim(),
              position,
              timestamp: new Date(data.timestamp * 1000).toLocaleTimeString(),
              confidence: data.confidence || 0
            }]);
            
            // Reset for next chunk
            setCurrentChunk('');
            setCurrentWordCount(0);
          }
        }
        
        setTranscriptChunks(prev => {
          const last = prev[prev.length - 1];
          const newChunk = {
            id: Date.now() + Math.random(),
            text: data.text,
            isFinal: data.is_final,
            timestamp: new Date(data.timestamp * 1000).toLocaleTimeString(),
            speaker: data.speaker || null,
            confidence: data.confidence || 0,
            detectedLanguage: data.detected_language || null,
            topics: data.topics || [],
            speakerColor: getSpeakerColor(data.speaker)
          };

          // Always show interim results as the last chunk
          if (!data.is_final) {
            if (last && !last.isFinal) {
              // Update the last interim chunk
              return [...prev.slice(0, -1), newChunk];
            } else {
              // Add new interim chunk
              return [...prev, newChunk];
            }
          } else {
            // Finalized: if last is interim, replace it with final
            if (last && !last.isFinal) {
              return [...prev.slice(0, -1), { ...newChunk, isFinal: true }];
            }
            // If last is already final and matches, skip (dedup)
            if (last && last.isFinal && last.text === data.text) {
              return prev;
            }
            // Otherwise, add as new finalized chunk
            return [...prev, { ...newChunk, isFinal: true }];
          }
        });
        break;
      }
      case 'error':
        setError(data.message);
        break;
      default:
        break;
    }
  }, [transcriptionMode, currentChunk, wordChunkSize]);

  // WebSocket connection management
  const connectWebSocket = useCallback((userId = null) => {
    // Prevent multiple simultaneous connection attempts
    if (websocketRef.current?.readyState === WebSocket.OPEN || 
        websocketRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Clear any existing reconnection timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close any existing connection first
    if (websocketRef.current) {
      websocketRef.current.close(1000, 'Reconnecting');
      websocketRef.current = null;
    }

    // Use provided userId or current sessionId
    const currentSessionId = userId || sessionId;
    const wsUrl = `ws://localhost:8000/ws/${currentSessionId}?role=${isViewer ? 'viewer' : 'recorder'}`;
    console.log('Connecting to WebSocket with session ID:', currentSessionId, 'URL:', wsUrl, 'isViewer:', isViewer);
    
    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connected successfully');
        setIsConnected(true);
        setConnectionStatus('connected');
        setError(null);
        reconnectAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setConnectionStatus('disconnected');
        
        // Don't reconnect if it's a "Too many connections" error or manual close
        if (event.code === 1008) {
          console.log('Too many connections error - not reconnecting');
          setConnectionStatus('connection limit reached');
          return;
        }
        
        // Only attempt reconnection if not manually closed and we're not already reconnecting
        if (event.code !== 1000 && 
            reconnectAttemptsRef.current < maxReconnectAttempts && 
            !reconnectTimeoutRef.current) {
          reconnectAttemptsRef.current++;
          setConnectionStatus(`reconnecting (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
          
          // Use exponential backoff to prevent rapid reconnections
          const delay = reconnectDelay * Math.pow(2, reconnectAttemptsRef.current - 1);
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connectWebSocket(userId);
          }, delay);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('WebSocket connection failed');
        setConnectionStatus('error');
      };

      websocketRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setError('Failed to create WebSocket connection');
      setConnectionStatus('error');
    }
  }, [sessionId, isViewer, handleWebSocketMessage]);

  const disconnectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (websocketRef.current) {
      websocketRef.current.close(1000, 'User initiated disconnect');
      websocketRef.current = null;
    }
    
    setIsConnected(false);
    setConnectionStatus('disconnected');
  }, []);

  // Audio recording and streaming
  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setRecordingStartTime(Date.now());
      setRecordingTime(0);
      
      // Request microphone access with optimal settings for accuracy
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000, // 16kHz for Deepgram compatibility
          channelCount: 1
        }
      });

      // Set up audio analysis for visualization
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);

      // Start audio level monitoring
      const updateAudioLevel = () => {
        if (analyserRef.current && isRecording) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
          setAudioLevel(average);
          
          animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
        }
      };

      // Use Web Audio API to get raw PCM data
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000 // Force 16kHz sample rate
      });
      const audioSource = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      
      processor.onaudioprocess = (event) => {
        if (websocketRef.current?.readyState === WebSocket.OPEN) {
          // Get raw PCM data from the audio buffer
          const inputBuffer = event.inputBuffer;
          const inputData = inputBuffer.getChannelData(0);
          
          // Convert float32 to int16 (linear16 format) with amplification
          const int16Array = new Int16Array(inputData.length);
          const amplification = 2.0; // Increase volume
          for (let i = 0; i < inputData.length; i++) {
            int16Array[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768 * amplification));
          }
          
          // Debug: log audio data occasionally
          if (Math.random() < 0.05) { // Log 5% of the time
            const maxValue = Math.max(...inputData);
            const minValue = Math.min(...inputData);
            const maxInt16 = Math.max(...int16Array);
            const minInt16 = Math.min(...int16Array);
            
            console.log('Audio data:', {
              samples: inputData.length,
              sampleRate: inputBuffer.sampleRate,
              maxValue: maxValue,
              minValue: minValue,
              maxInt16: maxInt16,
              minInt16: minInt16,
              bufferSize: int16Array.buffer.byteLength,
              hasAudio: maxValue > 0.01 || minValue < -0.01
            });
            
            if (maxValue < 0.01 && minValue > -0.01) {
              console.warn('⚠️ Audio appears to be silent');
            } else if (maxValue > 0.1 || minValue < -0.1) {
              console.log('✅ Good audio detected');
            }
          }
          
          // Send the raw audio data
          websocketRef.current.send(int16Array.buffer);
        }
      };
      
      audioSource.connect(processor);
      processor.connect(audioContext.destination);
      
      // Store references for cleanup
      mediaRecorderRef.current = {
        audioContext,
        audioSource,
        processor,
        stream
      };

      // Web Audio API is already processing audio automatically
      setIsRecording(true);
      updateAudioLevel();

      // Start recording timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

      // Ensure WebSocket is connected
      if (!isConnected) {
        connectWebSocket();
      }

    } catch (err) {
      console.error('Failed to start recording:', err);
      setError(err.message || 'Failed to access microphone');
    }
  }, [isConnected, connectWebSocket]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      // Clean up Web Audio API components
      if (mediaRecorderRef.current.processor) {
        mediaRecorderRef.current.processor.disconnect();
      }
      if (mediaRecorderRef.current.audioSource) {
        mediaRecorderRef.current.audioSource.disconnect();
      }
      if (mediaRecorderRef.current.audioContext) {
        mediaRecorderRef.current.audioContext.close();
      }
      if (mediaRecorderRef.current.stream) {
        mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }

    setIsRecording(false);
    setAudioLevel(0);
    setRecordingTime(0);
    setRecordingStartTime(null);
  }, []);

  const clearTranscripts = useCallback(() => {
    setTranscriptChunks([]);
  }, []);

  const copyTranscript = useCallback((text) => {
    navigator.clipboard.writeText(text).then(() => {
      // Could add a toast notification here
      console.log('Transcript copied to clipboard');
    });
  }, []);

  const generateShareLink = useCallback(() => {
    const shareData = {
      sessionId: sessionId,
      transcripts: transcriptChunks.map(t => ({
        text: t.text,
        timestamp: t.timestamp,
        speakerColor: t.speakerColor
      })),
      settings: {
        chunkSize,
        playbackSpeed,
        autoScroll
      },
      isLiveRecording: isRecording,
      recordingTime: recordingTime,
      timestamp: Date.now()
    };
    
    const encodedData = btoa(JSON.stringify(shareData));
    const shareUrl = `${window.location.origin}${window.location.pathname}?share=${encodedData}`;
    console.log('Generated share link with sessionId:', sessionId, 'URL:', shareUrl);
    return shareUrl;
  }, [sessionId, transcriptChunks, chunkSize, playbackSpeed, autoScroll, isRecording, recordingTime]);

  const copyShareLink = useCallback(() => {
    const url = generateShareLink();
    navigator.clipboard.writeText(url).then(() => {
      console.log('Share link copied to clipboard');
      setShareUrl(url);
      setShowShareModal(true);
    });
  }, [generateShareLink]);

  const loadSharedData = useCallback(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const shareData = urlParams.get('share');
    
    if (shareData) {
      try {
        const decodedData = JSON.parse(atob(shareData));
        
        if (decodedData.transcripts) {
          setTranscriptChunks(decodedData.transcripts.map(t => ({
            id: Date.now() + Math.random(),
            text: t.text,
            isFinal: true,
            timestamp: t.timestamp || new Date().toLocaleTimeString(),
            speakerColor: t.speakerColor || 'blue'
          })));
        }
        if (decodedData.settings) {
          setChunkSize(decodedData.settings.chunkSize || 40);
          setPlaybackSpeed(decodedData.settings.playbackSpeed || 1);
          setAutoScroll(decodedData.settings.autoScroll !== false);
        }
        
        // Show info if this was a live recording
        if (decodedData.isLiveRecording) {
          console.log('Loaded shared live recording session:', decodedData.sessionId);
          console.log('Current sessionId:', sessionId);
          console.log('isViewer already set to:', isViewer);
        }
      } catch (error) {
        console.error('Failed to load shared data:', error);
      }
    }
  }, [sessionId]);

  // Format recording time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording();
      disconnectWebSocket();
    };
  }, [stopRecording, disconnectWebSocket]);

  // Connect WebSocket on mount (for both recorder and viewer)
  useEffect(() => {
    console.log('WebSocket connection effect triggered, sessionId:', sessionId, 'isViewer:', isViewer);
    if (!websocketRef.current && !reconnectTimeoutRef.current) {
      connectWebSocket();
    }
    // eslint-disable-next-line
  }, [isViewer]);

  // Load shared data on mount
  useEffect(() => {
    loadSharedData();
  }, [loadSharedData]);

  // Connection health monitoring
  useEffect(() => {
    const healthCheck = setInterval(() => {
      if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
        // Connection is healthy
        setConnectionStatus('connected');
      } else if (websocketRef.current && websocketRef.current.readyState === WebSocket.CLOSED) {
        // Connection is closed, attempt to reconnect
        console.log('Connection health check: WebSocket is closed, attempting reconnect');
        if (!reconnectTimeoutRef.current) {
          connectWebSocket();
        }
      }
    }, 10000); // Check every 10 seconds

    return () => clearInterval(healthCheck);
  }, [connectWebSocket]);

  return (
    <div className="App">
      {/* Top Control Panel */}
      <div className="control-panel">
        <div className="control-left">
          {isViewer ? (
            <div className="viewer-status">
              <span className="live-indicator">🔴</span>
              <span>Live Viewing Session</span>
            </div>
          ) : (
            <>
              {isRecording ? (
                <button className="stop-button" onClick={stopRecording}>
                  <span className="stop-icon">⏹</span>
                  Stop Recording
                </button>
              ) : (
                <button 
                  className="start-button" 
                  onClick={startRecording}
                  disabled={!isConnected && connectionStatus !== 'reconnecting'}
                >
                  <span className="mic-icon">🎤</span>
                  Start Recording
                </button>
              )}
            </>
          )}
          
          {isRecording && (
            <div className="recording-progress">
              <label>Recording Progress</label>
              <div className="progress-bar">
                <div 
                  className="progress-fill"
                  style={{ width: `${(recordingTime / 60) * 100}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>

        <div className="control-right">
          <div className={`connection-status ${connectionStatus}`}>
            <span className="status-dot"></span>
            {connectionStatus === 'connected' ? 'Connected' : connectionStatus}
          </div>
          
          {isRecording && (
            <div className="recording-timer">
              {formatTime(recordingTime)} / 10:00
            </div>
          )}
        </div>
      </div>

      {/* NEW: Transcription Mode Controls */}
      <div className="transcription-mode-section">
        <div className="mode-toggle">
          <label className="mode-label">
            <input 
              type="checkbox" 
              checked={transcriptionMode}
              onChange={(e) => setTranscriptionMode(e.target.checked)}
              className="mode-checkbox"
            />
            <span className="mode-text">🎯 Voice Transcription Mode</span>
          </label>
        </div>
        
        {transcriptionMode && (
          <div className="transcription-options">
            <div className="control-group">
              <label>Word Chunk Size:</label>
              <div className="chunk-controls">
                <button 
                  className="control-btn"
                  onClick={() => setWordChunkSize(Math.max(10, wordChunkSize - 1))}
                  disabled={wordChunkSize <= 10}
                >
                  -
                </button>
                <span className="control-value">{wordChunkSize} words</span>
                <button 
                  className="control-btn"
                  onClick={() => setWordChunkSize(Math.min(20, wordChunkSize + 1))}
                  disabled={wordChunkSize >= 20}
                >
                  +
                </button>
              </div>
            </div>
            
            <div className="control-group">
              <label>Current Words:</label>
              <span className="word-counter">{currentWordCount}/{wordChunkSize}</span>
            </div>
            
            <div className="control-group">
              <label>Continue While Coding:</label>
              <button 
                className={`toggle-btn ${isTranscribingWhileCoding ? 'active' : ''}`}
                onClick={() => setIsTranscribingWhileCoding(!isTranscribingWhileCoding)}
              >
                {isTranscribingWhileCoding ? 'ON' : 'OFF'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Customizable Controls */}
      <div className="controls-section">
        <div className="control-group">
          <label>Chunk Size:</label>
          <div className="chunk-controls">
            <button 
              className="control-btn"
              onClick={() => setChunkSize(prev => Math.max(10, prev - 5))}
              disabled={chunkSize <= 10}
            >
              -
            </button>
            <span className="control-value">{chunkSize}</span>
            <button 
              className="control-btn"
              onClick={() => setChunkSize(prev => Math.min(100, prev + 5))}
              disabled={chunkSize >= 100}
            >
              +
            </button>
          </div>
        </div>
        
        <div className="control-group">
          <label>Speed:</label>
          <input
            type="range"
            min="0.5"
            max="2"
            step="0.1"
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
            className="speed-slider"
          />
          <span className="control-value">{playbackSpeed}x</span>
        </div>
        
        <div className="control-group">
          <label>Auto-scroll:</label>
          <button 
            className={`toggle-btn ${autoScroll ? 'active' : ''}`}
            onClick={() => setAutoScroll(!autoScroll)}
          >
            {autoScroll ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      {/* Share Button - Only show for recorders, not viewers */}
      {!isViewer && (
        <div className="share-section">
          <button 
            className="share-button" 
            onClick={copyShareLink}
          >
            <span className="copy-icon">📋</span>
            {isRecording ? 'Share Live Link' : 'Copy Share Link'}
          </button>
        </div>
      )}

      {/* Share Modal */}
      {showShareModal && (
        <div className="modal-overlay" onClick={() => setShowShareModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>{isRecording ? 'Live Share Link Generated!' : 'Share Link Generated!'}</h3>
            <p>
              {isRecording 
                ? 'Your live recording session is now shareable with real-time updates.' 
                : 'Your transcript has been shared with the following settings:'
              }
            </p>
            <div className="share-settings">
              <div>Chunk Size: {chunkSize}</div>
              <div>Speed: {playbackSpeed}x</div>
              <div>Auto-scroll: {autoScroll ? 'ON' : 'OFF'}</div>
              {isRecording && (
                <>
                  <div>Status: 🔴 Live Recording</div>
                  <div>Recording Time: {formatTime(recordingTime)}</div>
                </>
              )}
              {transcriptChunks.length > 0 && (
                <div>Transcripts: {transcriptChunks.length} entries</div>
              )}
            </div>
            <div className="share-url">
              <input 
                type="text" 
                value={shareUrl} 
                readOnly 
                className="url-input"
              />
              <button 
                className="copy-url-btn"
                onClick={() => navigator.clipboard.writeText(shareUrl)}
              >
                Copy
              </button>
            </div>
            <button 
              className="close-modal-btn"
              onClick={() => setShowShareModal(false)}
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* NEW: Floating Bubbles */}
      {transcriptionMode && floatingBubbles.map(bubble => (
        <FloatingBubble
          key={bubble.id}
          text={bubble.text}
          position={bubble.position}
          timestamp={bubble.timestamp}
          confidence={bubble.confidence}
          onClose={() => setFloatingBubbles(prev => prev.filter(b => b.id !== bubble.id))}
        />
      ))}

      {/* Error Display */}
      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {/* Transcripts Container */}
      <div className="transcripts-container">
        {transcriptChunks.length === 0 ? (
          <div className="empty-state">
            <p>Start recording to see live transcriptions...</p>
          </div>
        ) : (
          <div className="transcripts-list">
            {transcriptChunks.map((chunk, index) => (
              <div
                key={chunk.id}
                className={`transcript-card ${chunk.isFinal ? 'final' : 'interim'}`}
              >
                <div className="transcript-number">{index + 1}</div>
                <div className="transcript-content">
                  <div className="transcript-header">
                    <span 
                      className="speaker-dot" 
                      style={{ backgroundColor: chunk.speakerColor }}
                      title={chunk.speaker ? `Speaker ${chunk.speaker}` : 'Unknown Speaker'}
                    ></span>
                    <span className="transcript-text">{chunk.text}</span>
                    <button 
                      className="copy-transcript-btn"
                      onClick={() => copyTranscript(chunk.text)}
                      title="Copy transcript"
                    >
                      📋
                    </button>
                  </div>
                  <div className="transcript-meta">
                    <div className="transcript-timestamp">{chunk.timestamp}</div>
                    {chunk.speaker && (
                      <div className="speaker-info">Speaker {chunk.speaker}</div>
                    )}
                    {chunk.confidence > 0 && (
                      <div className="confidence-score">
                        Confidence: {Math.round(chunk.confidence * 100)}%
                      </div>
                    )}
                    {chunk.detectedLanguage && (
                      <div className="language-info">
                        Language: {chunk.detectedLanguage}
                      </div>
                    )}
                    {chunk.topics && chunk.topics.length > 0 && (
                      <div className="topics-info">
                        Topics: {chunk.topics.join(', ')}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
