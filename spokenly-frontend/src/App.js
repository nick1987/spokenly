import React, { useState, useRef, useEffect, useCallback } from 'react';
import "./App.css"

// Global variable to ensure sessionId is only generated once
let globalSessionId = null;

// Speaker color mapping (currently unused but kept for potential future use)
const speakerColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'];

function getSpeakerColor(speakerId) {
  if (!speakerId) return '#666666'; // Default color for unknown speakers
  return speakerColors[speakerId % speakerColors.length];
}

function getInitialSessionId() {
  if (globalSessionId) {
    return globalSessionId;
  }

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

  const newSessionId = 'session-' + Math.random().toString(36).substr(2, 9);
  globalSessionId = newSessionId;
  return newSessionId;
}

const App = () => {
  // --- Component State ---
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [completedText, setCompletedText] = useState(''); // Holds the finalized paragraph


  // State to hold the history of all completed sentences/paragraphs.
  const [transcriptHistory, setTranscriptHistory] = useState([]);

  // State to hold the current sentence being actively transcribed.
  const [currentUtterance, setCurrentUtterance] = useState({ confirmed: '', interim: '' });
  const utteranceRef = useRef({ confirmed: '', interim: '' }); // Ref to avoid stale state in callbacks.

  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [audioLevel, setAudioLevel] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);

  // Settings State
  const [chunkSize, setChunkSize] = useState(40);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [autoScroll, setAutoScroll] = useState(true);

  // Share Modal State
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareUrl, setShareUrl] = useState('');

  // --- Refs ---
  const sessionIdRef = useRef(getInitialSessionId());
  const sessionId = sessionIdRef.current;

  const [isViewer, setIsViewer] = useState(() => {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.has('share');
  });

  const mediaRecorderRef = useRef(null);
  const websocketRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const recordingTimerRef = useRef(null);

  // --- Constants ---
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 1000;

  // --- WebSocket Message Handling ---
  const handleWebSocketMessage = useCallback((data) => {
    switch (data.type) {
      case 'transcript': {
        // This handles the live updates for the *current* sentence.
        if (data.is_final) {
          setCompletedText(prev => (prev ? prev + ' ' : '') + data.text)
          // The transcription service has finalized this part of the sentence.
          utteranceRef.current.confirmed += ` ${data.text}`;
          utteranceRef.current.interim = ''; // The interim prediction is now confirmed.

        } else {
          // This is a temporary, predictive part.
          const newPart = data.text.substring(utteranceRef.current.confirmed.length);
          utteranceRef.current.interim = newPart;
        }
        // Update the UI with the latest state of the current sentence.
        setCurrentUtterance({ ...utteranceRef.current });
        break;
      }

      case 'utterance_end': {
        // The speaker has paused, so the current sentence is considered complete.
        const finishedUtterance = utteranceRef.current.confirmed;

        if (finishedUtterance.trim()) {
          // Add the completed sentence to our history array.
          setTranscriptHistory(prevHistory => [...prevHistory, finishedUtterance]);
        }

        // Reset the utterance ref and state to prepare for the next sentence.
        const resetUtterance = { confirmed: '', interim: '' };
        utteranceRef.current = resetUtterance;
        setCurrentUtterance(resetUtterance);
        break;
      }

      case 'error':
        setError(data.message);
        break;

      default:
        break;
    }
  }, []); // Empty dependency array ensures this function is created only once.

  // --- WebSocket Connection Management ---
  const connectWebSocket = useCallback((userId = null) => {
    if (websocketRef.current?.readyState === WebSocket.OPEN || websocketRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }
    if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);

    const currentSessionId = userId || sessionId;
    const wsUrl = `ws://0.0.0.0:8000/ws/${currentSessionId}?role=${isViewer ? 'viewer' : 'recorder'}`;

    try {
      const ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        setIsConnected(true);
        setConnectionStatus('connected');
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
        setIsConnected(false);
        if (event.code === 1008) {
          setConnectionStatus('connection limit reached');
          return;
        }
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          setConnectionStatus(`reconnecting (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
          const delay = reconnectDelay * Math.pow(2, reconnectAttemptsRef.current - 1);
          reconnectTimeoutRef.current = setTimeout(() => connectWebSocket(userId), delay);
        } else {
          setConnectionStatus('disconnected');
        }
      };
      ws.onerror = (e) => {
        if( websocketRef.current?.readyState !== WebSocket.CONNECTING ){
          console.error(websocketRef.current?.readyState)
          setError('WebSocket connection failed');
          setConnectionStatus('error');
        }
      };
      websocketRef.current = ws;
    } catch (error) {
      setError('Failed to create WebSocket connection');
      setConnectionStatus('error');
    }
  }, [sessionId, isViewer, handleWebSocketMessage]);

  const disconnectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    websocketRef.current?.close(1000, 'User initiated disconnect');
  }, []);

  // --- Audio Recording & Streaming ---
  const startRecording = useCallback(async () => {
    try {
      setError(null);
      setRecordingTime(0);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: 16000, channelCount: 1 }
      });

      // Audio level visualization setup
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);

      const updateAudioLevel = () => {
        if (analyserRef.current && mediaRecorderRef.current) { // Check if still recording
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);
          const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
          setAudioLevel(average);
          animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
        }
      };

      // Audio processing for WebSocket
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      const audioSource = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (event) => {
        if (websocketRef.current?.readyState === WebSocket.OPEN) {
          const inputData = event.inputBuffer.getChannelData(0);
          const int16Array = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            int16Array[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
          }
          websocketRef.current.send(int16Array.buffer);
        }
      };

      audioSource.connect(processor);
      processor.connect(audioContext.destination);

      mediaRecorderRef.current = { audioContext, audioSource, processor, stream };
      setIsRecording(true);
      updateAudioLevel();

      recordingTimerRef.current = setInterval(() => setRecordingTime(prev => prev + 1), 1000);

      if (!isConnected) connectWebSocket();

    } catch (err) {
      setError(err.message || 'Failed to access microphone');
    }
  }, [isConnected, connectWebSocket]);

  const stopRecording = useCallback(() => {
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);

    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stream?.getTracks().forEach(track => track.stop());
      mediaRecorderRef.current.processor?.disconnect();
      mediaRecorderRef.current.audioSource?.disconnect();
      mediaRecorderRef.current.audioContext?.close();
      mediaRecorderRef.current = null;
    }

    audioContextRef.current?.close();

    setIsRecording(false);
    setAudioLevel(0);
    setRecordingTime(0);
  }, []);

  const copyTranscript = useCallback((text) => {
    navigator.clipboard.writeText(text).then(() => {
      // Could add a toast notification here
      console.log('Transcript copied to clipboard');
    });
  }, []);

  // --- Utility & Lifecycle ---
  const generateShareLink = useCallback(() => {
    const fullTranscript = [...transcriptHistory, currentUtterance.confirmed].join(' ');
    const shareData = {
      sessionId: sessionId,
      transcripts: [{ text: fullTranscript, timestamp: new Date().toLocaleTimeString() }],
      settings: { chunkSize, playbackSpeed, autoScroll },
    };
    const encodedData = btoa(JSON.stringify(shareData));
    return `${window.location.origin}${window.location.pathname}?share=${encodedData}`;
  }, [sessionId, transcriptHistory, currentUtterance, chunkSize, playbackSpeed, autoScroll]);

  const copyShareLink = useCallback(() => {
    const url = generateShareLink();
    navigator.clipboard.writeText(url).then(() => {
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
        if (decodedData.transcripts && decodedData.transcripts.length > 0) {
          // Simple load: split by sentences. A more robust solution might be needed.
          setTranscriptHistory(decodedData.transcripts[0].text.split('. ').filter(Boolean));
        }
        if (decodedData.settings) {
          setChunkSize(decodedData.settings.chunkSize || 40);
          setPlaybackSpeed(decodedData.settings.playbackSpeed || 1);
          setAutoScroll(decodedData.settings.autoScroll !== false);
        }
      } catch (error) {
        console.error('Failed to load shared data:', error);
      }
    }
  }, []);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleChunkChange = (amount) => {
    setChunkSize(prev => Math.max(10, Math.min(100, prev + amount)));
  }

  useEffect(() => {
    // Initial connection attempt
    connectWebSocket();
    // Load any shared data from URL
    loadSharedData();

    // Cleanup on unmount
    return () => {
      stopRecording();
      disconnectWebSocket();
    };
  }, [connectWebSocket, disconnectWebSocket, stopRecording, loadSharedData]);

  const hasContent = transcriptHistory.length > 0 || currentUtterance.confirmed || currentUtterance.interim;

  return (
    <div className="App">
      {/* <style>{AppStyles}</style> */}
      <header className="control-panel">
        <div className="control-left">
          {isViewer ? (
            <div className="viewer-status"><span className="live-indicator">🔴</span><span>Live Viewing</span></div>
          ) : (
            isRecording ? (
              <button className="stop-button" onClick={stopRecording}>⏹ Stop Recording</button>
            ) : (
              <button className="start-button" onClick={startRecording} disabled={connectionStatus === 'disconnected'}>🎤 Start Recording</button>
            )
          )}
        </div>
        <div className="control-right">
          <div className={`connection-status ${connectionStatus}`}><span className="status-dot"></span>{connectionStatus}</div>
          {isRecording && <div className="recording-timer">{formatTime(recordingTime)}</div>}
        </div>
      </header>

      {!isViewer && (
        <section className="controls-section">
          <div className="control-group">
            <label>Chunk Size (ms)</label>
            <div className="chunk-controls">
              <button className="control-btn" onClick={() => handleChunkChange(-10)} disabled={chunkSize <= 10}>-</button>
              <span className="control-value">{chunkSize}</span>
              <button className="control-btn" onClick={() => handleChunkChange(10)} disabled={chunkSize >= 100}>+</button>
            </div>
          </div>
          <div className="control-group">
            <label>Speed</label>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.1"
              value={playbackSpeed}
              className="speed-slider"
              onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
            />
            <span className="control-value">{playbackSpeed.toFixed(1)}x</span>
          </div>
          <div className="control-group">
            <label>Auto-Scroll</label>
            <button
              className={`toggle-btn ${autoScroll ? 'active' : ''}`}
              onClick={() => setAutoScroll(!autoScroll)}
            >
              {autoScroll ? 'On' : 'Off'}
            </button>
          </div>
        </section>
      )}

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

      {showShareModal && (
        <div className="modal-overlay" onClick={() => setShowShareModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3>Share Link Generated!</h3>
            <div className="share-url">
              <input type="text" value={shareUrl} readOnly />
              <button onClick={() => navigator.clipboard.writeText(shareUrl)}>Copy</button>
            </div>
            <button className="close-modal-btn" onClick={() => setShowShareModal(false)}>Close</button>
          </div>
        </div>
      )}

      {error && <div className="error-message">⚠️ {error}</div>}

      <main className="transcripts-container">
        {hasContent ? (
          <div className="transcripts-list">
            {/* Render the history of completed transcripts */}
            {transcriptHistory.map((text, index) => (
              <div
                key={index}
                className="transcript-card final"
              >
                <div className="transcript-number">{index + 1}</div>
                <div className="transcript-content">
                  <div className="transcript-header">
                    {/* NOTE: The speaker dot is omitted here.
              To re-add it, your `transcriptHistory` would need to contain speaker objects.
              e.g., <span className="speaker-dot" style={{ backgroundColor: item.speakerColor }}></span> 
            */}
                    <span className="transcript-text">{text}</span>
                    <button
                      className="copy-transcript-btn"
                      onClick={() => copyTranscript(text)}
                      title="Copy transcript"
                    >
                      📋
                    </button>
                  </div>
                  {/* NOTE: The metadata section (timestamps, confidence, etc.) is omitted.
            This data is not available in the new `transcriptHistory` array of strings.
            You would need to enrich the data structure to display it.
          */}
                </div>
              </div>
            ))}

            {/* Render the current, live transcript */}
            {(currentUtterance.confirmed || currentUtterance.interim) && (
              <div className="transcript-card interim">
                <div className="transcript-number">{transcriptHistory.length + 1}</div>
                <div className="transcript-content">
                  <div className="transcript-header">
                    <span className="transcript-text">
                      {/* This preserves your new logic for displaying confirmed vs. interim text */}
                      <span>{currentUtterance.confirmed ? (completedText ? " " : "") + currentUtterance.confirmed : ""}</span>
                      <span className="interim-text">{currentUtterance.interim}</span>
                      <span>&nbsp;</span>
                    </span>
                    <button
                      className="copy-transcript-btn"
                      onClick={() => copyTranscript(
                        `${currentUtterance.confirmed || ''} ${currentUtterance.interim || ''}`.trim()
                      )}
                      title="Copy transcript"
                    >
                      📋
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="empty-state">
            <p>Start recording to see live transcriptions...</p>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
