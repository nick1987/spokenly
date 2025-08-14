# Spokenly Latency Optimization Guide

## Overview

This guide explains the latency optimizations implemented in Spokenly to provide real-time transcription with minimal delay. The system now supports multiple latency profiles that you can choose from based on your specific needs.

## Key Optimizations Made

### 1. Backend Optimizations

#### Deepgram Configuration
- **Reduced utterance_end_ms**: From 5000ms to 1000ms (configurable)
- **Aggressive endpointing**: From 300ms to 150ms (configurable)
- **Enabled interim results**: More frequent partial transcriptions
- **Disabled unnecessary features**: Speaker diarization, alternatives, etc.
- **Optimized VAD**: Less aggressive voice activity detection

#### Audio Processing
- **Smaller audio buffers**: Reduced from 4096 to 1024 samples
- **Faster chunk processing**: Reduced chunk size from 40ms to 20ms
- **Optimized sample rate**: Maintained at 16kHz for best performance

### 2. Frontend Optimizations

#### Real-time Display
- **Immediate interim display**: Shows partial results as they come in
- **Reduced UI updates**: More efficient state management
- **Smaller audio chunks**: Faster processing and transmission

#### WebSocket Handling
- **Optimized message processing**: Faster parsing and display
- **Reduced buffer sizes**: Smaller audio chunks for lower latency

## Latency Profiles

The system now includes four pre-configured latency profiles:

### 1. Ultra-Low Latency
- **Description**: Fastest possible transcription with minimal accuracy trade-offs
- **Use Case**: Live presentations, real-time conversations
- **Settings**:
  - Utterance end: 500ms
  - Endpointing: 100ms
  - Audio buffer: 512 samples
  - Chunk size: 10ms

### 2. Low Latency (Default)
- **Description**: Good balance between speed and accuracy
- **Use Case**: General transcription, meetings, interviews
- **Settings**:
  - Utterance end: 1000ms
  - Endpointing: 150ms
  - Audio buffer: 1024 samples
  - Chunk size: 20ms

### 3. Standard
- **Description**: Balanced approach with good accuracy
- **Use Case**: Documentation, content creation
- **Settings**:
  - Utterance end: 2000ms
  - Endpointing: 300ms
  - Audio buffer: 2048 samples
  - Chunk size: 40ms

### 4. High Accuracy
- **Description**: Slower but more accurate transcription
- **Use Case**: Legal proceedings, medical documentation
- **Settings**:
  - Utterance end: 5000ms
  - Endpointing: 500ms
  - Audio buffer: 4096 samples
  - Chunk size: 50ms

## How to Use

### Via Frontend
1. Start the application
2. Look for the "Latency Profile" dropdown in the controls section
3. Select your preferred profile
4. The system will automatically adjust all settings

### Via API
```bash
# Get available profiles
curl http://localhost:8000/latency-profiles

# Set a specific profile
curl -X POST http://localhost:8000/set-latency-profile/ULTRA_LOW
```

## Environment Variables

You can also configure latency settings directly in the `.env` file:

```env
# Low Latency Configuration
ENABLE_INTERIM_RESULTS=true
UTTERANCE_END_MS=1000
ENDPOINTING=150
INTERIM_RESULTS_INTERVAL=100
VAD_EVENTS=true
VAD_AGGRESSIVENESS=1
```

## Performance Tips

### For Ultra-Low Latency:
1. Use a wired internet connection
2. Close other bandwidth-intensive applications
3. Use a high-quality microphone
4. Speak clearly and at a moderate pace

### For Best Accuracy:
1. Use the "High Accuracy" profile
2. Ensure quiet background environment
3. Speak at a normal pace with clear pronunciation
4. Use a high-quality microphone

## Troubleshooting

### High Latency Issues:
1. Check your internet connection speed
2. Try switching to a lower latency profile
3. Close other applications using the microphone
4. Check if your browser supports WebRTC

### Accuracy Issues:
1. Switch to a higher accuracy profile
2. Improve microphone quality
3. Reduce background noise
4. Speak more clearly and at a normal pace

## Technical Details

### Deepgram Parameters Explained:
- **utterance_end_ms**: How long to wait after silence before finalizing a transcript
- **endpointing**: How quickly to detect the end of speech
- **interim_results**: Whether to send partial results before finalizing
- **vad_events**: Voice activity detection events
- **vad_aggressiveness**: How sensitive the voice detection is (1=least, 3=most)

### Audio Processing:
- **Buffer size**: Smaller buffers = lower latency but more CPU usage
- **Chunk size**: Smaller chunks = faster transmission but more overhead
- **Sample rate**: 16kHz provides good quality with reasonable processing

## Future Improvements

Planned enhancements for even lower latency:
1. WebRTC optimization
2. Audio compression improvements
3. Parallel processing for multiple audio streams
4. Machine learning-based latency prediction
5. Adaptive latency based on network conditions
