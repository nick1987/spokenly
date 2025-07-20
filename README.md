# Spokenly - Real-Time Speech Recognition

A real-time speech-to-text application with live transcription capabilities using React frontend and FastAPI backend with Deepgram integration.

## 🚀 Features

- **Real-time Speech Recognition**: Live audio transcription using Deepgram API
- **WebSocket Communication**: Seamless real-time communication between frontend and backend
- **Audio Processing**: Raw PCM audio capture and processing
- **Live Transcript Display**: Real-time transcript updates in the UI
- **Audio Quality Monitoring**: Built-in audio amplitude detection and debugging
- **Cross-platform**: Works on any modern browser with microphone access

## 🏗️ Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐    HTTP API    ┌─────────────────┐
│   React Frontend │ ◄─────────────► │  FastAPI Backend │ ◄─────────────► │  Deepgram API   │
│                 │                 │                 │                 │                 │
│ • Audio Capture │                 │ • WebSocket     │                 │ • Speech-to-    │
│ • UI Display    │                 │   Server        │                 │   Text          │
│ • Real-time     │                 │ • Audio Process │                 │ • Real-time     │
│   Updates       │                 │ • API Gateway   │                 │   Streaming     │
└─────────────────┘                 └─────────────────┘                 └─────────────────┘
```

## 📁 Project Structure

```
spokenly/
├── spokenly-frontend/          # React frontend application
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   └── App.css            # Styles
│   ├── package.json           # Frontend dependencies
│   └── README.md              # Frontend documentation
├── spokenly-backend/           # FastAPI backend server
│   ├── main.py                # Main server application
│   ├── requirements.txt       # Python dependencies
│   ├── .env.example           # Environment variables template
│   └── README_ENHANCED.md     # Backend documentation
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🛠️ Technology Stack

### Frontend
- **React**: Modern UI framework
- **Web Audio API**: Raw audio capture
- **WebSocket**: Real-time communication
- **CSS3**: Modern styling

### Backend
- **FastAPI**: High-performance Python web framework
- **WebSocket**: Real-time bidirectional communication
- **Deepgram API**: Speech-to-text service
- **Pydantic**: Data validation and settings management

### Audio Processing
- **Raw PCM**: Linear16 format audio
- **16kHz Sample Rate**: Optimized for speech recognition
- **Mono Channel**: Single audio channel processing

## 🚀 Quick Start

### Prerequisites
- Node.js (v14 or higher)
- Python 3.9+
- Deepgram API key

### 1. Clone the Repository
```bash
git clone <repository-url>
cd spokenly
```

### 2. Setup Backend
```bash
cd spokenly-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your Deepgram API key
```

### 3. Setup Frontend
```bash
cd ../spokenly-frontend

# Install dependencies
npm install
```

### 4. Run the Application

#### Start Backend Server
```bash
cd spokenly-backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
```
Backend will be available at: `http://localhost:8000`

#### Start Frontend Development Server
```bash
cd spokenly-frontend
npm start
```
Frontend will be available at: `http://localhost:3000`

### 5. Use the Application
1. Open `http://localhost:3000` in your browser
2. Grant microphone permissions when prompted
3. Click "Start Recording" to begin live transcription
4. Speak clearly into your microphone
5. View real-time transcripts in the UI

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the `spokenly-backend` directory:

```env
# Deepgram API Configuration
DEEPGRAM_API_KEY=your_deepgram_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Audio Configuration
SAMPLE_RATE=16000
CHANNELS=1
CHUNK_SIZE=4096
```

### Audio Settings

The application is configured for optimal speech recognition:
- **Sample Rate**: 16kHz (optimal for speech)
- **Channels**: Mono (single channel)
- **Format**: Linear16 (raw PCM)
- **Chunk Size**: 4096 samples

## 🔧 Development

### Frontend Development
```bash
cd spokenly-frontend
npm start          # Start development server
npm run build      # Build for production
npm test           # Run tests
```

### Backend Development
```bash
cd spokenly-backend
source venv/bin/activate
python main.py     # Run development server
```

### API Endpoints

#### Health Check
- `GET /health` - Server health status

#### WebSocket
- `WS /ws/{session_id}` - Real-time audio streaming and transcription

## 🐛 Troubleshooting

### Common Issues

1. **"MediaRecorder is not defined"**
   - The application now uses Web Audio API instead of MediaRecorder
   - This error should not occur with the current implementation

2. **Silent Audio Detection**
   - Check microphone permissions
   - Ensure microphone is not muted
   - Speak louder or closer to the microphone
   - Check browser console for audio amplitude logs

3. **No Transcripts Appearing**
   - Verify Deepgram API key is correct
   - Check backend logs for connection errors
   - Ensure audio amplitude is sufficient (> 0.1)
   - Check network connectivity

4. **Sample Rate Issues**
   - Frontend is configured for 16kHz
   - Backend expects 16kHz audio
   - Check console logs for sample rate mismatches

### Debug Information

The application includes comprehensive logging:

#### Frontend Console
- Audio data amplitude and quality
- WebSocket connection status
- Audio processing errors

#### Backend Logs
- WebSocket connection events
- Audio processing statistics
- Deepgram API responses
- Error details and stack traces

## 📝 API Documentation

### WebSocket Message Format

#### Client to Server (Audio Data)
```json
{
  "type": "audio",
  "data": "base64_encoded_audio_data",
  "session_id": "unique_session_id"
}
```

#### Server to Client (Transcript)
```json
{
  "type": "transcript",
  "text": "recognized speech text",
  "confidence": 0.95,
  "is_final": true
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Deepgram](https://deepgram.com/) for speech recognition API
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) for the frontend framework 