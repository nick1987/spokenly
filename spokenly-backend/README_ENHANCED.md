# Spokenly Enhanced Backend - High-Quality Transcription with Multi-Dialect Support

## 🚀 Enhanced Features

### Audio Quality Improvements
- **Noise Reduction**: Spectral subtraction to remove background noise
- **Audio Enhancement**: High-pass and low-pass filtering for cleaner audio
- **Voice Activity Detection (VAD)**: Removes silence to improve transcription accuracy
- **Audio Compression**: Dynamic range compression for better speech clarity

### Multi-Dialect and Language Support
- **100+ Languages and Dialects**: Support for major world languages and regional variants
- **Automatic Language Detection**: Detects spoken language from audio samples
- **Dialect-Specific Optimization**: Tailored Deepgram parameters for each language family
- **Code-Switching Support**: Handles mixed-language speech (e.g., Hindi-English)

### Advanced Transcription Features
- **Speaker Diarization**: Identifies and tracks different speakers
- **Confidence Scoring**: Shows transcription confidence levels
- **Topic Detection**: Automatically identifies conversation topics
- **Smart Formatting**: Automatic punctuation, paragraph detection, and number formatting
- **Utterance Detection**: Natural speech segmentation

## 🛠 Installation

### Prerequisites
- Python 3.8+
- Deepgram API key (Nova-2 model access recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Audio Processing Libraries
The enhanced version requires additional audio processing libraries:
```bash
pip install numpy scipy librosa webrtcvad soundfile
```

## ⚙️ Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
# Deepgram API Configuration
DEEPGRAM_API_KEY=your_deepgram_api_key_here
DEEPGRAM_MODEL=nova-2                    # Best accuracy model
DEEPGRAM_TIER=enhanced                   # Enhanced tier for better quality
DEEPGRAM_LANGUAGE=en-US                  # Default language

# Audio Processing Settings
SAMPLE_RATE=16000                        # Audio sample rate
ENABLE_VAD=true                          # Voice Activity Detection
VAD_AGGRESSIVENESS=3                     # VAD sensitivity (0-3)
ENABLE_NOISE_REDUCTION=true              # Spectral noise reduction
ENABLE_AUDIO_ENHANCEMENT=true            # Audio filtering and compression
```

## 🌍 Supported Languages and Dialects

### English Variants
- `en-US` (American English)
- `en-GB` (British English)
- `en-AU` (Australian English)
- `en-IN` (Indian English)
- `en-NZ` (New Zealand English)

### Spanish Variants
- `es-ES` (European Spanish)
- `es-MX` (Mexican Spanish)
- `es-AR` (Argentine Spanish)
- `es-CO` (Colombian Spanish)
- `es-PE` (Peruvian Spanish)

### French Variants
- `fr-FR` (French)
- `fr-CA` (Canadian French)
- `fr-BE` (Belgian French)
- `fr-CH` (Swiss French)

### German Variants
- `de-DE` (German)
- `de-AT` (Austrian German)
- `de-CH` (Swiss German)

### Indian Languages
- `hi-IN` (Hindi)
- `bn-IN` (Bengali)
- `ta-IN` (Tamil)
- `te-IN` (Telugu)
- `kn-IN` (Kannada)
- `ml-IN` (Malayalam)
- `gu-IN` (Gujarati)
- `pa-IN` (Punjabi)
- `or-IN` (Odia)
- `as-IN` (Assamese)

### Arabic Variants
- `ar-SA` (Saudi Arabic)
- `ar-EG` (Egyptian Arabic)
- `ar-MA` (Moroccan Arabic)
- `ar-DZ` (Algerian Arabic)

### Asian Languages
- `zh-CN` (Simplified Chinese)
- `zh-TW` (Traditional Chinese)
- `zh-HK` (Hong Kong Chinese)
- `ja-JP` (Japanese)
- `ko-KR` (Korean)
- `th-TH` (Thai)
- `vi-VN` (Vietnamese)
- `id-ID` (Indonesian)
- `ms-MY` (Malay)
- `tl-PH` (Filipino)

### European Languages
- `pt-BR` (Brazilian Portuguese)
- `pt-PT` (European Portuguese)
- `ru-RU` (Russian)
- `it-IT` (Italian)
- `it-CH` (Swiss Italian)
- `nl-NL` (Dutch)
- `nl-BE` (Belgian Dutch)
- `pl-PL` (Polish)
- `tr-TR` (Turkish)
- `sv-SE` (Swedish)
- `da-DK` (Danish)
- `no-NO` (Norwegian)
- `fi-FI` (Finnish)
- `he-IL` (Hebrew)

### And many more...

## 🎯 Usage

### Start the Server
```bash
python main.py
```

### API Endpoints

#### WebSocket Connection
```
ws://localhost:8000/ws/{session_id}?role={recorder|viewer}
```

#### Get Supported Languages
```bash
GET /languages
```

#### Get Audio Settings
```bash
GET /audio-settings
```

#### Get Session Transcripts
```bash
GET /transcripts/{session_id}
```

## 📊 Enhanced Transcript Format

The enhanced backend returns rich transcript data:

```json
{
  "type": "transcript",
  "text": "Hello, how are you today?",
  "speaker": 0,
  "is_final": true,
  "confidence": 0.95,
  "detected_language": "en-US",
  "topics": ["greeting", "wellbeing"],
  "utterances": ["Hello, how are you today?"],
  "timestamp": 1640995200.123
}
```

### Frontend Display Features
- **Speaker Identification**: Color-coded speaker dots
- **Confidence Indicators**: Visual confidence scores
- **Language Detection**: Shows detected language
- **Topic Tags**: Displays conversation topics
- **Real-time Updates**: Live transcription with interim results

## 🔧 Audio Processing Pipeline

1. **Audio Input**: Raw audio from microphone
2. **Noise Reduction**: Spectral subtraction removes background noise
3. **Audio Enhancement**: High-pass/low-pass filtering
4. **VAD Processing**: Removes silence segments
5. **Language Detection**: Analyzes initial audio samples
6. **Deepgram Processing**: Enhanced parameters for detected language
7. **Post-processing**: Speaker diarization, topic detection, formatting

## 🎛️ Customization

### VAD Aggressiveness Levels
- `0`: Least aggressive (keeps more audio)
- `1`: Low aggressiveness
- `2`: Medium aggressiveness
- `3`: Most aggressive (removes more silence)

### Noise Reduction Strength
Adjust `noise_reduction_strength` in AudioProcessor class:
- `0.05`: Light noise reduction
- `0.1`: Standard noise reduction
- `0.2`: Heavy noise reduction

### Language-Specific Optimizations
Add custom parameters in `_get_deepgram_params()`:

```python
elif detected_language.startswith("your-language"):
    base_params.update({
        "model": "nova-2",
        "language": "your-language-code",
        "smart_format": "true",
        "diarize": "true"
    })
```

## 🚨 Troubleshooting

### Audio Processing Not Available
If you see "Audio processing libraries not available" warning:
```bash
pip install numpy scipy librosa webrtcvad soundfile
```

### Poor Transcription Quality
1. Check microphone quality and positioning
2. Reduce background noise
3. Adjust VAD aggressiveness
4. Enable noise reduction and audio enhancement
5. Use Nova-2 model with enhanced tier

### Language Detection Issues
1. Ensure sufficient audio samples for detection
2. Check if language is in supported list
3. Manually set language in environment variables

## 📈 Performance Tips

1. **Use Nova-2 Model**: Best accuracy for all languages
2. **Enable Enhanced Tier**: Better quality for complex audio
3. **Optimize VAD Settings**: Balance between accuracy and processing speed
4. **Monitor Confidence Scores**: Low confidence indicates potential issues
5. **Use Speaker Diarization**: Helps with multi-speaker scenarios

## 🔮 Future Enhancements

- **Custom Language Models**: Train models for specific domains
- **Real-time Translation**: Multi-language translation
- **Emotion Detection**: Sentiment analysis from speech
- **Accent Recognition**: Fine-grained accent detection
- **Domain-Specific Optimization**: Medical, legal, technical vocabularies 