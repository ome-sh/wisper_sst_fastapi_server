# Whisper Audio Transcription Service

A FastAPI-based audio transcription service using OpenAI's Whisper model with multi-format support and automatic language detection.

## Features

- **Multi-Format Support**: Handles WAV, OGG, MP4, M4A, and other common audio formats
- **Language Detection**: Automatic language detection or forced language specification
- **Audio Processing**: Automatic conversion to 16kHz mono WAV for optimal transcription
- **File Management**: Saves processed audio files with unique timestamps and UUIDs
- **User Tracking**: Optional user ID tagging for file organization
- **WhatsApp Compatible**: Supports WhatsApp audio message formats (OGG, M4A)

## Quick Start

### Installation

```bash
pip install fastapi uvicorn torch transformers soundfile pydub
```

### Run the Service

```bash
python whisper_server.py
```

The service will start on `http://0.0.0.0:5006`

## API Usage

### Transcribe Audio

**Endpoint:** `POST /transcribe`

**Parameters:**
- `audio` (file): Audio file to transcribe
- `fileType` (form): Audio format (wav, ogg, mp4, m4a, etc.) - defaults to 'wav'
- `userId` (form, optional): User identifier for file organization
- `language` (form, optional): Force specific language (e.g., 'en', 'de', 'fr')

**Example with curl:**

```bash
curl -X POST "http://localhost:5006/transcribe" \
  -F "audio=@recording.ogg" \
  -F "fileType=ogg" \
  -F "userId=user123" \
  -F "language=en"
```

**Response:**

```json
{
  "text": "Hello, this is a test transcription.",
  "audio_file": "audio_files/user123_20250802_143052_a1b2c3d4.wav",
  "detected_language": "en"
}
```

## Configuration

- **Model**: Uses `openai/whisper-small` (can be changed in code)
- **Audio Directory**: Files are saved to `audio_files/` directory
- **Sample Rate**: Audio is converted to 16kHz mono for processing

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- Uvicorn
- SoundFile
- Pydub
- FFmpeg (for audio format conversion)

## License

AGPL-3.0 (Whisper model retains its original Apache 2.0 license)
