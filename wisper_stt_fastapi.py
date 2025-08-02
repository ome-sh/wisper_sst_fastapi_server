#License
#
# This software is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
#
# Copyright (C) 2025 Roland Kohlhuber
#
# Note: The AI model used by this software (openai/whisper-small) retains its original Apache 2.0 license and is not subject to the AGPL license terms.
#
# For the complete license text, see: https://www.gnu.org/licenses/agpl-3.0.html

from fastapi import FastAPI, UploadFile, File, Form
from starlette.responses import JSONResponse
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import tempfile
import os
import numpy as np
from pydub import AudioSegment
import io
import datetime
import uuid

app = FastAPI()

# Load model and processor - doing this at startup to avoid reloading for each request
print("Loading model and processor...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
print("Model and processor loaded successfully!")

# Directory to save WAV files
SAVE_DIR = "audio_files"
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_filename():
    """Generate a unique filename with timestamp and UUID"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{unique_id}.wav"

def convert_to_wav(audio_data, file_type, user_id=None):
    """Convert audio data to WAV format with 16kHz sample rate and save to disk"""
    with tempfile.NamedTemporaryFile(suffix=f'.{file_type}', delete=False) as temp_in:
        temp_in.write(audio_data)
        temp_in_path = temp_in.name
    
    # Output path for the converted WAV file
    temp_out_path = temp_in_path + '.wav'
    
    # Handle different input formats
    if file_type == 'ogg':
        audio = AudioSegment.from_ogg(temp_in_path)
    elif file_type == 'mp4' or file_type == 'm4a':  # WhatsApp audio format
        audio = AudioSegment.from_file(temp_in_path, format=file_type)
    else:
        # If we receive an unknown format, attempt to open it with pydub
        audio = AudioSegment.from_file(temp_in_path)
    
    # Convert to WAV with 16kHz sample rate, mono
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(temp_out_path, format='wav')
    
    # Read the audio data as numpy array
    speech_array, sample_rate = sf.read(temp_out_path)
    
    # Generate a filename and save to disk
    user_prefix = f"{user_id}_" if user_id else ""
    filename = user_prefix + generate_filename()
    save_path = os.path.join(SAVE_DIR, filename)
    
    # Save the WAV file to disk
    # sf.write(save_path, speech_array, sample_rate)
    # print(f"Saved WAV file to: {save_path}")
    
    # Clean up temp files
    os.unlink(temp_in_path)
    os.unlink(temp_out_path)
    
    return speech_array, save_path

@app.post('/transcribe')
async def transcribe_audio(
    audio: UploadFile = File(...),
    fileType: str = Form('wav'),
    userId: str = Form(None),
    language: str = Form(None)
):
    try:
        # Get the audio data from the request
        audio_data = await audio.read()
        
        file_type = fileType.lower()  # Default to wav if not specified
        
        # Convert to WAV format if needed
        if file_type == 'wav':
            # If already WAV, just read it
            with io.BytesIO(audio_data) as temp_wav:
                speech_array, sample_rate = sf.read(temp_wav)
                
            # Save the WAV file
            user_prefix = f"{userId}_" if userId else ""
            filename = user_prefix + generate_filename()
            save_path = os.path.join(SAVE_DIR, filename)
            sf.write(save_path, speech_array, sample_rate)
            print(f"Saved WAV file to: {save_path}")
        else:
            # Convert to WAV and save
            speech_array, save_path = convert_to_wav(audio_data, file_type, userId)
        
        # Process audio for model input
        input_features = processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features
        
        # Set forced decoder IDs if language is specified
        forced_decoder_ids = None
        if language:
            try:
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
                print(f"Using forced language: {language}")
            except:
                print(f"Language {language} not supported, using auto-detection")
                forced_decoder_ids = None
        
        # Generate transcription
        if forced_decoder_ids:
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        else:
            # Let the model auto-detect the language
            predicted_ids = model.generate(input_features)
        
        # Decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Get full output with special tokens to identify the detected language
        full_output = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
        detected_lang = "auto-detected"
        
        # Try to extract the language token from the output
        if "<|" in full_output and "|>" in full_output:
            lang_token_parts = full_output.split("<|")
            if len(lang_token_parts) > 1:
                lang_part = lang_token_parts[1].split("|>")[0]
                if lang_part not in ["startoftranscript", "transcribe", "notimestamps", "endoftext"]:
                    detected_lang = lang_part
        
        print(f"Transcription complete. Detected language: {detected_lang}")
        
        return {
            'text': transcription,
            'audio_file': save_path,
            'detected_language': detected_lang
        }
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(content={'error': str(e)}, status_code=500)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5006)