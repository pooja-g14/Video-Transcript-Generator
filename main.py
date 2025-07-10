from moviepy import VideoFileClip
import numpy as np
import soundfile as sf
import librosa
import os
import torch
from models import load_models

def process_audio(input_audio_path, whisper_model, vad_model, get_speech_timestamps):
    try:
        print(f"Processing audio for silence removal: {input_audio_path}")
        audio, sr = librosa.load(input_audio_path, sr=16000)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for processing: {device}")
        audio_tensor = torch.tensor(audio).to(device)
        speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, threshold=0.3,
                                                min_speech_duration_ms=200, min_silence_duration_ms=800,
                                                window_size_samples=512, speech_pad_ms=700,
                                                return_seconds=False)
        buffer_samples = int(0.3 * sr)
        buffered_segments = [(max(0, seg['start'] - buffer_samples), min(len(audio), seg['end'] + buffer_samples))
                             for seg in speech_timestamps]
        merged_segments_buffered = []
        if buffered_segments:
            current_start, current_end = buffered_segments[0]
            for next_start, next_end in buffered_segments[1:]:
                if next_start - current_end < 0.8 * sr:
                    current_end = next_end
                else:
                    merged_segments_buffered.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            merged_segments_buffered.append((current_start, current_end))
        non_silent_audio = np.concatenate([audio[start:end] for start, end in merged_segments_buffered] +
                                          [np.zeros(int(0.15 * sr))] * len(merged_segments_buffered[:-1]))
        processed_audio_path = os.path.join('outputs', "temp_audio_processed.wav")
        sf.write(processed_audio_path, non_silent_audio, sr)
        print(f"Silence removed. Processed audio saved to: {processed_audio_path}")
        print(f"Reduced audio length: Original: {len(audio)/sr:.1f}s → New: {len(non_silent_audio)/sr:.1f}s")

        print("Transcribing processed audio...")
        transcription_result = whisper_model.transcribe(
            processed_audio_path,
            language="en",
            task="transcribe",
            fp16=torch.cuda.is_available(),
            verbose=False,
            beam_size=3,
            initial_prompt="The audio is a classroom lecture, transcribe the main speaker accurately."
        )
        return transcription_result["text"]
    except Exception as e:
        print(f"Error during audio processing and transcription: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_video_for_transcript(video_path):
    try:
        print(f"📹 Processing video: {video_path}")
        
        # Load models
        whisper_model, vad_model, get_speech_timestamps, _ = load_models()

        # Extract Audio
        print("🔊 Extracting audio...")
        clip = VideoFileClip(video_path)
        audio_path = os.path.join('outputs', "temp_audio.wav")
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()
        print(f"🎵 Audio saved to: {audio_path}")

        # Process and transcribe
        transcript_text = process_audio(audio_path, whisper_model, vad_model, get_speech_timestamps)

        # Cleanup
        for f in [audio_path, os.path.join('outputs', "temp_audio_processed.wav")]:
            if os.path.exists(f):
                os.remove(f)
                print(f"🗑️ Removed intermediate file: {f}")

        print("✅ Video processing complete.")
        return transcript_text

    except Exception as e:
        print(f"❌ An error occurred during the process: {e}")
        import traceback
        traceback.print_exc()
        return None