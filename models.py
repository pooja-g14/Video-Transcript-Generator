import whisper
import torch

def load_models():
    # Whisper model
    # whisper_model = whisper.load_model("large-v3")
    whisper_model = whisper.load_model("small")
    print("Whisper model loaded.")

    # Silero VAD model
    print("Loading Silero VAD model...")
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    vad_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vad_model.to(vad_device)
    (get_speech_timestamps, _, _, _, _) = vad_utils
    print("Silero VAD model loaded.")

    return whisper_model, vad_model, get_speech_timestamps, vad_device