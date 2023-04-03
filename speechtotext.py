from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# Audio record libraries
import torch
import sounddevice as sd
from scipy.io.wavfile import write
from datasets import load_dataset

access_token = "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"
# Speech-to-Text pretrained model and processor
stt_model_repo_name = "STT_Model_Finale"
stt_processor = Wav2Vec2Processor.from_pretrained(f"LowGI/{stt_model_repo_name}",use_auth_token=access_token)
stt_model = Wav2Vec2ForCTC.from_pretrained(f"LowGI/{stt_model_repo_name}",use_auth_token=access_token)
#huggingface_hub.logout()

def speech_to_text(duration):
    # Record speech
    # Sampling frequency
    freq = 16000
    # Start recorder with the given values
    # of duration and sample frequency
    waveforms = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    # Record audio for the given number of seconds
    sd.wait()
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write("Recording/request.wav", freq, waveforms)

    speech = load_dataset("Recording", split="train[:]")
    waveforms = speech['audio'][0]['array']
    
    with torch.no_grad():
        input_values = torch.tensor(waveforms).unsqueeze(0)
        logits = stt_model(input_values).logits
        
    ids = torch.argmax(logits, dim=-1)
    text = stt_processor.batch_decode(ids)[0]
    
    return text