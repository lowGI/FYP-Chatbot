import torch
import torchaudio
import sounddevice as sd
import soundfile as sf
from speechbrain.pretrained import HIFIGAN
from speechbrain.pretrained import Tacotron2

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir")
mel_specs = torch.rand(2, 80,298)
waveforms = hifi_gan.decode_batch(mel_specs)
# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

def text_to_speech(text):
    # encode and decode text
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save('Recording/response.wav', waveforms.squeeze(1), 22050)

    # Play speech
    array, smp_rt = sf.read('Recording/response.wav', dtype = 'float32')
    waveforms_1_dim = waveforms[0][0]
    freq = 22050
    # play audio
    sd.play(waveforms_1_dim, freq)
    # Wait until file is done playing
    sd.wait()
    # stop the audio  
    sd.stop()