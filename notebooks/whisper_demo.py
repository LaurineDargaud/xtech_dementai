import torch
import whisper

model = whisper.load_model("medium.en")
#result = model.transcribe("audio.mp3")
#print(result["text"])

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio_control.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)
print('mel:', mel.shape)
mel = mel.view(1, mel.shape[0], mel.shape[1])

# get embed audio
output = model.embed_audio(mel)

output = output.detach().numpy()[0]

print(output.shape)

print(output)