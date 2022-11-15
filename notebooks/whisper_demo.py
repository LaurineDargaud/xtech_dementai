import torch
import whisper
import numpy as np
import pickle

#file_name = "../data/processed/dementia/pitt/cookie/001-0.npy"
file_name = "../data/raw/Pitt/Dementia/Cookie/001-0.mp3"
model_to_load = "medium.en"

device = torch.device(f'cuda:0')

print('Model loaded')
model = whisper.load_model(model_to_load).to(device)
"""
options = whisper.DecodingOptions(language="en", without_timestamps=True)

mel = torch.from_numpy(np.load(file_name)).to(model.device)
results = model.decode(mel, options)

hypotheses = []

hypotheses.extend([result.text for result in results])
print(hypotheses)
"""

result = model.transcribe(file_name, verbose=True)
#import pdb;pdb.set_trace()

with open('filename.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


# load audio and pad/trim it to fit 30 seconds
#audio = whisper.load_audio(file_name)
#audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
#mel = whisper.log_mel_spectrogram(audio).to(model.device)
#print('mel:', mel.shape)
#mel = mel.view(1, mel.shape[0], mel.shape[1])

# get embed audio
#output = model.embed_audio(mel)

#output = output.detach().numpy()[0]