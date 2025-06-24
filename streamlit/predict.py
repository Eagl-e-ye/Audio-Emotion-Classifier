import torch
import librosa
import numpy as np
from model import DeepAudioEmotionCNN 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepAudioEmotionCNN(num_classes=8).to(device)

state_dict = torch.load('model.pth') 
model.load_state_dict(state_dict)
model.eval()

emotions = ["NEUTRAL 😐", "CALM 😌", "HAPPY 😺", "SAD 😥", "ANGRY 😠", "FEARFUL 😨", "DISGUST 🥴", "SURPRISED 😯"]  # change as per your model


def predict_emotion(extracted_data, sr=22050):
    
    target_length = int(3.6 * sr) 
    
  
    y, _ = librosa.effects.trim(extracted_data, top_db=20)

    if len(y) < target_length:
        pad_total = target_length - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right))
    else:
        start = (len(y) - target_length) // 2
        y = y[start:start + target_length]

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, n_fft=1024, hop_length=512)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    stacked_spec = np.stack([log_mel_spec, mfcc])
    tensor_spec = torch.tensor(stacked_spec, dtype=torch.float32)
    input_tensor = tensor_spec.clone().detach().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return emotions[predicted.item()]
