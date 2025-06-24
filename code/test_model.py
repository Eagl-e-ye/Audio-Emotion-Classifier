import torch.nn as nn
import torch
import librosa
import numpy as np
import os
import glob

class DeepAudioEmotionCNN(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.3):
        super(DeepAudioEmotionCNN, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            self._make_conv_block(2, 64, dropout_rate),
            self._make_conv_block(64, 128, dropout_rate),
            self._make_conv_block(128, 256, dropout_rate),
            self._make_conv_block(256, 512, dropout_rate),
            self._make_conv_block(512, 512, dropout_rate),
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def _make_conv_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate)
        )
    def forward(self, x):
        x = self.conv_blocks(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_audio(model, filepath, sr=22050, n_mels=128):
    try:
        y, _ = librosa.load(filepath, sr=sr)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        target_length = int(3.5 * sr)
        if len(y) < target_length:
            y = np.pad(y, ((target_length - len(y)) // 2, (target_length - len(y) + 1) // 2))
        else:
            start = (len(y) - target_length) // 2
            y = y[start:start + target_length]

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mels, n_fft=1024, hop_length=512)
        mfcc = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min())

        input_tensor = torch.tensor(np.stack([log_mel_spec, mfcc]), dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            return torch.max(outputs, 1)[1].item()
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def main():
    emotions = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}
    
    model = DeepAudioEmotionCNN(num_classes=8).to(device)
    model.load_state_dict(torch.load('model.pth', weights_only=True, map_location=device))
    model.eval()
    
    choice = input("Enter (1) for single file or (2) for directory: ").strip()
    path = input("Enter file/directory path: ").strip()
    
    if choice == '1':
        pred = predict_audio(model, path)
        print(f"{path}: {emotions[pred + 1] if pred is not None else 'Error'}")
    
    elif choice == '2':
        wav_files = glob.glob(os.path.join(path, "*.wav")) + glob.glob(os.path.join(path, "*.WAV"))
        if not wav_files:
            print("No WAV files found!")
            return
        
        for file in wav_files:
            pred = predict_audio(model, file)
            emotion = emotions[pred + 1] if pred is not None else 'Error'
            print(f"{os.path.basename(file)}: {emotion}")
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()