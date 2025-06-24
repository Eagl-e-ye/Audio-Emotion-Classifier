import torch.nn as nn
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
        #dense layers
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