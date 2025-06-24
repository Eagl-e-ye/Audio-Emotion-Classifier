# Audio-Emotion-Classifier
## 📌 Project Description

This project aims to design and implement an end-to-end pipeline for emotion classification from speech/audio data using deep learning techniques. Recognizing emotions conveyed through voice is essential for enhancing human-computer interaction, with practical applications in areas such as mental health monitoring, customer service automation, virtual assistants, and multimedia content analysis.

To achieve accurate emotion recognition, the system leverages advanced audio processing methods by extracting two complementary feature representations: Mel Spectrogram and Mel Frequency Cepstral Coefficients (MFCC). These features are combined into a 2-channel input, capturing both the temporal-frequency dynamics and perceptual characteristics of speech signals.

The combined feature tensor is then fed into a Convolutional Neural Network (CNN) that learns to model subtle variations in tone, pitch, and intensity associated with different emotional states. This approach enables the model to effectively identify and categorize diverse emotions expressed in speech or song, delivering robust classification performance.

---

## 🔄 Pre-processing Methodology

### 🔉 Audio Loading and Trimming

- Loaded `.wav` files of variable lengths.
- Data augmentation: 
  - Pitch Shifting: Randomly shifted the pitch up or down by up to ±2 semitones
  - Time Stretching: Temporally stretched or compressed the audio by a random factor between 0.9 and 1.1, then resampled to maintain original length.
  - Additive Gaussian Noise: Added low-level Gaussian noise with an amplitude scaled to 0.5% of the signal max amplitude to simulate background noise.
- Duration normalization: After analyzing the dataset, the useful content was found to average around **3.5 seconds**, so each audio was trimmed or padded to this fixed duration.

### 🎧 Feature Extraction

- **MFCC** and **Mel Spectrogram** were generated separately using the same configuration:
  - `n_mels` = *128*
  - Time frames selected to match after trimming
- Converted **Mel Spectrogram** to **log scale** to simulate human hearing
- **Normalized** the features
- **Stacked** MFCC and Mel Spectrogram into a **2-channel tensor**, suitable as CNN input

---


## 🧠 Model Pipeline

The model used for this project is a deep Convolutional Neural Network (CNN) designed to learn hierarchical features from **2-channel audio representations**—stacked MFCC and Mel Spectrogram images.

### ⚙️ Architecture Overview

The architecture (`DeepAudioEmotionCNN`) consists of:

- **5 Convolutional Blocks**:
  - Each block contains:
    - Two convolutional layers with `3×3` kernels
    - Batch normalization and ReLU activation
    - Max pooling for downsampling
    - 2D dropout for regularization

- **Dense Classifier**:
  - A 3-layer fully connected head:
    - `512 → 256 → 128 → num_classes`
    - ReLU activation and dropout in between layers

> **Input:** `(batch_size, 2, 128, time_frames)`  
> **Output:** `(batch_size, num_classes=8)`

The 2-channel input is formed by stacking MFCC and Mel Spectrogram features for each audio clip.

---

### 🔧 Overfitting & Optimization Strategies

Early training runs showed clear signs of **overfitting**. To address this, the following strategies were implemented:

- 🔁 **Dropout Regularization**:
  - Used both `Dropout2d` in convolutional blocks and `Dropout` in dense layers
  - Tuned dropout rate between **0.3 and 0.5**

- ⚙️ **Learning Rate Tuning**:
  - Tried different learning rates (`1e-3`, `5e-4`, `1e-4`)

- 🔀 **Feature Fusion**:
  - Combined MFCC and Mel Spectrogram features into a single **2-channel input**
  - This fusion significantly improved generalization performance and validation accuracy

- 🧂 **Label Smoothing**:
  - Introduced label smoothing with a final smoothing factor of 0.05
  - Helped reduce overconfidence in predictions and improved model calibration on unseen data

- ⚖️ Class Imbalance Handling:
  - Observed imbalanced class distribution in the dataset
  - Applied class weighting in the loss function using weights:
    [1.3, 1.0, 2.0, 3.5, 1.0, 1.5, 2.0, 1.0]
  - This helped the model pay more attention to underrepresented and lower-performing classes

---

### 🕒 Training Considerations

- Training time increased due to deeper architecture and higher input dimensionality
- Despite the heavier compute requirement, the performance boost was substantial
- With proper tuning and feature combination, the model achieved **82-84%** accuracy

---

### 📘 Model Summary

| Component          | Details                          |
|--------------------|----------------------------------|
| Input              | 2-channel (MFCC + Mel), 128×T    |
| Conv Layers        | 5 blocks, 64→128→256→512→512     |
| Pooling            | MaxPool2d + GlobalAvgPool2d      |
| Fully Connected    | 512 → 256 → 128 → 8              |
| Activation         | ReLU                             |
| Regularization     | BatchNorm + Dropout (0.3–0.5)    |
| Optimizer          | Adam (custom LR experiments)     |
| Output             | 8 emotion classes                |


---

## 📈 Performance

The model demonstrated strong performance on the emotion classification task, particularly after combining **Mel Spectrogram** and **MFCC** features into a 2-channel input.

### 🔹 Individual Feature Evaluation:
- **Mel Spectrogram only**: Achieved accuracy ranging from **50% to 55%**
- **MFCC only**: Achieved accuracy in the same range (**50% to 60%**)
- These results confirmed both were effective, but had limitations when used in isolation

### 🔹 Combined Feature Input (MFCC + Mel Spectrogram):
- Final training accuracy: **~83%** after sufficient epochs
- The combined approach enabled the model to learn richer emotional cues from the audio

---

### 🧾 Final Evaluation Metrics:
- **Overall Accuracy**: `82.50%`
- **F1 Score (Macro-Averaged)**: `82.29%`

#### ✅ Per-Class Accuracy:
| Class | Accuracy     |
|-------|--------------|
| 0     | 84.21%      |
| 1     | 82.67%      |
| 2     | 82.67%      |
| 3     | 78.67%      |
| 4     | 88.00%      |
| 5     | 78.67%      |
| 6     | 84.62%      |
| 7     | 82.05%      |

---

### 📊 Confusion Matrix:

|       | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 | Pred 5 | Pred 6 | Pred 7 |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| **True 0** | 32     | 1      | 1      | 4      | 0      | 0      | 0      | 0      |
| **True 1** | 4      | 62     | 2      | 7      | 0      | 0      | 0      | 0      |
| **True 2** | 2      | 1      | 62     | 2      | 2      | 3      | 1      | 2      |
| **True 3** | 1      | 2      | 1      | 59     | 1      | 10     | 0      | 1      |
| **True 4** | 0      | 0      | 0      | 0      | 66     | 0      | 7      | 2      |
| **True 5** | 0      | 0      | 2      | 10     | 2      | 59     | 1      | 1      |
| **True 6** | 0      | 0      | 1      | 1      | 1      | 2      | 33     | 1      |
| **True 7** | 1      | 0      | 1      | 0      | 0      | 1      | 4      | 32     |

---

## ✅ Why MFCC + Mel Spectrogram?

- **Mel Spectrogram** captures the **temporal and frequency structure**, ideal for CNNs.
- **MFCC** mimics human auditory perception, emphasizing **emotion-relevant frequencies**.
- When used together:
  - Complement each other's strengths
  - Compact and noise-resistant
  - Proven effective in research for emotion detection tasks


## 📂 Dataset
- https://zenodo.org/records/1188976#.XCx-tc9KhQI
  - Audio_Speech_Actors_1-24
  - Audio_Song_Actors_1-24


