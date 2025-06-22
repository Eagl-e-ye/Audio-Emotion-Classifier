# Audio-Emotion-Classifier
## 📌 Project Description

This project aims to design and implement an end-to-end pipeline for emotion classification from speech/audio data using deep learning techniques. Recognizing emotions conveyed through voice is essential for enhancing human-computer interaction, with practical applications in areas such as mental health monitoring, customer service automation, virtual assistants, and multimedia content analysis.

To achieve accurate emotion recognition, the system leverages advanced audio processing methods by extracting two complementary feature representations: Mel Spectrogram and Mel Frequency Cepstral Coefficients (MFCC). These features are combined into a 2-channel input, capturing both the temporal-frequency dynamics and perceptual characteristics of speech signals.

The combined feature tensor is then fed into a Convolutional Neural Network (CNN) that learns to model subtle variations in tone, pitch, and intensity associated with different emotional states. This approach enables the model to effectively identify and categorize diverse emotions expressed in speech or song, delivering robust classification performance.

---

## 🔄 Pre-processing Methodology

### 🔉 Audio Loading and Trimming

- Loaded `.wav` files of variable lengths.
- Silence removal: Trimmed the start and end of each file using a decibel threshold of **< 20 dB** to eliminate low-energy silence.
- Duration normalization: After analyzing the dataset, the useful content was found to average around **3.5 seconds**, so each audio was trimmed or padded to this fixed duration.

### 🎧 Feature Extraction

- **MFCC** and **Mel Spectrogram** were generated separately using the same configuration:
  - `n_mels` = *128*
  - `n_fft` = *1024*
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
  - Settled on a balance that offered fast convergence without instability

- 🔀 **Feature Fusion**:
  - Combined MFCC and Mel Spectrogram features into a single **2-channel input**
  - This fusion significantly improved generalization performance and validation accuracy

---

### 🕒 Training Considerations

- Training time increased due to deeper architecture and higher input dimensionality
- Despite the heavier compute requirement, the performance boost was substantial
- With proper tuning and feature combination, the model achieved **over 95% accuracy** and eventually **100% training accuracy** with more epochs

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
- **Mel Spectrogram only**: Achieved accuracy ranging from **70% to 75%**
- **MFCC only**: Achieved accuracy in the same range (**70% to 75%**)
- These results confirmed both were effective, but had limitations when used in isolation

### 🔹 Combined Feature Input (MFCC + Mel Spectrogram):
- Initial training accuracy: **85%+**
- Final training accuracy: **~90%** after sufficient epochs
- The combined approach enabled the model to learn richer emotional cues from the audio

---

### 🧾 Final Evaluation Metrics:
- **Overall Accuracy**: `95.00%`
- **F1 Score (Macro-Averaged)**: `95.00%`

#### ✅ Per-Class Accuracy:
| Class | Accuracy     |
|-------|--------------|
| 0     | 100.00%      |
| 1     | 100.00%      |
| 2     | 100.00%      |
| 3     | 100.00%      |
| 4     | 100.00%      |
| 5     | 100.00%      |
| 6     | 100.00%      |
| 7     | 100.00%      |

---

### 📊 Confusion Matrix:

|       | Pred 0 | Pred 1 | Pred 2 | Pred 3 | Pred 4 | Pred 5 | Pred 6 | Pred 7 |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| **True 0** | 37     | 0      | 0      | 0      | 0      | 0      | 0      | 0      |
| **True 1** | 0      | 63     | 0      | 0      | 0      | 0      | 0      | 0      |
| **True 2** | 0      | 0      | 79     | 0      | 0      | 0      | 0      | 0      |
| **True 3** | 0      | 0      | 0      | 78     | 0      | 0      | 0      | 0      |
| **True 4** | 0      | 0      | 0      | 0      | 74     | 0      | 0      | 0      |
| **True 5** | 0      | 0      | 0      | 0      | 0      | 85     | 0      | 0      |
| **True 6** | 0      | 0      | 0      | 0      | 0      | 0      | 35     | 0      |
| **True 7** | 0      | 0      | 0      | 0      | 0      | 0      | 0      | 40     |

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


