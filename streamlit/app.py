import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from predict import predict_emotion  
from streamlit_lottie import st_lottie 
import requests
import time

st.set_page_config(page_title="🎧 Audio Emotion Classifier", layout="wide")

from streamlit_lottie import st_lottie
import requests
import streamlit as st

# Load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_x62chJ.json")

col1, col2 = st.columns([1, 12])  

with col1:
    st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", 
        height=100, 
        width=150,
        key="hello_lottie",
    )

with col2:
    st.markdown("<h1 style='font-size: 50px; color: white; padding-top: 10px;'>🎧 Audio Emotion Classifier</h1>", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
st.markdown("Upload an audio file and let the model predict the emotion!")

uploaded_file = st.file_uploader("Upload your audio file (wav)", type="wav")

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    y, sr = librosa.load(uploaded_file, sr=22050)
    st.success(f"Audio loaded successfully at {sr}Hz")  

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    st.subheader("🎯 Predicted Emotion")
    with st.spinner("Predicting emotion..."):
        time.sleep(1.1)
        predicted_emotion = predict_emotion(y)
        st.markdown(
            f"<h1 style='font-size: 48px; color: #1DB954; padding-top: 10px;'>🧠 Emotion: {predicted_emotion}</h1>",
            unsafe_allow_html=True
        )


    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(12, 1))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)


    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mel Spectrogram")
        fig_mel, ax_mel = plt.subplots(figsize=(6, 3))  # Reduced height
        img_mel = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax_mel)
        fig_mel.colorbar(img_mel, ax=ax_mel, format='%+2.0f dB')
        ax_mel.set_title("Mel Spectrogram")
        st.pyplot(fig_mel)

    with col2:
        st.subheader("MFCC")
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(6, 2.45))  # Reduced height
        img_mfcc = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax_mfcc)
        fig_mfcc.colorbar(img_mfcc, ax=ax_mfcc)
        ax_mfcc.set_title("MFCC")
        st.pyplot(fig_mfcc)

    


