# data_processing.py

import librosa
import numpy as np


def extract_features(voice, sr):
    # MFCC 추출
    mfcc = librosa.feature.mfcc(y=voice, sr=sr, n_mfcc=13)

    # Mel-spectrogram 추출
    mel_spec = librosa.feature.melspectrogram(y=voice, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Spectral Contrast 추출
    spectral_contrast = librosa.feature.spectral_contrast(y=voice, sr=sr)

    return mfcc, mel_spec_db, spectral_contrast

'''
def combine_and_label(voice_file, noise_file, voice_start, voice_end, sr=16000, num_augments=5):

    # 음성 파일 로드
    voice, _ = librosa.load(voice_file, sr=sr, offset=voice_start, duration=voice_end - voice_start)
    # 잡음 파일 로드
    noise, _ = librosa.load(noise_file, sr=sr)

    # 음성과 잡음의 길이를 맞춤
    if len(noise) < len(voice):
        noise = np.tile(noise, int(np.ceil(len(voice) / len(noise))))[:len(voice)]
    else:
        noise = noise[:len(voice)]

    # 여러 증강 데이터를 저장할 리스트
    augmented_data = []
    for _ in range(num_augments):
        noise_ratio = np.random.uniform(0.1, 0.9)
        noisy_voice = voice + noise_ratio * noise

        # 특성 추출 등 추가 작업...
        mfcc = librosa.feature.mfcc(y=noisy_voice, sr=sr, n_mfcc=40)
        mel_spec = librosa.feature.melspectrogram(y=noisy_voice, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spectral_contrast = librosa.feature.spectral_contrast(y=noisy_voice, sr=sr)

        # 라벨링 (음성 구간을 1로 설정)
        labels = np.ones(len(noisy_voice))  # 1: 음성 구간

        # 증강된 데이터를 리스트에 저장
        augmented_data.append((noisy_voice, labels, mfcc, mel_spec_db, spectral_contrast, sr))

    return augmented_data
'''