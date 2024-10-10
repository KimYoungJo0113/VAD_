# data_generation.py
import os
import random
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset
from data_processing import extract_features


# 음성 파일이 들어있는 폴더 경로
voice_folder = r'D:/한국어 음성 데이터'
# 잡음 파일이 들어있는 폴더 경로
noise_folder = r'D:/잡음'

# 데이터셋 클래스 정의
class AudioDataset(Dataset):
    def __init__(self, voice_folder, noise_folder, sr=16000, n_mfcc=13):
        self.voice_files = [os.path.join(voice_folder, f) for f in os.listdir(voice_folder) if f.endswith('.mp3')]
        self.noise_files = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith('.mp3')]
        self.sr = sr
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.voice_files)

    def __getitem__(self, idx):
        # 음성 파일과 잡음 파일 경로 선택
        voice_path = self.voice_files[idx]
        noise_path = random.choice(self.noise_files)  # 무작위로 잡음 파일 선택

        # 음성 및 잡음 파일 로드
        voice, _ = librosa.load(voice_path, sr=16000)
        noise, _ = librosa.load(noise_path, sr=16000)

        # 음성 및 잡음의 길이를 맞춤 (짧은 쪽에 맞추기)
        if len(noise) < len(voice):
            voice = voice[:len(noise)]
        else:
            noise = noise[:len(voice)]

        # 음성에 잡음을 추가 (0.1 ~ 0.9 사이의 무작위 잡음 비율 설정)
        noise_ratio = random.uniform(0.1, 0.9)
        noisy_voice = voice + noise_ratio * noise

        # 특성 추출
        mfcc, mel_spec_db, spectral_contrast = extract_features(noisy_voice, sr=16000)

        # 음성 구간은 1, 잡음 구간은 0으로 라벨링
        labels = np.zeros(len(noisy_voice))
        labels[:len(voice)] = 1  # 음성 구간만 1로 라벨링

        # 텐서로 변환하여 반환
        mfcc_tensor = torch.FloatTensor(mfcc)
        mel_spec_db_tensor = torch.FloatTensor(mel_spec_db)
        spectral_contrast_tensor = torch.FloatTensor(spectral_contrast)
        labels_tensor = torch.FloatTensor(labels)

        return mfcc_tensor, mel_spec_db_tensor, spectral_contrast_tensor, labels_tensor


# 음성 및 잡음 결합과 라벨링 함수 정의
'''
def combine_and_label(voice_file, noise_file, voice_start, voice_end, sr=16000, noise_ratio=0.5):
    # 음성 파일 로드
    voice, _ = librosa.load(voice_file, sr=sr, offset=voice_start, duration=voice_end - voice_start)

    # 잡음 파일 로드
    noise, _ = librosa.load(noise_file, sr=sr)

    # 음성과 잡음의 길이를 맞춤
    if len(noise) < len(voice):
        noise = np.tile(noise, int(np.ceil(len(voice) / len(noise))))[:len(voice)]
    else:
        noise = noise[:len(voice)]

    # 음성과 잡음을 결합
    noisy_voice = voice + noise_ratio * noise

    # 특성 추출
    mfcc, mel_spec_db, spectral_contrast = extract_features(noisy_voice, sr)

    # 라벨링 (음성 구간을 1로 설정)
    labels = np.zeros(len(noisy_voice))
    labels[:] = 1  # voice_start부터 voice_end까지는 음성 구간으로 라벨링

    return noisy_voice,
'''