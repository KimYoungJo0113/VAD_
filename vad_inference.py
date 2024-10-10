# vad_inference.py

import os
import random
import librosa
import numpy as np
import torch

# 음성과 노이즈를 결합하는 함수 정의
def combine_audio_and_noise(voice, noise, noise_ratio=0.5):
    if len(noise) < len(voice):
        noise = np.tile(noise, int(np.ceil(len(voice) / len(noise))))[:len(voice)]
    else:
        noise = noise[:len(voice)]
    return voice + noise_ratio * noise

# VAD 추론 함수 정의
def vad_cnn(voice_file, model, noise_file=None, noise_dir=None, chunk_duration=0.3, overlap=0.1, threshold=0.6):
    # 음성 파일 로드 및 샘플링 레이트 설정
    voice, sr = librosa.load(voice_file, sr=16000)
    # 음성 파일을 chunk_duration과 overlap을 사용하여 분할
    chunk_size = int(chunk_duration * sr)
    overlap_size = int(overlap * sr)
    step = chunk_size - overlap_size

    # 예외 처리: chunk_duration과 overlap이 유효한 step 크기를 생성하지 않는 경우
    if step <= 0:
        raise ValueError("The combination of chunk_duration and overlap results in an invalid step size. Ensure chunk_duration > overlap.")

    if noise_file is None and noise_dir:
        noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav') or f.endswith('.mp3')]
        noise_file = os.path.join(noise_dir, random.choice(noise_files))

    if noise_file:
        noise, _ = librosa.load(noise_file, sr=sr)
        noisy_voice = combine_audio_and_noise(voice, noise)
    else:
        noisy_voice = voice

    vad_segments = []

    # 추론 단계: 모델을 사용하여 각 chunk에 대한 음성 확률 예측
    for start in range(0, len(noisy_voice) - chunk_size + 1, step):
        end = start + chunk_size
        chunk_voice = noisy_voice[start:end]

        # MFCC 특징 추출
        mfcc = librosa.feature.mfcc(y=chunk_voice, sr=sr, n_mfcc=13)
        # 모델 입력 크기에 맞게 패딩
        if mfcc.shape[1] < model.max_len:
            pad_width = ((0, 0), (0, model.max_len - mfcc.shape[1])) # 패딩
            mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)
        else:
            mfcc = mfcc[:, :model.max_len]
        # 텐서로 변환
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0)
        # 추론
        with torch.no_grad():
            output = model(mfcc_tensor)
            probabilities = torch.softmax(output, dim=1)
            voice_prob = probabilities[0][1].item()
        # 확률이 임계값보다 높으면 음성으로 판단
        if voice_prob > threshold:
            vad_segments.append((start / sr, end / sr))

    # 병합 단계: 연속된 구간 병합 개선
    merged_segments = []
    merge_tolerance = 0.2  # 병합할 수 있는 최대 간격 (0.2초)
    for start, end in vad_segments:
        if merged_segments and start - merged_segments[-1][1] < merge_tolerance:
            merged_segments[-1] = (merged_segments[-1][0], end)
        else:
            merged_segments.append((start, end))

    return merged_segments
