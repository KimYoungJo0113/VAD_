# model_training.py

import torch
import torch.nn as nn
import torch.nn.functional as F  # torch.nn.functional은 nn.Module의 함수 버전입니다.
import torch.optim as optim
import librosa
import numpy as np
from torch.utils.data import Dataset

# VADCNN 모델 클래스 정의
class VADCNN(nn.Module):
    def __init__(self, max_len=100):
        super(VADCNN, self).__init__()
        self.max_len = max_len  # max_len 속성을 추가
        self.conv1 = nn.Conv1d(13, 32, kernel_size=5, stride=1, padding=2) # MFCC의 13개 특징을 입력으로 받아 32개의 특징을 출력
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2) # 32개의 특징을 입력으로 받아 64개의 특징을 출력
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2) # 64개의 특징을 입력으로 받아 128개의 특징을 출력
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5) # 드롭아웃 추가
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    # forward 메서드에 max_len을 사용하여 MFCC의 길이를 패딩
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

'''
# AudioDataset 클래스 정의
class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, sr=16000, n_mfcc=13, max_len=None):
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.n_mfcc = n_mfcc

        # 모든 오디오 파일의 MFCC 길이를 계산하여 패딩을 위한 최대 길이 설정
        mfcc_lengths = [librosa.feature.mfcc(y=librosa.load(f, sr=sr)[0], sr=sr, n_mfcc=n_mfcc).shape[1] for f in audio_files]
        self.max_len = max(mfcc_lengths) if max_len is None else max_len # max_len을 사용하여 최대 길이 설정

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        file_path = self.audio_files[index]
        label = self.labels[index]

        y, sr = librosa.load(file_path, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)

        if mfcc.shape[1] < self.max_len:
            pad_width = ((0, 0), (0, self.max_len - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)
        else:
            mfcc = mfcc[:, :self.max_len] # 길면 자르기

        mfcc = torch.FloatTensor(mfcc)
        label = torch.LongTensor([label]).squeeze()
        return mfcc, label
'''

# 모델 학습 함수 정의
def train_model(model, train_loader, epochs=50):
    print("Starting model training...")
    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) # 스케줄러 추가

    for epoch in range(epochs):
        model.train() # 모델을 학습 모드로 설정
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (mfcc, label) in enumerate(train_loader): # 미니배치 단위로 학습
            optimizer.zero_grad() # 기울기 초기화
            outputs = model(mfcc) # 모델 예측
            loss = criterion(outputs, label) # 손실 계산
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트

            running_loss += loss.item() # 손실 누적
            _, predicted = outputs.max(1) # 가장 높은 확률을 가진 클래스 선택
            total += label.size(0) # 라벨 개수 누적
            correct += predicted.eq(label).sum().item() # 정답 개수 누적

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(
                    f'    [{epoch + 1}, {i + 1:5d}/{len(train_loader)}] loss: {running_loss / (i + 1):.3f} | Acc: {100. * correct / total:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss:.3f} | Acc: {epoch_acc:.2f}%")
        scheduler.step(epoch_loss) # 스케줄러에 현재 손실 전달

        current_lr = optimizer.param_groups[0]['lr'] # 현재 학습률 출력
        print(f"Current learning rate: {current_lr}")

        print("Finished Training")
    return model