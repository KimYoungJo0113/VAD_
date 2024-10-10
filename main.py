import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import config
from model_training import VADCNN, train_model
from vad_inference import vad_cnn, combine_audio_and_noise
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import librosa
from torch.utils.data import DataLoader
from data_generation import AudioDataset


def process_test_set(model, voice_folder, noise_folder, label_folder, use_random_noise):
    results = {}
    all_true_labels = []
    all_pred_labels = []

    for filename in os.listdir(voice_folder):
        if filename.endswith('.mp3'):
            file_path = os.path.join(voice_folder, filename)
            print(f"{filename} 처리 중...")

            if use_random_noise:
                noise_file = None
            else:
                noise_filename = filename.replace("음성", "잡음")
                noise_file = os.path.join(noise_folder, noise_filename) if os.path.exists(os.path.join(noise_folder, noise_filename)) else None

            vad_segments = vad_cnn(file_path, model, noise_file=noise_file)

            # 레이블 파일 찾기
            file_number = filename.split('_')[-1].split('.')[0]
            label_file = os.path.join(label_folder, f"레이블 {file_number}.txt")
            print(f"찾고 있는 레이블 파일: {label_file}")  # 디버깅용 출력
            if os.path.exists(label_file):
                with open(label_file, 'r', encoding='utf-8') as f:
                    true_segments = []
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            start, end, label = map(float, parts[:3])
                            if label == 1:  # 음성 구간만 포함
                                true_segments.append((start, end))
            else:
                print(f"경고: {filename}에 대한 레이블 파일을 찾을 수 없습니다.")
                print(f"현재 디렉토리 내용: {os.listdir(label_folder)}")  # 디버깅용 출력
                continue


            audio_length = librosa.get_duration(filename=file_path)
            frame_rate = 100
            num_frames = int(audio_length * frame_rate)

            true_labels = np.zeros(num_frames)
            pred_labels = np.zeros(num_frames)

            for start, end in true_segments:
                true_labels[int(start * frame_rate):int(end * frame_rate)] = 1

            for start, end in vad_segments:
                pred_labels[int(start * frame_rate):int(end * frame_rate)] = 1

            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

            results[filename] = vad_segments

    if len(all_true_labels) > 0 and len(all_pred_labels) > 0:
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        precision = precision_score(all_true_labels, all_pred_labels, zero_division=1)
        recall = recall_score(all_true_labels, all_pred_labels, zero_division=1)
        f1 = f1_score(all_true_labels, all_pred_labels, zero_division=1)
    else:
        accuracy = precision = recall = f1 = 0.0

    print(f"총 프레임 수: {len(all_true_labels)}")
    print(f"예측된 음성 프레임 수: {sum(all_pred_labels)}")
    print(f"실제 음성 프레임 수: {sum(all_true_labels)}")

    return results, accuracy, precision, recall, f1


def main():
    task = input("작업을 선택하세요: 'vad'는 VAD 추론, 'train'은 모델 학습, 'test'는 테스트 세트 평가: ").strip().lower()

    if task == 'train':
        # 오디오 파일과 라벨 파일들이 모두 들어있는 폴더 경로 설정
        voice_folder = r'E:/한국어 음성 데이터'
        noise_folder = r'E:/잡음'  # 잡음 데이터 폴더 경로 추가

        print("데이터 폴더에서 AudioDataset을 생성 중...")
        try:
            train_dataset = AudioDataset(voice_folder=voice_folder, noise_folder=noise_folder)
            print("AudioDataset 생성 성공.")
            print(f"훈련 데이터셋 길이: {len(train_dataset)}")

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            print("DataLoader 생성 성공.")

            model = VADCNN()
            trained_model = train_model(model, train_loader)

            model_save_path = 'vad_cnn_model.pth'
            torch.save({
                'model_state_dict': trained_model.state_dict(),
            }, model_save_path)

            if os.path.exists(model_save_path):
                print(f"모델 학습 완료 및 '{model_save_path}'로 저장됨")
            else:
                print("오류: 모델 파일이 제대로 저장되지 않았습니다.")
        except Exception as e:
            print(f"훈련 중 오류 발생: {str(e)}")

    elif task in ['vad', 'test']:
        model_save_path = 'vad_cnn_model.pth'
        if not os.path.exists(model_save_path):
            print(f"오류: {model_save_path}가 존재하지 않습니다. 먼저 모델을 학습하세요.")
            return

        checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
        model = VADCNN()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if task == 'vad':
            audio_file = input("VAD를 수행할 오디오 파일 경로를 입력하세요: ").strip()
            use_random_noise = input("랜덤 노이즈를 사용할까요? (예/아니오): ").strip().lower() == 'yes'

            if use_random_noise:
                noise_file = None
            else:
                noise_file = input("노이즈 파일 경로를 입력하세요 (없으면 Enter를 누르세요): ").strip() or None

            vad_segments = vad_cnn(audio_file, model, noise_file=noise_file)
            print("VAD 결과 구간:", vad_segments)

            # Visualization
            y, sr = librosa.load(audio_file, sr=config.CONFIG['sr'])
            if noise_file:
                y_noise, _ = librosa.load(noise_file, sr=sr)
                y = combine_audio_and_noise(y, y_noise)

            plt.figure(figsize=(12, 6))
            librosa.display.waveshow(y, sr=sr, alpha=0.6)
            plt.title('Audio Waveform with VAD Segments')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')

            for idx, (start, end) in enumerate(vad_segments):
                label = 'Detected VAD' if idx == 0 else '_'
                plt.axvspan(start, end, color='red', alpha=0.3, label=label)

            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()

        else:  # task == 'test'
            voice_folder = config.CONFIG['audio_folder']
            noise_folder = config.CONFIG['noise_dir']
            label_folder = config.CONFIG['label_folder']
            use_random_noise = input("모든 테스트 파일에 랜덤 노이즈를 사용할까요? (예/아니오): ").strip().lower() == 'yes'

            if not all(os.path.isdir(folder) for folder in [voice_folder, noise_folder, label_folder]):
                print(f"오류: 지정된 폴더 중 하나 이상이 유효하지 않습니다.")
                return

            results, accuracy, precision, recall, f1 = process_test_set(model, voice_folder, noise_folder, label_folder, use_random_noise)

            print("\n테스트 세트 VAD 결과:")
            for filename, segments in results.items():
                print(f"{filename}: {segments}")

            print("\n테스트 세트 평가 메트릭:")
            print(f"정확도(Accuracy): {accuracy:.4f}")
            print(f"정밀도(Precision): {precision:.4f}")
            print(f"재현율(Recall): {recall:.4f}")
            print(f"F1 점수(F1 Score): {f1:.4f}")

    else:
        print("잘못된 작업입니다. 'train', 'vad', 또는 'test'를 선택하세요.")


if __name__ == "__main__":
    main()
