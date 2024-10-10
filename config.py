# config.py

CONFIG = {
    'audio_folder': 'E:/테스트 데이터/음성',  # 테스트 음성 데이터 폴더로 변경
    'noise_dir': 'E:/테스트 데이터/잡음',     # 테스트 잡음 데이터 폴더로 변경
    'label_folder': 'E:/테스트 데이터/음성',  # 레이블 파일이 음성 파일과 같은 폴더에 있으므로
    'model_path': 'vad_cnn_model.pth',
    'chunk_duration': 0.3,
    'overlap': 0.1,
    'threshold': 0.6,
    'min_duration': 0.1,
    'merge_threshold': 0.2,
    'max_duration': 5.0,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'sr': 16000,
    'n_mfcc': 13,

}