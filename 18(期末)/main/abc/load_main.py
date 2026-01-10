# load_data.py 或直接放 main.py
import os
import librosa
import pandas as pd
import re

def load_cat_vocalizations(data_path='cat_vocalization_dataset'):
    """
    載入所有 .wav 檔，並根據命名規則提取標籤
    命名格式: C_NNNNN_BB_SS_OOOOO_RXX.wav
    C = B (brushing), F (waiting for food), I (isolation)
    """
    files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
    if not files:
        raise FileNotFoundError("資料夾中無 .wav 檔")

    data = []
    for filename in files:
        filepath = os.path.join(data_path, filename)

        # 使用正規表達式提取 C (第一個字母)
        match = re.match(r'([BFI])_', filename.upper())
        if not match:
            print(f"[Warning] 跳過無效檔名: {filename}")
            continue
        context = match.group(1)
        if context == 'B':
            context_label = 'brushing'
        elif context == 'F':
            context_label = 'waiting for food'
        elif context == 'I':
            context_label = 'isolation'
        else:
            context_label = 'unknown'

        # 載入音訊
        y, sr = librosa.load(filepath, sr=None)  # 保留原始取樣率

        data.append({
            'filename': filename,
            'audio': y,
            'sr': sr,
            'context': context,          # 原始 C (B/F/I)
            'context_label': context_label  # 完整描述
        })

    df = pd.DataFrame(data)
    print(f"成功載入 {len(df)} 筆音訊")
    print("情境分佈:")
    print(df['context_label'].value_counts())
    return df