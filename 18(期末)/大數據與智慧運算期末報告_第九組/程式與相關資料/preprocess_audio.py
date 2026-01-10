# preprocess_audio.py
import librosa
import numpy as np

def denoise_and_normalize(audio, sr):
    """簡單去噪與正規化"""
    audio = librosa.effects.preemphasis(audio)
    audio = audio / np.max(np.abs(audio) + 1e-8)  # 避免除0
    return audio

def segment_cat_calls(audio, sr, min_duration=0.2, top_db=20):
    """
    使用能量偵測分割單一叫聲
    min_duration: 最小長度（秒）
    """
    intervals = librosa.effects.split(audio, top_db=top_db, frame_length=2048, hop_length=512)
    segments = []
    min_samples = int(min_duration * sr)
    for start, end in intervals:
        if end - start >= min_samples:
            segments.append(audio[start:end])
    return segments

def process_dataset(raw_audio, raw_srs, raw_labels, use_preemphasis=True):
    segments = []
    segment_labels = []

    for audio, sr, label in zip(raw_audio, raw_srs, raw_labels):

        if use_preemphasis:
            audio = denoise_and_normalize(audio, sr) # Step 1：去噪與正規化

        segs = segment_cat_calls(audio, sr, min_duration=0.2, top_db=20) # Step 2：音訊分割

        for seg in segs:
            segments.append(seg)
            segment_labels.append(label)

    print(f"提取 {len(segments)} 個貓叫片段")
    return segments, segment_labels
