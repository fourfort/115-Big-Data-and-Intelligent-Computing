# extract_shapelets.py
import numpy as np
from pyts.classification import LearningShapelets
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import time

def extract_shapelet_features(audio_segments, labels, target_length=5000):
    print(f"處理 {len(audio_segments)} 個片段（降採樣到長度 {target_length}）...")
    start_time = time.time()
    X_processed = []
    for seg in audio_segments:
        seg = np.array(seg, dtype=np.float32)
        if len(seg) != target_length:
            indices = np.linspace(0, len(seg)-1, target_length)
            seg = np.interp(indices, np.arange(len(seg)), seg)
        X_processed.append(seg)
    X = np.array(X_processed)
    print(f"處理完成！Shape: {X.shape}")
    le = LabelEncoder()
    y = le.fit_transform(labels) # 轉換標籤為數值
    print(f"標籤類別: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    # 關鍵修正：使用正確參數，避免記憶體爆炸
    shapelet_model = LearningShapelets(
        n_shapelets_per_size=1,min_shapelet_length=0.05,shapelet_scale=2,max_iter=5,tol=1e-2,random_state=42,verbose=1,n_jobs=1)
    print("開始 Shapelet 學習...")
    fit_start = time.time()
    shapelet_model.fit(X, y)
    fit_time = time.time() - fit_start
    print(f"Shapelet 學習完成，用時: {fit_time:.1f} 秒")
    # 用 decision_function 作為特徵（(n_samples, n_classes)）
    features = shapelet_model.decision_function(X)
    # 學到的 shapelets
    shapelets = shapelet_model.shapelets_
    # 分群註記
    print("對 Shapelet 特徵分群並註記描述...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    shapelet_clusters = kmeans.fit_predict(features)
    cluster_descriptions = ['piercing_sound', 'soft_meow', 'aggressive_call']
    annotated_shapelets = [cluster_descriptions[cluster] for cluster in shapelet_clusters]
    total_time = time.time() - start_time
    print(f"提取 {features.shape[1]} 個類別判別特徵，總用時: {total_time:.1f} 秒")
    return features, shapelets, le, annotated_shapelets



