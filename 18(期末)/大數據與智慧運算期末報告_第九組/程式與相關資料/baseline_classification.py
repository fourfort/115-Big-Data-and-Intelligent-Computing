# baseline_classification.py
import time
import numpy as np
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix

def extract_mfcc_features(segments, n_mfcc=20):
    """提取 MFCC 平均特徵"""
    mfcc_features = []
    for seg in segments:
        mfcc = librosa.feature.mfcc(y=np.array(seg), sr=44100, n_mfcc=n_mfcc)
        mfcc_features.append(mfcc.mean(axis=1))
    return np.array(mfcc_features)

def extract_mel_features(segments):
    """提取 Mel Spectrogram 平均特徵（與 main 中一致）"""
    mel_features = []
    for seg in segments:
        mel = librosa.feature.melspectrogram(y=np.array(seg), sr=44100)
        mel_features.append(mel.mean(axis=1))
    return np.array(mel_features)

def train_and_evaluate(X, labels, model, model_name="", random_state=42, test_size=0.2):
    """
    通用訓練與評估函數
    Returns: accuracy, report_dict, training_time
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    
    print(f"{model_name} Accuracy: {accuracy:.4f} (Time: {train_time:.2f}s)")
    
    return accuracy, report, train_time, le, model  # 順便回傳 le 和 fitted model（後續可能用於全資料預測）

# === 各模型專用函數 ===
def mfcc_svm_rf_classify(segments, labels):
    """MFCC + SVM 和 RF"""
    mfcc_X = extract_mfcc_features(segments)
    
    svm_accuracy, svm_report, svm_time, le, _ = train_and_evaluate(
        mfcc_X, labels, SVC(kernel='rbf', random_state=42), "SVM (MFCC)"
    )
    rf_accuracy, rf_report, rf_time, _, _ = train_and_evaluate(
        mfcc_X, labels, RandomForestClassifier(n_estimators=100, random_state=42), "RF (MFCC)"
    )

    return svm_accuracy, rf_accuracy, svm_report, rf_report, svm_time, rf_time

def dt_shapelet_classify(shapelet_features, labels):
    """Shapelet + DecisionTree"""
    return train_and_evaluate(
        shapelet_features, labels,
        DecisionTreeClassifier(random_state=42),
        model_name="DT (Shapelet)"
    )[:3]  # 只回傳 accuracy, report, time

def rf_shapelet_classify(shapelet_features, labels, n_estimators=200):
    """Shapelet + RandomForest"""
    return train_and_evaluate(
        shapelet_features, labels,
        RandomForestClassifier(n_estimators=n_estimators, random_state=42),
        model_name="RF (Shapelet)"
    )[:3]

def rf_mel_classify(segments, labels, n_estimators=100):
    """Mel Spectrogram + RandomForest"""
    mel_X = extract_mel_features(segments)
    return train_and_evaluate(
        mel_X, labels,
        RandomForestClassifier(n_estimators=n_estimators, random_state=42),
        model_name="RF (MelSpec)"
    )[:3]


def plot_confusion_matrix(y_true, y_pred, labels_list, title, output_path):
    """
    繪製並儲存混淆矩陣
    
    Parameters:
    - y_true: 真實標籤 (list or array)
    - y_pred: 預測標籤 (list or array)
    - labels_list: 標籤名稱列表，例如 ['brushing', 'waiting for food', 'isolation']
    - title: 圖表標題
    - output_path: 儲存路徑
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_list, yticklabels=labels_list,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Success] Saved confusion matrix: {output_path}")