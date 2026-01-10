# main.py
import os
import librosa
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from preprocess_audio import process_dataset
from extract_shapelets import extract_shapelet_features
from mine_association_rules import visualize_rules_table
from mine_association_rules import mine_rules
from baseline_classification import (mfcc_svm_rf_classify,dt_shapelet_classify,rf_shapelet_classify,rf_mel_classify,plot_confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

def load_and_label_dataset(data_path=r'cat_vocalization_dataset'):
    """載入所有 .wav 檔，從檔名提取情境標籤 C (B/F/I),以回傳: raw_audio, raw_srs, labels, df_info"""
    files = [f for f in os.listdir(data_path) if f.lower().endswith('.wav')]
    if not files:
        raise FileNotFoundError(f"資料夾 {data_path} 中無 .wav 檔")

    raw_audio = []
    raw_srs = []
    labels = []
    filenames = []

    print(f"載入 {len(files)} 個音檔並提取標籤...")
    for filename in files:
        filepath = os.path.join(data_path, filename)

        # 提取 C (第一個字母 B/F/I)
        match = re.match(r'([BFI])_', filename.upper())
        if not match:
            print(f"[Warning] 跳過無效檔名: {filename}")
            continue
        context = match.group(1)
        label_map = {'B': 'brushing', 'F': 'waiting for food', 'I': 'isolation'}
        label = label_map.get(context, 'unknown')

        y, sr = librosa.load(filepath, sr=None)
        raw_audio.append(y)
        raw_srs.append(sr)
        labels.append(label)
        filenames.append(filename)

    df_info = pd.DataFrame({'filename': filenames, 'context': labels})
    print("標籤分佈:")
    print(df_info['context'].value_counts())
    return raw_audio, raw_srs, labels, df_info

if __name__ == "__main__":
    data_path = r'dataset'
    # 1. 載入資料
    raw_audio, raw_srs, raw_labels, info_df = load_and_label_dataset(data_path)
    # 2. 前處理
    print("\n開始前處理與片段切割...")
    try:
        segments, segment_labels = process_dataset(raw_audio, raw_srs, raw_labels, use_preemphasis=True)
    except TypeError:
        print("[Info] fallback 到舊版 process_dataset")
        segments, segment_labels = process_dataset(raw_audio, raw_labels)
    labels = segment_labels
    print(f"成功提取 {len(segments)} 個貓叫片段")

    # 3. 監督式 Shapelet 提取
    print("\n提取 Shapelet 特徵（監督式）...")
    shapelet_features, shapelets, le, annotated_shapelets = extract_shapelet_features(
        segments, labels, target_length=2000
    )
    
# ==================== 分類模型訓練 ====================
    print("\n執行 MFCC Baseline 分類...")
    svm_accuracy, rf_mfcc_accuracy, svm_report, rf_mfcc_report, _, _ = mfcc_svm_rf_classify(segments, labels)

    dt_accuracy, dt_report, dt_time = dt_shapelet_classify(shapelet_features, labels)

    rf_sh_accuracy, rf_sh_report, rf_sh_time = rf_shapelet_classify(shapelet_features, labels, n_estimators=200)

    mel_accuracy, mel_report, mel_time = rf_mel_classify(segments, labels, n_estimators=100)

    # ==================== 找出整體最佳模型 ====================
    model_performances = {
        'SVM (MFCC)': svm_accuracy,
        'RF (MFCC)': rf_mfcc_accuracy,
        'DT (Shapelet)': dt_accuracy,
        'RF (MelSpec)': mel_accuracy,
        'RF (Shapelet)': rf_sh_accuracy
    }
    best_model_name = max(model_performances, key=model_performances.get)
    print(f"\n[Info] 整體最佳模型: {best_model_name} (Accuracy: {model_performances[best_model_name]:.4f})")

    # ==================== 取得各模型對全部樣本的預測標籤（用於關聯規則） ====================
    # DT (Shapelet)
    dt_full = DecisionTreeClassifier(random_state=42)
    dt_full.fit(shapelet_features, le.transform(labels))
    pred_dt = le.inverse_transform(dt_full.predict(shapelet_features)) # 將數值轉回標籤

    # RF (Shapelet)
    rf_sh_full = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_sh_full.fit(shapelet_features, le.transform(labels))
    pred_rf_sh = le.inverse_transform(rf_sh_full.predict(shapelet_features)) # 將數值轉回標籤

    # 最佳模型預測（簡化處理，實際可擴展）
    pred_best = pred_rf_sh if best_model_name == 'RF (Shapelet)' else pred_dt

    rule_configs = [
        ("DT (Shapelet) Predictions", pred_dt, "charts/association_rules_dt_shapelet.png"),
        ("RF (Shapelet) Predictions", pred_rf_sh, "charts/association_rules_rf_shapelet.png"),
        (f"Best Model ({best_model_name}) Predictions", pred_best, f"charts/association_rules_best_{best_model_name.replace(' ', '_').lower()}.png")
    ]

    for title_suffix, pred_labels, output_path in rule_configs:
        print(f"\n挖掘關聯規則（{title_suffix}）...")
        rules = mine_rules(
            shapelet_features, pred_labels.tolist(), annotated_shapelets,
            min_support=0.008, min_confidence=0.35, binarize_threshold=0.25
        )
        if not rules.empty:
            visualize_rules_table(rules, output_path, title_suffix)
        else:
            print(f"無規則產生 ({title_suffix})")
        
# ====視覺化 Shapelets 匹配位置圖====
    print("\n產生真實 Shapelets 匹配位置圖（每環境 1 張）...")
    unique_labels = np.unique(labels)
    sr_example = 44100  # 假設統一採樣率
    # Shapelets 與音訊位置比對 
    # 深度檢查 shapelets 的結構
    print(f"[Debug] shapelets 型態: {type(shapelets)}, 長度: {len(shapelets)}")
    if len(shapelets) > 0:
        print(f"[Debug] shapelets[0] 型態: {type(shapelets[0])}, shape: {shapelets[0].shape}")
        print(f"[Debug] shapelets[0][0] 型態: {type(shapelets[0][0])}")
        if hasattr(shapelets[0][0], 'shape'):
            print(f"[Debug] shapelets[0][0] shape: {shapelets[0][0].shape}")
        else:
            print(f"[Debug] shapelets[0][0] 值: {shapelets[0][0]}")

    for label in unique_labels:
        try:
            # 隨機選一個該類別的片段作為代表
            class_indices = [i for i, l in enumerate(labels) if l == label]
            rep_idx = np.random.choice(class_indices)
            
            # 確保 audio_seg 是乾淨的 1D numpy array
            audio_seg = np.array(segments[rep_idx]).flatten().astype(np.float64)
            
            if len(audio_seg) == 0:
                print(f"[Warning] Empty audio segment for {label}, skipping...")
                continue
            
            print(f"[Info] Processing {label}: audio length = {len(audio_seg)}")
            
            fig, ax = plt.subplots(figsize=(16, 5))
            time_axis = np.arange(len(audio_seg))
            ax.plot(time_axis, audio_seg, color='blue', linewidth=1.2, label='Raw Waveform', alpha=0.8)

            # 計算每個 shapelet 的最小距離
            min_global_dist = np.inf
            best_shapelet_idx = 0
            best_start = 0
            best_len_sh = 0
            
            for j in range(len(shapelets)):
                try:
                    raw_shapelet = shapelets[j]
                    # 處理 shape=(2,) 的情況，其中每個元素是一個陣列
                    if raw_shapelet.shape == (2,):
                        # 檢查第一個元素是否是陣列
                        if isinstance(raw_shapelet[0], np.ndarray):
                            # 將兩個陣列連接起來
                            shapelet = np.concatenate([raw_shapelet[0].flatten(), raw_shapelet[1].flatten()])
                        else:
                            # 如果是純數值，直接使用
                            shapelet = raw_shapelet.flatten()
                    else:
                        # 其他情況，遞迴 flatten
                        def recursive_flatten(arr):
                            result = []
                            for item in arr:
                                if isinstance(item, (np.ndarray, list)):
                                    result.extend(recursive_flatten(item))
                                else:
                                    result.append(item)
                            return result
                        
                        shapelet = np.array(recursive_flatten(raw_shapelet))
                    
                    # 確保是 float64
                    shapelet = shapelet.astype(np.float64)
                    
                    print(f"[Debug] Shapelet {j} 處理後長度: {len(shapelet)}")
                    
                    if len(shapelet) == 0:
                        print(f"[Warning] Shapelet {j} 長度為 0")
                        continue
                    
                    # 限制 shapelet 長度
                    len_sh = min(len(shapelet), len(audio_seg))
                    shapelet = shapelet[:len_sh]
                    
                    if len_sh < 2:
                        print(f"[Warning] Shapelet {j} 太短: {len_sh}")
                        continue
               
                except Exception as e:
                    print(f"[Warning] Cannot process shapelet {j}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # 滑動窗口搜索
                local_min_dist = np.inf
                local_best_start = 0
                
                # 使用較大的步長加速
                step_size = max(1, len_sh // 20)
                
                for start in range(0, len(audio_seg) - len_sh + 1, step_size):
                    window = audio_seg[start:start + len_sh].copy()
                    
                    if len(window) != len(shapelet):
                        continue
                    
                    try:
                        diff = window - shapelet
                        dist = np.sqrt(np.sum(diff * diff))
                        
                        if dist < local_min_dist:
                            local_min_dist = dist
                            local_best_start = start
                    except Exception as e:
                        continue

                # 更新全局最佳
                if local_min_dist < min_global_dist and local_min_dist != np.inf:
                    min_global_dist = local_min_dist
                    best_shapelet_idx = j
                    best_start = local_best_start
                    best_len_sh = len_sh
                    print(f"[Debug] 找到更好的匹配: shapelet {j}, dist={local_min_dist:.2f}")

            # 檢查是否找到有效匹配
            if min_global_dist == np.inf or best_len_sh == 0:
                print(f"[Warning] No valid shapelet match found for {label}, skipping plot...")
                plt.close()
                continue

            print(f"[Success] 最佳匹配: shapelet {best_shapelet_idx}, start={best_start}, length={best_len_sh}, dist={min_global_dist:.2f}")

            # 繪製最佳匹配區域
            end = min(best_start + best_len_sh, len(audio_seg))
            ax.plot(time_axis[best_start:end], audio_seg[best_start:end],
                    color='red', linewidth=3, label='Most Important Shapelet Match')
            # 在 ax.plot 後加這行
            ax.set_ylim(audio_seg.min() * 1.1, audio_seg.max() * 1.1)  # 放大振幅範圍
            # 在塗紅後加
            ax.fill_between(time_axis[best_start:end], audio_seg[best_start:end], color='red', alpha=0.3)

            # 添加註記
            description = annotated_shapelets[best_shapelet_idx] if best_shapelet_idx < len(annotated_shapelets) else f'Shapelet {best_shapelet_idx+1}'
 
        # === 雙圖：時域 + 頻域 ===
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.3)
            # 上：時域
            ax1 = fig.add_subplot(gs[0])
            time_axis = np.arange(len(audio_seg))
            ax1.plot(time_axis, audio_seg, color='blue', linewidth=1.2, label='Waveform')
            end = min(best_start + best_len_sh, len(audio_seg))
            ax1.plot(time_axis[best_start:end], audio_seg[best_start:end], color='red', linewidth=3, label='Best Shapelet Match')
            ax1.fill_between(time_axis[best_start:end], audio_seg[best_start:end], color='red', alpha=0.3)
            ax1.set_title(f'{label} - Time Domain with Discriminative Shapelet Match')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(alpha=0.3)
            # 下：頻域 Mel Spectrogram
            ax2 = fig.add_subplot(gs[1])
            S = librosa.feature.melspectrogram(y=audio_seg, sr=sr_example, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr_example, fmax=8000, ax=ax2)
            fig.colorbar(img, ax=ax2, format='%+2.0f dB')
            # 標記 shapelet 匹配時間區間
            t_start = best_start / sr_example
            t_end = end / sr_example
            ax2.axvspan(t_start, t_end, color='red', alpha=0.4, label='Best Shapelet Match Region')
            ax2.legend()
            mid_t = (t_start + t_end) / 2
            ax2.annotate(f"{description}\n(dist={min_global_dist:.1f})",
                         xy=(mid_t, S_dB.max() * 0.8), xytext=(0, 10),
                         textcoords='offset points', ha='center', color='white',
                         fontsize=11, bbox=dict(boxstyle="round,pad=0.5", fc="red", alpha=0.8))
            ax2.set_title(f'{label} - Frequency Domain (Mel Spectrogram)')
            ax2.set_xlabel('Time (s)')
            plt.suptitle(f'Representative Vocalization: {label}\nRed Region = Most Discriminative Shapelet Match', fontsize=16, y=0.98)
            filename = f"charts/{label.replace(' ', '_')}_shapelet_match_time_freq.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[Success] Saved dual-domain plot: {filename}")
        except Exception as e:
            print(f"[Error] {label} plot failed: {e}")
            continue
    print("\n所有最重要的 Shapelet 匹配圖已完成！")

# # ==================== PCA 散佈圖 ====================
#     print("\n產生 PCA 散佈圖...")
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(shapelet_features)
#     pca = PCA(n_components=2, random_state=42)
#     X_pca = pca.fit_transform(X_scaled)
#     plt.figure(figsize=(10, 8))
#     colors = {'brushing': 'green', 'waiting for food': 'orange', 'isolation': 'red'}
#     for lbl in np.unique(labels):
#         idx = np.array(labels) == lbl
#         plt.scatter(X_pca[idx, 0], X_pca[idx, 1], c=colors[lbl], label=f'{lbl} (n={idx.sum()})', alpha=0.8, s=60)
#     plt.title('PCA Projection of Samples in Shapelet Feature Space')
#     plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
#     plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
#     plt.legend(title='Context')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig("charts/shapelet_pca_scatter.png", dpi=300)
#     plt.close()
#     print("[Success] Saved PCA scatter plot: charts/shapelet_pca_scatter.png")

    # ==================== 多指標比較表 ====================
    def extract_metrics(report, acc):
        if report and 'macro avg' in report:
            macro = report['macro avg']
            return {
                'Accuracy': round(report.get('accuracy', acc), 3),
                'Precision': round(macro['precision'], 3),
                'Recall': round(macro['recall'], 3),
                'F1-score': round(macro['f1-score'], 3)
            }
        return {'Accuracy': round(acc, 3), 'Precision': '-', 'Recall': '-', 'F1-score': '-'}

    comparison = {
        'SVM (MFCC)': extract_metrics(svm_report, svm_accuracy),
        'RF (MFCC)': extract_metrics(rf_mfcc_report, rf_mfcc_accuracy),
        'DT (Shapelet)': extract_metrics(dt_report, dt_accuracy),
        'RF (MelSpec)': extract_metrics(mel_report, mel_accuracy),
        'RF (Shapelet)': extract_metrics(rf_sh_report, rf_sh_accuracy),
    }
    df_comp = pd.DataFrame.from_dict(comparison, orient='index')
    df_comp['Time (s)'] = [0.0, 0.0, round(dt_time, 2), round(mel_time, 2), round(rf_sh_time, 2)]
    df_comp = df_comp[['Accuracy', 'Precision', 'Recall', 'F1-score']]  # 移除 Time
    os.makedirs("data", exist_ok=True)
    df_comp.to_csv("data/model_comparison_detailed.csv")
    print("\n=== Model Comparison (Macro Avg) ===")
    print(df_comp)

# === 表格圖片呈現（左上角加入 "Techniques" 標題 + 最大值紅色標記）===
    print("\n產生模型比較表格圖片（左上角為 Techniques）...")
    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')

    # 準備顯示資料
    display_df = df_comp.copy()
    display_df = display_df.round(3)
    # 轉為字串列表（內容）
    table_data = display_df.astype(str).values.tolist()
    # 欄位標題與模型名稱
    columns = display_df.columns.tolist()
    rows = display_df.index.tolist()
    # 建立表格（包含左上角標題）
    table = ax.table(cellText=table_data,colLabels=columns,rowLabels=rows,loc='center',cellLoc='center',colLoc='center',bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.4)
    # 清除自動生成的 rowLabels 外觀，改為手動控制
    for i in range(len(rows)):
        table[(i+1, -1)].visible_edges = ''  # 隱藏左側邊框（視覺上融入）
    # 將 "Techniques" 作為第一欄標題，並將模型名稱移到第一欄內容 ===
    # 重構表格資料,以及重新建立表格
    new_table_data = []
    for i, row_name in enumerate(rows):
        row = [row_name] + display_df.iloc[i].astype(str).tolist()
        new_table_data.append(row)
    new_columns = ['Techniques'] + columns
    table = ax.table(cellText=new_table_data,colLabels=new_columns,loc='center',cellLoc='center',colLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.4)
    # === 樣式設定 ===
    num_cols = len(new_columns)
    num_rows = len(rows)
    # 標題列（第一列）：深藍底白字粗體
    for j in range(num_cols):
        cell = table[0, j]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
        cell.set_height(0.1)

    # 內容格子：交替背景 + 最大值紅色標記
    highlight_cols = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    col_indices = {col: idx for idx, col in enumerate(new_columns)}
    for col_name in highlight_cols:
        if col_name in col_indices:
            col_idx = col_indices[col_name]
            valid_vals = [float(display_df.iloc[r][col_name]) for r in range(num_rows) 
                         if display_df.iloc[r][col_name] != '-']
            if valid_vals:
                max_val = max(valid_vals)
                for r in range(num_rows):
                    cell_val = display_df.iloc[r][col_name]
                    if cell_val != '-' and float(cell_val) == max_val:
                        cell = table[r + 1, col_idx]
                        cell.set_text_props(weight='bold', color='red', fontsize=12)
    # 交替行顏色 + 邊框
    for i in range(1, num_rows + 1):
        for j in range(num_cols):
            cell = table[i, j]
            cell.set_linewidth(1.2)
            cell.set_edgecolor('gray')
            if i % 2 == 0:
                cell.set_facecolor('#F8F8F8')
            else:
                cell.set_facecolor('white')
    # Techniques 欄（第一欄）特別加粗模型名稱
    for i in range(1, num_rows + 1):
        cell = table[i, 0]
        cell.set_text_props(weight='bold', color='black')
    plt.title('Model Performance Comparison (Macro Average)', 
              fontsize=16, weight='bold', pad=30)
    plt.tight_layout()

    table_img_path = "charts/model_comparison_table.png"
    plt.savefig(table_img_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Success] Saved enhanced comparison table with 'Techniques' header: {table_img_path}")

    # === 雙軸長條圖 + 黑色標註 + 處理小數值與標註重疊 ===
    print("產生雙軸比較長條圖（含數值標註）...")

    # 建立包含 Time 的 DataFrame（專門用於此圖）
    df_bar = pd.DataFrame({
        'Accuracy': [svm_accuracy, rf_mfcc_accuracy, dt_accuracy, mel_accuracy, rf_sh_accuracy],
        'Time (s)': [0.0, 0.0, round(dt_time, 4), round(mel_time, 4), round(rf_sh_time, 4)]
    }, index=['SVM (MFCC)', 'RF (MFCC)', 'DT (Shapelet)', 'RF (MelSpec)', 'RF (Shapelet)'])

    fig, ax1 = plt.subplots(figsize=(13, 7))  # 稍微加寬，避免最右邊擠壓

    # Accuracy 長條（左軸）
    bars1 = ax1.bar(df_bar.index, df_bar['Accuracy'], color='skyblue', label='Accuracy', alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=12, color='black')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y', labelcolor='black')

    # Time 長條（右軸）
    ax2 = ax1.twinx()
    bars2 = ax2.bar(df_bar.index, df_bar['Time (s)'], color='orange', alpha=0.6, width=0.4, label='Time (s)')
    ax2.set_ylabel('Training + Inference Time (s)', fontsize=12, color='black')
    max_time = df_bar['Time (s)'].max()
    ax2.set_ylim(0, max(max_time * 1.3, 0.1))  # 至少留一點空間給小值標註
    ax2.tick_params(axis='y', labelcolor='black')

    # === Accuracy 標註 ===
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    # === Time 標註（重點優化：處理小數值與避免重疊）===
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        model_name = df_bar.index[i]
        
        # 顯示到小數點第4位（即使是0）
        text = f'{height:.4f}'
        
        # 判斷是否為小值（< 0.05），若是的話把標註往上移到圖表內部上方
        if height < 0.05:
            y_pos = max_time * 0.15 if max_time > 0 else 0.02  # 放在圖表下部上方
            va = 'bottom'
        else:
            y_pos = height + max_time * 0.02
            va = 'bottom'
        
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                text, ha='center', va=va, fontsize=10, fontweight='bold', color='black')

    # 標題與圖例
    ax1.set_title('Model Comparison: Performance vs Computation Time', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.xticks(rotation=20, ha='right')
    plt.grid(True, alpha=0.3, axis='y', which='major')
    plt.tight_layout()

    bar_img_path = "charts/model_comparison_bar_labeled.png"
    plt.savefig(bar_img_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Success] Saved labeled bar chart: {bar_img_path}")

# ==================== 混淆矩陣 ====================
    print("\n產生混淆矩陣...")

    # 重新 split 取得測試集（確保標籤對齊）
    _, X_test, _, y_test_true = train_test_split(shapelet_features, labels, test_size=0.2, random_state=42, stratify=labels)

    # 取得測試集預測
    rf_sh_pred_test = le.inverse_transform(rf_sh_full.predict(X_test))
    best_pred_test = rf_sh_pred_test if best_model_name == 'RF (Shapelet)' else le.inverse_transform(dt_full.predict(X_test))

    unique_labels = sorted(set(labels))

    plot_confusion_matrix(
        y_test_true, best_pred_test, unique_labels,
        f'Confusion Matrix - Best Model ({best_model_name})',
        'charts/confusion_matrix_best.png'
    )
    plot_confusion_matrix(
        y_test_true, rf_sh_pred_test, unique_labels,
        'Confusion Matrix - RF (Shapelet)',
        'charts/confusion_matrix_rf_shapelet.png'
    )

    print("\n=== 所有任務完成！已產生：")
    print("   - 3 張關聯規則表格")
    print("   - 模型比較表（無 Time）")
    print("   - 雙軸圖（含 Time）")
    print("   - 2 張混淆矩陣")
    print("\n=== 所有任務完成！ .csv 存於 data/，圖表存於 charts/ ===")