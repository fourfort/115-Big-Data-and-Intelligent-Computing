# simple_shapelet_visualize.py - 最終版（每環境 1 張圖，藍線原波形 + 紅線 shapelet 區域）

import numpy as np
import matplotlib.pyplot as plt

def visualize_simple_shapelets(segments, labels, n_shapelets_per_class=1, sub_length=400):
    print(f"提取每類 1 個代表樣本，並繪製 {n_shapelets_per_class} 個 shapelet 區域（紅線覆蓋）...")
    
    unique_labels = np.unique(labels)
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11
    
    for label in unique_labels:
        class_idx = [i for i, l in enumerate(labels) if l == label]
        if not class_idx:
            continue
        
        # 選 1 個代表樣本
        rep_idx = np.random.choice(class_idx)
        rep_seg = np.array(segments[rep_idx], dtype=float)
        
        # 正規化
        rep_norm = (rep_seg - rep_seg.mean()) / (rep_seg.std() + 1e-8)
        
        seg_len = len(rep_norm)
        if seg_len < sub_length:
            continue
        
        # 提取 shapelet 位置（均勻分布）
        starts = np.linspace(0, seg_len - sub_length, n_shapelets_per_class, dtype=int)
        
        # 產生 1 張圖
        fig, ax = plt.subplots(figsize=(14, 5))
        
        time_axis = np.arange(seg_len)
        
        # 畫整個藍色原波形
        ax.plot(time_axis, rep_norm, color='blue', linewidth=1.5, label='Normalized Waveform')
        
        # 對每個 shapelet 區域覆蓋紅線
        for i, start in enumerate(starts):
            end = start + sub_length
            ax.plot(time_axis[start:end], rep_norm[start:end], color='red', linewidth=2, label=f'Shapelet {i+1} Region')
        
        ax.set_title(f'Representative Waveform with Shapelet Regions - {label}\n'
                     f'(Blue = Full Waveform, Red = Shapelet Areas)')
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Normalized Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"charts/{label.replace(' ', '_')}_waveform_comparison.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[Success] Saved: {filename}")
    
    print("所有 3 張對齊对比圖已完成！")