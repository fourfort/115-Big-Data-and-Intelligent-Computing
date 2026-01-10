# mine_association_rules.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import Binarizer

def mine_rules(shapelet_features, labels, annotated_shapelets=None, 
               min_support=0.03, min_confidence=0.5, binarize_threshold=0.3):
    print("\n=== 開始關聯規則探勘===")
    # 1. 檢查每個環境的樣本數
    label_counts = pd.Series(labels).value_counts()
    print(f"\n環境樣本分佈:\n{label_counts}")
    # 2. 二值化 Shapelet 特徵
    print(f"\n二值化 Shapelet 特徵 (threshold={binarize_threshold})...")
    binarizer = Binarizer(threshold=binarize_threshold)
    binary_matrix = binarizer.fit_transform(shapelet_features)
    
    # 統計每個 shapelet 的激活率
    activation_rates = binary_matrix.mean(axis=0)
    print(f"Shapelet 激活率範圍: {activation_rates.min():.3f} ~ {activation_rates.max():.3f}")
    
    # 3. 建立交易矩陣
    df = pd.DataFrame(binary_matrix, 
                      columns=[f'shapelet_{i}' for i in range(binary_matrix.shape[1])])
    
    # 4. 加入 shapelet 類型（在 extract_shapelets時提取的語義標籤）
    if annotated_shapelets is not None and len(annotated_shapelets) > 0:
        print("\n加入 Shapelet 類型描述...")
        # 將註記轉為簡化標籤
        shapelet_types = []
        for desc in annotated_shapelets:
            if 'high' in desc.lower() or 'sharp' in desc.lower() or 'piercing' in desc.lower():
                shapelet_types.append('high_freq')
            elif 'low' in desc.lower() or 'soft' in desc.lower() or 'gentle' in desc.lower():
                shapelet_types.append('low_freq')
            else:
                shapelet_types.append('mid_freq')   
        # One-hot encoding
        type_df = pd.get_dummies(shapelet_types, prefix='type')
        df = pd.concat([df, type_df], axis=1) 
    # 5. 加入情境標籤
    context_df = pd.get_dummies(labels, prefix='context')
    df = pd.concat([df, context_df], axis=1)
    # 轉為布林值
    df = df.astype(bool)
    print(f"\n交易矩陣形狀: {df.shape}")
    print(f"特徵類型統計:")
    print(f"  - Shapelet 特徵: {binary_matrix.shape[1]}")
    if annotated_shapelets:
        print(f"  - Shapelet 類型: {len(type_df.columns)}")
    print(f"  - 情境標籤: {len(context_df.columns)}")
    # 6. 挖掘頻繁項集（多策略）
    all_rules = [] 
    # 策略 1: 標準探勘
    print(f"\n策略 1: 標準探勘 (support={min_support}, confidence={min_confidence})...")
    try:
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        if not frequent_itemsets.empty:
            rules = association_rules(frequent_itemsets, metric="confidence", 
                                     min_threshold=min_confidence)
            all_rules.append(rules)
            print(f"  發現 {len(rules)} 條規則")
    except Exception as e:
        print(f"  策略 1 失敗: {e}")

    # # 策略 2: 降低支持度（針對少數樣本環境）
    # min_support_low = max(0.01, 2 / len(labels))  # 至少 2 個樣本
    # if min_support_low < min_support:
    #     print(f"\n策略 2: 降低支持度探勘 (support={min_support_low:.3f})...")
    #     try:
    #         frequent_itemsets_low = apriori(df, min_support=min_support_low, use_colnames=True)
    #         if not frequent_itemsets_low.empty:
    #             rules_low = association_rules(frequent_itemsets_low, metric="confidence", 
    #                                          min_threshold=min_confidence * 0.8)  # 稍微放寬
    #             all_rules.append(rules_low)
    #             print(f"  發現 {len(rules_low)} 條規則")
    #     except Exception as e:
    #         print(f"  策略 2 失敗: {e}")

    # # 策略 2: 針對每個環境單獨探勘
    print(f"\n策略 2: 環境特定探勘...")
    for env in label_counts.index:
        env_mask = np.array(labels) == env
        df_env = df[env_mask].copy()
        # 只保留該環境的 context 欄位
        env_col = f'context_{env}'
        if env_col not in df_env.columns:
            continue    
        support_env = max(0.3, 3 / env_mask.sum())  # 環境內至少 30% 或 3 個樣本 
        try:
            freq_env = apriori(df_env, min_support=support_env, use_colnames=True)
            if not freq_env.empty:
                rules_env = association_rules(freq_env, metric="confidence", min_threshold=0.6)
                # 過濾：前項是 shapelet，後項是該環境
                rules_env = rules_env[
                    rules_env['antecedents'].apply(lambda x: any('shapelet' in str(i) for i in x)) &
                    rules_env['consequents'].apply(lambda x: env_col in x)]
                all_rules.append(rules_env)
                print(f"  {env}: 發現 {len(rules_env)} 條規則")
        except Exception as e:
            print(f"  {env} 探勘失敗: {e}")
    # 7. 合併所有規則並去重
    if not all_rules:
        print("\n[Warning] 未發現任何規則！")
        return pd.DataFrame()  
    combined_rules = pd.concat(all_rules, ignore_index=True)
    # 8. 過濾規則：確保前項有 shapelet，後項有 context
    combined_rules = combined_rules[combined_rules['antecedents'].apply(lambda x: any('shapelet' in str(item) for item in x)) &
        combined_rules['consequents'].apply(lambda x: any('context' in str(item) for item in x))]
    # 9. 去除重複規則（基於 antecedents 和 consequents）
    combined_rules['rule_str'] = combined_rules.apply(
        lambda row: str(sorted(row['antecedents'])) + ' -> ' + str(sorted(row['consequents'])), axis=1)
    combined_rules = combined_rules.drop_duplicates(subset=['rule_str']).drop(columns=['rule_str'])
    # 10. 排序並確保每個環境至少有一條規則
    combined_rules = combined_rules.sort_values(['lift', 'confidence'], ascending=False)
    
    # 檢查覆蓋的環境
    covered_envs = set()
    for _, row in combined_rules.iterrows():
        for item in row['consequents']:
            if 'context_' in str(item):
                env_name = str(item).replace('context_', '')
                covered_envs.add(env_name)
    
    print(f"\n=== 探勘完成 ===")
    print(f"總共發現 {len(combined_rules)} 條規則")
    print(f"涵蓋環境: {covered_envs}")
    
    missing_envs = set(label_counts.index) - covered_envs
    if missing_envs:
        print(f"[Warning] 以下環境未產生規則: {missing_envs}")
        print("建議：")
        print("  1. 降低 min_support 或 min_confidence")
        print("  2. 調整 binarize_threshold（當前 {})".format(binarize_threshold))
        print("  3. 檢查該環境的 shapelet 特徵是否顯著")
    
    return combined_rules


def visualize_rules_table(rules, output_path="charts/association_rules_table.png", title_suffix=""):
    """
    視覺化關聯規則表格（支援自訂檔名與標題）
    
    Parameters:
    - rules: mlxtend 產生的 rules DataFrame
    - output_path: 儲存路徑與檔名
    - title_suffix: 標題後綴，例如 "DT (Shapelet) Predictions"
    """
    if rules.empty:
        print(f"[Warning] 無規則可視覺化 ({output_path})")
        return
    display_rules = rules.copy()
    # 簡化顯示：所有 shapelet 統一為 "shapelet"
    def simplify_frozenset(fs):
        seen_shapelet = False
        items = []
        for item in fs:
            s = str(item)
            if 'context_' in s:
                items.append(s.replace('context_', ''))
            elif 'shapelet_' in s:
                if not seen_shapelet:
                    items.append('shapelet')
                    seen_shapelet = True
            else:
                items.append(s)
        return ', '.join(items)
    
    display_rules['antecedents'] = display_rules['antecedents'].apply(simplify_frozenset)
    display_rules['consequents'] = display_rules['consequents'].apply(simplify_frozenset)

    # 格式化與排序
    display_rules['support'] = display_rules['support'].round(3)
    display_rules['confidence'] = display_rules['confidence'].round(3)
    display_rules['lift'] = display_rules['lift'].round(2)
    display_rules = display_rules.sort_values('support', ascending=False).reset_index(drop=True)
    n_rules = len(display_rules)
    fig, ax = plt.subplots(figsize=(18, max(5, n_rules * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].values.tolist(),
                     colLabels=['Antecedents', 'Consequents', 'Support', 'Confidence', 'Lift'],
                     loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=[0,1,2,3,4])
    table.scale(1.5, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('gray')
        cell.set_linewidth(0.5)
        if row == 0:  # 標題列加粗
            cell.set_text_props(weight='bold')
    base_title = f"Association Rules (Sorted by Support, Total {n_rules} Rules)"
    title = base_title if not title_suffix else f"Association Rules - {title_suffix} (Total {n_rules} Rules)"
    plt.title(title, fontsize=14, pad=20)
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Success] Saved: {output_path}")


        

